"""
BrownCSR unified pipeline.

Chains DepthAnything → DINO segmentation → neural CAPTRA pose → FeaturePoseTracker.

Primary entry points:
  run_sequence_tracked  — image folder / list → FeaturePoseTracker
  run_video_tracked     — video file → FeaturePoseTracker

Old geometric (PCA-based) pipeline functions are archived in exploreCSR/pose/old/pipeline_geometric.py.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

from .config import CAPTRAConfig, PipelineConfig
from .depth import DepthAnything
from .segmentation import DINOSegmenter
from .pose import CAPTRA
from .visualization import print_pose_summary


# ── shared helpers ─────────────────────────────────────────────────────────────

def _load_rgb(image_path: str) -> np.ndarray:
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _show_preview(
    rgb: np.ndarray,
    mask: np.ndarray,
    captra_out: dict,
    frame_idx: int,
    window_name: str = "Pipeline Preview",
) -> None:
    """Overlay mask + pose text on the frame and display via cv2.imshow."""
    vis = rgb.astype(np.float32) / 255.0
    if mask is not None and mask.shape == rgb.shape[:2]:
        overlay = vis.copy()
        overlay[mask.astype(bool)] = overlay[mask.astype(bool)] * 0.5 + np.array([1.0, 0.0, 0.0]) * 0.5
        vis = overlay
    vis = (np.clip(vis, 0, 1) * 255).astype(np.uint8)
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    t = captra_out.get("translation", [0, 0, 0])
    valid = captra_out.get("valid", False)
    label = (
        f"Frame {frame_idx}  {'OK' if valid else 'INVALID'}"
        f"  t=[{t[0]:.3f},{t[1]:.3f},{t[2]:.3f}]"
    )
    cv2.putText(vis_bgr, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 0) if valid else (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow(window_name, vis_bgr)
    cv2.waitKey(1)


def _ensure_size(rgb: np.ndarray, w: int, h: int) -> np.ndarray:
    if rgb.shape[0] != h or rgb.shape[1] != w:
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    return rgb


# ── tracked sequence pipeline ──────────────────────────────────────────────────

def run_sequence_tracked(
    image_paths: List[str],
    weights_path: str,
    prompt: str,
    rot_dir: str,
    coord_dir: str,
    category: int,
    intrinsics: Optional[np.ndarray] = None,
    num_points: int = 4096,
    feature_key: str = "object_mean",
    verbose: bool = True,
    show_preview: bool = False,
) -> "FeaturePoseTracker":
    """
    Run the full tracked pipeline on an ordered list of images.

    For each frame:
      DepthAnything → depth_norm
      DINO segmentation → object mask
      Neural CAPTRA → pose (translation, rotation, scale)
      DINO features → object_mean, cls_token patch features
      FeaturePoseTracker records all of the above

    Parameters
    ----------
    image_paths : ordered list of image file paths
    weights_path: DepthAnything checkpoint (.pth)
    prompt      : text prompt describing the object for DINO segmentation
    rot_dir     : path to CAPTRA rotation net checkpoint dir
    coord_dir   : path to CAPTRA coordinate net checkpoint dir
    category    : NOCS category index (1=bottle 2=bowl 3=camera 4=can 5=laptop 6=mug)
    intrinsics  : (3, 3) camera K; defaults to NOCS real camera if None
    num_points  : point cloud size per frame
    feature_key : which DINO feature to track ("object_mean" or "cls_token")
    verbose     : print progress

    Returns
    -------
    FeaturePoseTracker with all frames recorded — call .plot(), .to_csv(), etc.
    """
    from .combination import FeaturePoseTracker, extract_object_features

    if not image_paths:
        raise ValueError("image_paths is empty.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    converter = DepthAnything(weights_path)
    segmenter = DINOSegmenter()
    captra    = CAPTRA(
        rot_dir   = rot_dir,
        coord_dir = coord_dir,
        category  = category,
        device    = device,
        intrinsics= intrinsics,
        num_points= num_points,
    )
    tracker = FeaturePoseTracker(feature_key=feature_key)

    if verbose:
        print(f"[PIPELINE] {len(image_paths)} frames, prompt={prompt!r}, category={category}")

    for idx, path in enumerate(image_paths):
        if verbose:
            print(f"\n--- Frame {idx}: {os.path.basename(path)} ---")

        image_tensor, _, w, h = converter.preprocess(path)
        rgb = _ensure_size(_load_rgb(path), w, h)

        depth_raw  = converter.predict(image_tensor)
        depth_norm = converter.postprocess(depth_raw, w, h)

        mask = segmenter.segment_from_prompt(
            image_tensor, prompt, output_size=(h, w), depth_map=depth_norm
        )

        out = captra.forward(rgb=rgb, depth_norm=depth_norm, mask=mask)

        features = extract_object_features(segmenter, image_tensor, mask)

        tracker.add_frame(
            frame_idx  = idx,
            features   = features,
            captra_output = out,
            mask       = mask,
            image_path = path,
        )

        if show_preview:
            _show_preview(rgb, mask, out, idx)

        if verbose:
            print_pose_summary(out)
            print(f"  masked patches: {features.get('num_masked_patches', '?')}")

    if show_preview:
        cv2.destroyAllWindows()

    if verbose:
        print(f"\n[PIPELINE] Done — {len(tracker.frames)} frames recorded")

    return tracker


# ── tracked video pipeline ─────────────────────────────────────────────────────

def run_video_tracked(
    video_path: str,
    weights_path: str,
    prompt: str,
    rot_dir: str,
    coord_dir: str,
    category: int,
    target_fps: float = 2.0,
    intrinsics: Optional[np.ndarray] = None,
    num_points: int = 4096,
    feature_key: str = "object_mean",
    verbose: bool = True,
    show_preview: bool = False,
) -> "FeaturePoseTracker":
    """
    Run the full tracked pipeline on a video file.

    Samples the video at target_fps, then runs the same
    DepthAnything → DINO → CAPTRA → FeaturePoseTracker pipeline as
    run_sequence_tracked.

    Parameters
    ----------
    video_path  : path to input video file
    weights_path: DepthAnything checkpoint (.pth)
    prompt      : text prompt for DINO segmentation
    rot_dir     : CAPTRA rotation net checkpoint dir
    coord_dir   : CAPTRA coordinate net checkpoint dir
    category    : NOCS category index (1–6)
    target_fps  : processing frame rate (default 2.0)
    intrinsics  : (3, 3) camera K; defaults to NOCS real camera
    num_points  : point cloud size per frame
    feature_key : DINO feature to track
    verbose     : print progress

    Returns
    -------
    FeaturePoseTracker with all sampled frames recorded.
    """
    from .combination import FeaturePoseTracker, extract_object_features

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    converter = DepthAnything(weights_path)
    segmenter = DINOSegmenter()
    captra    = CAPTRA(
        rot_dir   = rot_dir,
        coord_dir = coord_dir,
        category  = category,
        device    = device,
        intrinsics= intrinsics,
        num_points= num_points,
    )
    tracker = FeaturePoseTracker(feature_key=feature_key)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, int(round(native_fps / target_fps)))
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if verbose:
        print(
            f"[PIPELINE] {video_path} — {total} frames @ {native_fps:.1f} fps, "
            f"step={frame_step}, category={category}"
        )

    frame_idx   = -1
    sampled_idx = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % frame_step != 0:
            continue

        image_tensor, w, h = converter.frame_to_tensor(frame_bgr)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = _ensure_size(rgb, w, h)

        depth_raw  = converter.predict(image_tensor)
        depth_norm = converter.postprocess(depth_raw, w, h)

        mask = segmenter.segment_from_prompt(
            image_tensor, prompt, output_size=(h, w), depth_map=depth_norm
        )

        out = captra.forward(rgb=rgb, depth_norm=depth_norm, mask=mask)

        features = extract_object_features(segmenter, image_tensor, mask)

        tracker.add_frame(
            frame_idx  = sampled_idx,
            features   = features,
            captra_output = out,
            mask       = mask,
            timestamp  = frame_idx / native_fps,
        )

        if show_preview:
            _show_preview(rgb, mask, out, sampled_idx)

        if verbose:
            print(
                f"  Frame {sampled_idx} (video {frame_idx} / {frame_idx/native_fps:.1f}s): "
                f"valid={out.get('valid')}, msg={out.get('message')}"
            )

        sampled_idx += 1

    cap.release()
    if show_preview:
        cv2.destroyAllWindows()

    if verbose:
        print(f"\n[PIPELINE] {sampled_idx} frames recorded")

    return tracker
