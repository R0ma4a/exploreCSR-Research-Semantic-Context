from __future__ import annotations

from exploreCSR.combination import tracker

"""
Unified BrownCSR pipeline runner.

Provides ``run_single``, ``run_sequence``, and ``run_video`` which chain:
  DepthAnything (depth) → DINO (segmentation) → CAPTRA (pose)

Each function returns structured output dicts. The scripts in exploreCSR/scripts/
are thin CLI wrappers around these functions.
"""

import csv
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import CameraConfig, PipelineConfig
from .depth import DepthAnything
from .segmentation import DINOSegmenter
from .pose import CAPTRA, CAPTRAReferenceState
from .visualization import print_pose_summary


# ======================================================================
# Shared helpers
# ======================================================================


def _load_rgb(image_path: str) -> np.ndarray:
    """Load an image as RGB uint8 (H, W, 3)."""
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _ensure_size(rgb: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize RGB if it doesn't match the expected dimensions."""
    if rgb.shape[0] != h or rgb.shape[1] != w:
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    return rgb


def _build_captra(
    camera: CameraConfig, width: int, height: int
) -> CAPTRA:
    """Construct a CAPTRA instance from camera config and image dimensions."""
    K = camera.intrinsics_matrix(width, height)
    return CAPTRA(camera_intrinsics=K, depth_scale=camera.depth_scale)


def apply_scale_relative_to_anchor(
    output_dict: dict,
    anchor_extent_mean: Optional[float],
) -> Tuple[dict, Optional[float]]:
    """
    Re-express CAPTRA scale relative to the first valid reference frame.

    On the first valid frame, the anchor is set from the reference extents.
    Subsequent frames report scale as current_extent / anchor_extent.
    """
    ref = output_dict.get("reference_state")
    if ref is not None and anchor_extent_mean is None:
        anchor_extent_mean = float(np.mean(ref.extents))

    if ref is None or anchor_extent_mean is None:
        return output_dict, anchor_extent_mean

    curr_extent_mean = float(np.mean(ref.extents))
    scale_anchor = curr_extent_mean / max(anchor_extent_mean, 1e-8)

    output_dict["scale"] = scale_anchor
    pose_vector = output_dict.get("pose_vector")
    if pose_vector is not None and len(pose_vector) >= 7:
        pose_vector = np.asarray(pose_vector).copy()
        pose_vector[6] = scale_anchor
        output_dict["pose_vector"] = pose_vector

    return output_dict, anchor_extent_mean


def _process_single_frame(
    converter: DepthAnything,
    segmenter: DINOSegmenter,
    captra: CAPTRA,
    image_tensor,
    rgb: np.ndarray,
    original_w: int,
    original_h: int,
    prompt: Optional[str],
    use_prompt: bool,
    prev_state: Optional[CAPTRAReferenceState],
    anchor_extent_mean: Optional[float],
) -> Tuple[Dict[str, Any], Optional[CAPTRAReferenceState], Optional[float]]:
    """
    Core single-frame processing shared by all pipeline modes.

    Returns (output_dict, updated_prev_state, updated_anchor_extent_mean).
    """
    # Depth
    depth_raw = converter.predict(image_tensor)
    depth_norm = converter.postprocess(depth_raw, original_w, original_h)

    # Segmentation
    if use_prompt and prompt:
        mask = segmenter.segment_from_prompt(
            image_tensor, prompt, output_size=(original_h, original_w),
            depth_map=depth_norm,
        )
    else:
        mask = segmenter.generate_object_mask(
            image_tensor, depth_norm, (original_h, original_w)
        )

    # CAPTRA forward
    out = captra.forward(
        rgb=rgb,
        depth=depth_norm,
        seg_or_mask=mask,
        target_label=None,
        previous_reference_state=prev_state,
    )
    out, anchor_extent_mean = apply_scale_relative_to_anchor(out, anchor_extent_mean)

    # Update reference state
    ref = out.get("reference_state")
    new_prev = ref if ref is not None else prev_state

    return out, new_prev, anchor_extent_mean


# ======================================================================
# Single image
# ======================================================================


def run_single(
    image_path: str,
    weights_path: str,
    prompt: Optional[str] = None,
    camera: Optional[CameraConfig] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the full pipeline on a single image.

    Parameters
    ----------
    image_path   : path to RGB image
    weights_path : path to DepthAnything checkpoint
    prompt       : text prompt for segmentation (None → unsupervised)
    camera       : camera intrinsics config (None → defaults)
    verbose      : print progress and summary

    Returns
    -------
    CAPTRA output dict with pose, mask, points, diagnostics, etc.
    """
    if camera is None:
        camera = CameraConfig()

    if verbose:
        print(f"[PIPELINE] Image: {image_path}")
        print(f"[PIPELINE] Weights: {weights_path}")
        if prompt:
            print(f"[PIPELINE] Prompt: {prompt!r}")

    converter = DepthAnything(weights_path)
    segmenter = DINOSegmenter()

    image_tensor, _, original_w, original_h = converter.preprocess(image_path)
    rgb = _ensure_size(_load_rgb(image_path), original_w, original_h)

    captra = _build_captra(camera, original_w, original_h)

    out, _, _ = _process_single_frame(
        converter=converter,
        segmenter=segmenter,
        captra=captra,
        image_tensor=image_tensor,
        rgb=rgb,
        original_w=original_w,
        original_h=original_h,
        prompt=prompt,
        use_prompt=prompt is not None,
        prev_state=None,
        anchor_extent_mean=None,
    )

    if verbose:
        print_pose_summary(out)

    return out


# ======================================================================
# Multi-image sequence
# ======================================================================


def run_sequence(
    image_paths: List[str],
    weights_path: str,
    prompt: str,
    camera: Optional[CameraConfig] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run CAPTRA over multiple images in temporal order, chaining reference
    state so each frame reports pose *change* vs the previous frame.

    Returns a list of CAPTRA output dicts, one per frame.
    """
    if camera is None:
        camera = CameraConfig()

    if not image_paths:
        raise ValueError("No images to process.")

    if verbose:
        print(f"[SEQUENCE] {len(image_paths)} frame(s), prompt={prompt!r}")

    converter = DepthAnything(weights_path)
    segmenter = DINOSegmenter()

    results: List[Dict[str, Any]] = []
    prev_state: Optional[CAPTRAReferenceState] = None
    captra: Optional[CAPTRA] = None
    anchor_extent_mean: Optional[float] = None

    for idx, path in enumerate(image_paths):
        if verbose:
            print(f"\n{'='*50}\n=== Frame {idx}: {path}\n{'='*50}")

        image_tensor, _, original_w, original_h = converter.preprocess(path)
        rgb = _ensure_size(_load_rgb(path), original_w, original_h)

        if captra is None:
            captra = _build_captra(camera, original_w, original_h)

        out, prev_state, anchor_extent_mean = _process_single_frame(
            converter=converter,
            segmenter=segmenter,
            captra=captra,
            image_tensor=image_tensor,
            rgb=rgb,
            original_w=original_w,
            original_h=original_h,
            prompt=prompt,
            use_prompt=True,
            prev_state=prev_state,
            anchor_extent_mean=anchor_extent_mean,
        )

        if verbose:
            print_pose_summary(out)

        results.append(out)

    return results


# ======================================================================
# Video processing
# ======================================================================


def run_video(
    video_path: str,
    weights_path: str,
    prompt: str,
    out_csv: Optional[str] = None,
    target_fps: float = 2.0,
    camera: Optional[CameraConfig] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run CAPTRA over sampled frames from a video file.

    Parameters
    ----------
    video_path  : path to input video
    weights_path: path to DepthAnything checkpoint
    prompt      : text prompt for segmentation
    out_csv     : path for CSV output (None → skip CSV)
    target_fps  : desired processing frame rate
    camera      : camera intrinsics config
    verbose     : print progress

    Returns
    -------
    List of CAPTRA output dicts, one per sampled frame.
    """
    if camera is None:
        camera = CameraConfig()

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if target_fps <= 0:
        raise ValueError("target_fps must be > 0")

    converter = DepthAnything(weights_path)
    segmenter = DINOSegmenter()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0 or np.isnan(native_fps):
        native_fps = 30.0
    frame_step = max(1, int(round(native_fps / target_fps)))

    if verbose:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[VIDEO] {video_path} — {total} frames, step={frame_step}")

    results: List[Dict[str, Any]] = []
    csv_rows: List[List] = []
    prev_state: Optional[CAPTRAReferenceState] = None
    captra: Optional[CAPTRA] = None
    anchor_extent_mean: Optional[float] = None

    frame_idx = -1
    sampled_idx = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % frame_step != 0:
            continue

        image_tensor, original_w, original_h = converter.frame_to_tensor(frame_bgr)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = _ensure_size(rgb, original_w, original_h)

        if captra is None:
            captra = _build_captra(camera, original_w, original_h)

        out, prev_state, anchor_extent_mean = _process_single_frame(
            converter=converter,
            segmenter=segmenter,
            captra=captra,
            image_tensor=image_tensor,
            rgb=rgb,
            original_w=original_w,
            original_h=original_h,
            prompt=prompt,
            use_prompt=True,
            prev_state=prev_state,
            anchor_extent_mean=anchor_extent_mean,
        )

        if verbose:
            print(f"\n=== Frame {sampled_idx} (video frame {frame_idx}) ===")
            print_pose_summary(out)

        results.append(out)

        # Build CSV row
        t = np.asarray(out.get("translation", [np.nan] * 3), dtype=float)
        r = np.asarray(out.get("rotation_euler", [np.nan] * 3), dtype=float)
        s = float(out.get("scale", np.nan))
        pv = np.asarray(out.get("pose_vector", [np.nan] * 7), dtype=float)

        csv_rows.append([
            sampled_idx, frame_idx, frame_idx / native_fps,
            bool(out.get("valid", False)), out.get("message", ""),
            float(t[0]), float(t[1]), float(t[2]),
            float(np.degrees(r[0])), float(np.degrees(r[1])), float(np.degrees(r[2])),
            s,
            *[float(pv[i]) for i in range(min(len(pv), 7))],
        ])

        sampled_idx += 1

    cap.release()

    # Write CSV
    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "sample_idx", "frame_idx", "timestamp_sec", "valid", "message",
                "tx", "ty", "tz", "rx_deg", "ry_deg", "rz_deg", "scale",
                "pose_tx", "pose_ty", "pose_tz", "pose_rx", "pose_ry", "pose_rz", "pose_scale",
            ])
            writer.writerows(csv_rows)
        if verbose:
            print(f"\n[VIDEO] CSV written: {out_csv}")

    if verbose:
        print(f"[VIDEO] Processed {sampled_idx} frames")

    return results


# ======================================================================
# Tracked variants (with DINO feature extraction + delta plotting)
# ======================================================================


def run_sequence_tracked(
    image_paths: List[str],
    weights_path: str,
    prompt: str,
    camera: Optional[CameraConfig] = None,
    feature_key: str = "object_mean",
    verbose: bool = True,
) -> "FeaturePoseTracker":

    from .combination import FeaturePoseTracker, extract_object_features

    if camera is None:
        camera = CameraConfig()

    if not image_paths:
        raise ValueError("No images to process.")

    if verbose:
        print(f"[TRACKED] {len(image_paths)} frame(s), prompt={prompt!r}")

    converter = DepthAnything(weights_path)
    segmenter = DINOSegmenter()
    tracker = FeaturePoseTracker(feature_key=feature_key)

    prev_state: Optional[CAPTRAReferenceState] = None
    captra: Optional[CAPTRA] = None
    anchor_extent_mean: Optional[float] = None

    for idx, path in enumerate(image_paths):
        if verbose:
            print(f"\n--- Frame {idx}: {path} ---")

        image_tensor, _, original_w, original_h = converter.preprocess(path)
        rgb = _ensure_size(_load_rgb(path), original_w, original_h)

        if captra is None:
            captra = _build_captra(camera, original_w, original_h)

        out, prev_state, anchor_extent_mean = _process_single_frame(
            converter=converter,
            segmenter=segmenter,
            captra=captra,
            image_tensor=image_tensor,
            rgb=rgb,
            original_w=original_w,
            original_h=original_h,
            prompt=prompt,
            use_prompt=True,
            prev_state=prev_state,
            anchor_extent_mean=anchor_extent_mean,
        )

        mask = out.get("mask", np.zeros((original_h, original_w), dtype=np.uint8))
        features = extract_object_features(segmenter, image_tensor, mask)

        tracker.add_frame(
            frame_idx=idx,
            features=features,
            captra_output=out,
            mask=mask,                 # ✅ NOW VALID
            image_path=path,
        )

        if verbose:
            print_pose_summary(out)
            print(f"  masked patches: {features['num_masked_patches']}")

    if verbose:
        print(f"\n[TRACKED] Done — {len(tracker.frames)} frames recorded")

    return tracker

def run_video_tracked(
    video_path: str,
    weights_path: str,
    prompt: str,
    target_fps: float = 2.0,
    camera: Optional[CameraConfig] = None,
    feature_key: str = "object_mean",
    verbose: bool = True,
) -> "FeaturePoseTracker":

    from .combination import FeaturePoseTracker, extract_object_features

    if camera is None:
        camera = CameraConfig()

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    converter = DepthAnything(weights_path)
    segmenter = DINOSegmenter()
    tracker = FeaturePoseTracker(feature_key=feature_key)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0 or np.isnan(native_fps):
        native_fps = 30.0

    frame_step = max(1, int(round(native_fps / target_fps)))

    prev_state: Optional[CAPTRAReferenceState] = None
    captra: Optional[CAPTRA] = None
    anchor_extent_mean: Optional[float] = None

    frame_idx = -1
    sampled_idx = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_idx += 1
        if frame_idx % frame_step != 0:
            continue

        image_tensor, original_w, original_h = converter.frame_to_tensor(frame_bgr)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = _ensure_size(rgb, original_w, original_h)

        if captra is None:
            captra = _build_captra(camera, original_w, original_h)

        out, prev_state, anchor_extent_mean = _process_single_frame(
            converter=converter,
            segmenter=segmenter,
            captra=captra,
            image_tensor=image_tensor,
            rgb=rgb,
            original_w=original_w,
            original_h=original_h,
            prompt=prompt,
            use_prompt=True,
            prev_state=prev_state,
            anchor_extent_mean=anchor_extent_mean,
        )

        mask = out.get("mask", np.zeros((original_h, original_w), dtype=np.uint8))
        features = extract_object_features(segmenter, image_tensor, mask)

        tracker.add_frame(
            frame_idx=sampled_idx,
            features=features,
            captra_output=out,
            mask=mask,   # ✅ NOW VALID
            timestamp=frame_idx / native_fps,
        )

        if verbose:
            print(f"Frame {sampled_idx} (video {frame_idx}): valid={out.get('valid')}")

        sampled_idx += 1

    cap.release()

    if verbose:
        print(f"\n[VIDEO-TRACKED] {sampled_idx} frames recorded")

    return tracker