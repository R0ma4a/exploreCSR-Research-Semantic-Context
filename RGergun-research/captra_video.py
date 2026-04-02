#!/usr/bin/env python
"""
Run CAPTRA over sampled frames from a video file.

Pipeline per sampled frame:
  1) RGB frame -> DepthAnything depth
  2) DINO prompt-guided mask via segment_from_prompt
  3) CAPTRA forward with previous_reference_state chaining

Outputs:
  - Console summaries in the same style as other CAPTRA scripts
  - CSV file with pose values for each sampled frame
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List, Optional

import cv2
import numpy as np
import torch


def _add_rged_to_path() -> None:
    root_dir = os.path.dirname(os.path.abspath(__file__))
    rged_dir = os.path.join(root_dir, "RGed-research")
    if rged_dir not in sys.path:
        sys.path.insert(0, rged_dir)


_add_rged_to_path()

import depth_anything  # type: ignore
import dino  # type: ignore
from captra import CAPTRA, CAPTRAReferenceState  # type: ignore
from captra_viz import print_pose_summary  # type: ignore


def apply_scale_relative_to_anchor(
    output_dict: dict,
    anchor_extent_mean: Optional[float],
) -> tuple[dict, Optional[float]]:
    """
    Re-express CAPTRA scale relative to the first valid reference frame.
    """
    ref = output_dict.get("reference_state")
    if ref is not None and anchor_extent_mean is None:
        anchor_extent_mean = float(np.mean(ref.extents))

    if ref is None or anchor_extent_mean is None:
        return output_dict, anchor_extent_mean

    curr_extent_mean = float(np.mean(ref.extents))
    denom = max(anchor_extent_mean, 1e-8)
    scale_anchor = curr_extent_mean / denom

    output_dict["scale"] = scale_anchor
    pose_vector = output_dict.get("pose_vector")
    if pose_vector is not None and len(pose_vector) >= 7:
        pose_vector = np.asarray(pose_vector).copy()
        pose_vector[6] = scale_anchor
        output_dict["pose_vector"] = pose_vector
    return output_dict, anchor_extent_mean


def frame_to_tensor_for_depth(
    frame_bgr: np.ndarray,
    device: torch.device,
    resize_hw: tuple[int, int] = (518, 518),
) -> tuple[torch.Tensor, int, int]:
    """
    Convert BGR frame into DepthAnything input tensor.

    Matches depth_anything.DepthAnything.image_to_tensor behavior:
    - Convert to RGB
    - Resize to 518x518
    - Scale to [0, 1]
    - NCHW tensor
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    original_h, original_w = rgb.shape[:2]

    rgb_resized = cv2.resize(rgb, (resize_hw[1], resize_hw[0]))
    rgb_resized = rgb_resized.astype(np.float32) / 255.0
    chw = np.transpose(rgb_resized, (2, 0, 1))
    tensor = torch.from_numpy(chw).unsqueeze(0).to(device)
    return tensor, original_w, original_h


def process_video(
    video_path: str,
    weights_path: str,
    prompt: str,
    out_csv: str,
    target_fps: float,
    fx: float,
    fy: float,
    cx: Optional[float],
    cy: Optional[float],
    depth_scale: float,
) -> None:
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"DepthAnything weights not found: {weights_path}")
    if target_fps <= 0:
        raise ValueError("--fps must be > 0")

    print(f"[CAPTRA-VIDEO] Video: {video_path}")
    print(f"[CAPTRA-VIDEO] Weights: {weights_path}")
    print(f"[CAPTRA-VIDEO] Prompt: {prompt!r}")
    print(f"[CAPTRA-VIDEO] Target FPS: {target_fps}")
    print(f"[CAPTRA-VIDEO] CSV output: {out_csv}")

    converter = depth_anything.DepthAnything(weights_path)
    segmenter = dino.dino()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0 or np.isnan(native_fps):
        native_fps = 30.0
    frame_step = max(1, int(round(native_fps / target_fps)))
    effective_fps = native_fps / frame_step

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(
        f"[CAPTRA-VIDEO] Native FPS: {native_fps:.3f}, step: {frame_step}, "
        f"effective FPS: {effective_fps:.3f}, total frames: {total_frames}"
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    csv_rows: List[List[object]] = []

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

        image_tensor, original_w, original_h = frame_to_tensor_for_depth(
            frame_bgr, converter.device
        )
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if rgb.shape[:2] != (original_h, original_w):
            rgb = cv2.resize(rgb, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        h, w = rgb.shape[:2]

        cx_use = cx if cx is not None else w / 2.0
        cy_use = cy if cy is not None else h / 2.0

        if captra is None:
            K = np.array(
                [[fx, 0.0, cx_use], [0.0, fy, cy_use], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            captra = CAPTRA(camera_intrinsics=K, depth_scale=depth_scale)
        else:
            captra.K[0, 2] = cx_use  # type: ignore[union-attr]
            captra.K[1, 2] = cy_use  # type: ignore[union-attr]

        depth_raw = converter.predict_depth(image_tensor)
        depth_norm = converter.process_depth(depth_raw, original_w, original_h)

        mask = segmenter.segment_from_prompt(
            image_tensor,
            prompt,
            output_size=(original_h, original_w),
        )

        assert captra is not None
        out = captra.forward(
            rgb=rgb,
            depth=depth_norm,
            seg_or_mask=mask,
            target_label=None,
            previous_reference_state=prev_state,
        )
        out, anchor_extent_mean = apply_scale_relative_to_anchor(out, anchor_extent_mean)

        print(f"\n=== Frame {sampled_idx} (video frame {frame_idx}) ===")
        print_pose_summary(out)
        region_diag = out.get("diagnostics", {}).get("region", {})
        print(f"[CAPTRA-VIDEO] mask pixels: {region_diag.get('num_mask_pixels', 'NA')}")
        print(f"[CAPTRA-VIDEO] valid depth points: {region_diag.get('num_valid_depth', 'NA')}")
        print(f"[CAPTRA-VIDEO] valid flag: {out.get('valid')} | message: {out.get('message')}")

        t = np.asarray(out.get("translation", [np.nan, np.nan, np.nan]), dtype=float)
        r = np.asarray(out.get("rotation_euler", [np.nan, np.nan, np.nan]), dtype=float)
        s = float(out.get("scale", np.nan))
        pv = np.asarray(out.get("pose_vector", [np.nan] * 7), dtype=float)

        timestamp_sec = frame_idx / native_fps
        csv_rows.append(
            [
                sampled_idx,
                frame_idx,
                timestamp_sec,
                bool(out.get("valid", False)),
                out.get("message", ""),
                float(t[0]),
                float(t[1]),
                float(t[2]),
                float(np.degrees(r[0])),
                float(np.degrees(r[1])),
                float(np.degrees(r[2])),
                s,
                float(pv[0]),
                float(pv[1]),
                float(pv[2]),
                float(pv[3]),
                float(pv[4]),
                float(pv[5]),
                float(pv[6]),
            ]
        )

        ref = out.get("reference_state")
        if ref is not None:
            prev_state = ref

        sampled_idx += 1

    cap.release()

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample_idx",
                "frame_idx",
                "timestamp_sec",
                "valid",
                "message",
                "tx",
                "ty",
                "tz",
                "rx_deg",
                "ry_deg",
                "rz_deg",
                "scale",
                "pose_tx",
                "pose_ty",
                "pose_tz",
                "pose_rx",
                "pose_ry",
                "pose_rz",
                "pose_scale",
            ]
        )
        writer.writerows(csv_rows)

    print("\n=== CAPTRA-VIDEO Summary ===")
    print(f"[CAPTRA-VIDEO] Sampled frames processed: {sampled_idx}")
    print(f"[CAPTRA-VIDEO] CSV written: {out_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CAPTRA on sampled video frames and export pose CSV."
    )
    parser.add_argument("--video", type=str, required=True, help="Path to input video file.")
    parser.add_argument(
        "--weights",
        type=str,
        default=r"C:\Users\roman\Downloads\depth_anything_v2_vitb.pth",
        help="Path to DepthAnything checkpoint (.pth).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help='Prompt for dino.segment_from_prompt, e.g. "bag" or "person".',
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="captra_video_pose.csv",
        help="CSV path for per-frame CAPTRA pose outputs.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Target processing FPS (frame sampling rate).",
    )
    parser.add_argument("--fx", type=float, default=500.0, help="Camera focal length fx.")
    parser.add_argument("--fy", type=float, default=500.0, help="Camera focal length fy.")
    parser.add_argument("--cx", type=float, default=None, help="Camera principal point cx.")
    parser.add_argument("--cy", type=float, default=None, help="Camera principal point cy.")
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to depth before CAPTRA.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_video(
        video_path=args.video,
        weights_path=args.weights,
        prompt=args.prompt,
        out_csv=args.out_csv,
        target_fps=args.fps,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        depth_scale=args.depth_scale,
    )


if __name__ == "__main__":
    main()

