#!/usr/bin/env python
"""
Run CAPTRA on a single RGB image and visualize the result.

This script mirrors RGed-research/main.py’s flow:
  - DepthAnything: depth from the same image tensor
  - DINO: prompt-guided mask via segment_from_prompt(...)
  - CAPTRA: RGB + depth + mask → pose / reference frame

The emphasis is on exercising and visualizing CAPTRA with the same
masking pipeline as main.py.
"""

import argparse
import os
import sys
from typing import Optional, Tuple

import cv2
import numpy as np


def _add_rged_to_path() -> None:
    """
    Add the RGed-research directory to sys.path so that we can import
    the existing modules (`depth_anything`, `dino`, `captra`, etc.)
    without modifying their locations.
    """
    root_dir = os.path.dirname(os.path.abspath(__file__))
    rged_dir = os.path.join(root_dir, "RGed-research")
    if rged_dir not in sys.path:
        sys.path.insert(0, rged_dir)


_add_rged_to_path()

# After updating sys.path we can safely import the research modules.
import depth_anything  # type: ignore
import dino  # type: ignore
from captra import CAPTRA  # type: ignore
from captra_viz import (  # type: ignore
    print_pose_summary,
    show_mask_overlay,
    show_masked_depth,
    show_pointcloud,
    show_reference_frame,
)


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


def load_rgb_for_captra(image_path: str) -> np.ndarray:
    """
    Load an RGB image for CAPTRA (HxWx3, uint8, RGB).

    Uses the same OpenCV backend as the existing code, but ensures
    the color ordering is RGB instead of BGR.
    """
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def run_captra_only(
    image_path: str,
    weights_path: str,
    prompt: str,
    fx: float,
    fy: float,
    cx: Optional[float],
    cy: Optional[float],
    depth_scale: float = 1.0,
) -> None:
    """
    Run CAPTRA (with internal depth + prompt-based DINO mask) on a single
    RGB image and show visualizations.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"DepthAnything weights not found: {weights_path}")

    print(f"[CAPTRA-ONLY] Using image: {image_path}")
    print(f"[CAPTRA-ONLY] Using DepthAnything checkpoint: {weights_path}")
    print(f"[CAPTRA-ONLY] Prompt: {prompt!r}")

    # 1) Initialize DepthAnything and DINO
    print("[CAPTRA-ONLY] Initializing DepthAnything...")
    converter = depth_anything.DepthAnything(weights_path)

    print("[CAPTRA-ONLY] Initializing DINO segmenter...")
    segmenter = dino.dino()

    # 2) Preprocess image for DepthAnything
    print("[CAPTRA-ONLY] Converting image to tensor for DepthAnything...")
    image_tensor, _, original_width, original_height = converter.image_to_tensor(image_path)

    # Load RGB separately for CAPTRA (full resolution, RGB)
    rgb = load_rgb_for_captra(image_path)
    if rgb.shape[0] != original_height or rgb.shape[1] != original_width:
        rgb = cv2.resize(rgb, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
    H, W, _ = rgb.shape
    print(f"[CAPTRA-ONLY] RGB shape for CAPTRA: {rgb.shape}")

    # 3) Predict depth and post-process to original resolution
    print("[CAPTRA-ONLY] Predicting depth with DepthAnything...")
    depth_raw = converter.predict_depth(image_tensor)
    depth_norm = converter.process_depth(depth_raw, original_width, original_height)
    print(f"[CAPTRA-ONLY] Depth map shape (normalized): {depth_norm.shape}")

    # 4) Prompt-based object mask (same API as RGed-research/main.py)
    print("[CAPTRA-ONLY] Generating object mask with DINO segment_from_prompt...")
    mask = segmenter.segment_from_prompt(
        image_tensor,
        prompt,
        output_size=(original_height, original_width),
    )
    print(f"[CAPTRA-ONLY] Object mask shape: {mask.shape}, unique values: {np.unique(mask)}")

    # 6) Initialize CAPTRA with camera intrinsics
    if cx is None:
        cx = W / 2.0
    if cy is None:
        cy = H / 2.0

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    print("[CAPTRA-ONLY] Initializing CAPTRA...")
    captra = CAPTRA(camera_intrinsics=K, depth_scale=depth_scale)

    # 7) Run CAPTRA forward
    print("[CAPTRA-ONLY] Running CAPTRA.forward() on single frame...")
    out = captra.forward(
        rgb=rgb,
        depth=depth_norm,
        seg_or_mask=mask,
        target_label=None,
        previous_reference_state=None,
    )
    out, _ = apply_scale_relative_to_anchor(out, anchor_extent_mean=None)

    # 8) Print a concise pose summary and diagnostics
    print("\n=== CAPTRA-ONLY Pose Output ===")
    print_pose_summary(out)

    region_diag = out.get("diagnostics", {}).get("region", {})
    print("\n[CAPTRA-ONLY] Region diagnostics:")
    print(f"  mask pixels: {region_diag.get('num_mask_pixels', 'NA')}")
    print(f"  valid depth points: {region_diag.get('num_valid_depth', 'NA')}")
    print(f"  valid flag: {out.get('valid')}")
    print(f"  message: {out.get('message')}")

    # 9) CAPTRA-focused visualizations
    print("[CAPTRA-ONLY] Launching CAPTRA visualizations...")

    mask = out["mask"]
    masked_depth = out["masked_depth"]
    points = out["object_points"]
    centroid = out["object_centroid"]
    axes = out["principal_axes"]

    # Use CAPTRA/DINO object mask directly
    obj_mask = mask.astype(bool)

    print("[CAPTRA-ONLY] Showing mask overlay...")
    show_mask_overlay(rgb, obj_mask)

    print("[CAPTRA-ONLY] Showing masked depth...")
    show_masked_depth(masked_depth, mask)

    print("[CAPTRA-ONLY] Showing object point cloud (subsampled)...")
    show_pointcloud(points, title="CAPTRA Object Point Cloud", max_points=50000)

    if centroid is not None and axes is not None:
        print("[CAPTRA-ONLY] Showing reference frame on point cloud (subsampled)...")
        show_reference_frame(
            points,
            centroid,
            axes,
            title="CAPTRA Reference Frame",
            max_points=50000,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CAPTRA (with internal depth + segmentation) on a single RGB image."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to RGB image (e.g., .jpg or .jpeg).",
    )
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
        help=(
            "Text prompt for prompt-guided masking (dino.segment_from_prompt), "
            "e.g. \"bag\", \"red car\". Same role as `prompt` in RGed-research/main.py."
        ),
    )
    parser.add_argument("--fx", type=float, default=500.0, help="Camera focal length fx.")
    parser.add_argument("--fy", type=float, default=500.0, help="Camera focal length fy.")
    parser.add_argument(
        "--cx",
        type=float,
        default=None,
        help="Camera principal point cx (pixels). Default: image_width / 2.",
    )
    parser.add_argument(
        "--cy",
        type=float,
        default=None,
        help="Camera principal point cy (pixels). Default: image_height / 2.",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to depth values before CAPTRA. "
        "Since DepthAnything outputs normalized depth, 1.0 is usually fine.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_captra_only(
        image_path=args.image,
        weights_path=args.weights,
        prompt=args.prompt,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        depth_scale=args.depth_scale,
    )


if __name__ == "__main__":
    main()

