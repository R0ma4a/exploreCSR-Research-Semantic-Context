#!/usr/bin/env python
"""
End-to-end test script for the current BrownCSR pipeline:

RGB image -> DepthAnything (depth) + DINO (segmentation) -> CAPTRA (pose).

This is meant as a practical, research-friendly driver that exercises:
  - depth_anything.DepthAnything (RGB-D prediction)
  - dino.dino (steps 1–4 + upsampling)
  - CAPTRA (object-centered pose estimation)
  - captra_viz (optional visualization)

Usage examples
--------------
Basic run with the default checkpoint path from RGed-research/main.py:

    python run_full_pipeline.py --image path/to/image.jpeg --show

Custom checkpoint and intrinsics:

    python run_full_pipeline.py \\
        --image path/to/image.jpeg \\
        --weights D:/path/to/depth_anything_v2_vitb.pth \\
        --fx 500 --fy 500 --cx 256 --cy 256 \\
        --show
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


# ------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------


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


def preprocess_for_dino(rgb: np.ndarray, size: Tuple[int, int] = (224, 224)) -> "torch.Tensor":
    """
    Preprocess an RGB image for the timm DINO model.

    DINO expects inputs of a fixed image size (e.g. 224x224 for
    vit_small_patch16_dinov3_qkvb), so we resize explicitly here
    instead of reusing the DepthAnything tensor size.
    """
    import torch  # local import to avoid hard dependency at module import time

    rgb_resized = cv2.resize(rgb, size)
    img = rgb_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean[None, None, :]) / std[None, None, :]

    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return tensor


# ------------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------------


def run_pipeline(
    image_path: str,
    weights_path: str,
    fx: float,
    fy: float,
    cx: Optional[float],
    cy: Optional[float],
    depth_scale: float = 1.0,
    k_clusters: int = 5,
    target_label: Optional[int] = None,
    show: bool = False,
) -> None:
    """
    Run the full pipeline on a single image:

    - Predict depth with DepthAnything
    - Segment with DINO (steps 1–4 + upsample)
    - Estimate pose with CAPTRA
    - Optionally visualize with captra_viz helpers
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"DepthAnything weights not found: {weights_path}")

    print(f"[INFO] Using image: {image_path}")
    print(f"[INFO] Using DepthAnything checkpoint: {weights_path}")

    # 1) Initialize DepthAnything and DINO
    print("[INFO] Initializing DepthAnything...")
    converter = depth_anything.DepthAnything(weights_path)

    print("[INFO] Initializing DINO segmenter...")
    segmenter = dino.dino()

    # 2) Preprocess image for DepthAnything
    print("[INFO] Converting image to tensor for DepthAnything...")
    image_tensor, _, original_width, original_height = converter.image_to_tensor(image_path)

    # Load RGB separately for CAPTRA (full resolution, RGB)
    rgb = load_rgb_for_captra(image_path)
    if rgb.shape[0] != original_height or rgb.shape[1] != original_width:
        # In practice these should match, but guard just in case
        rgb = cv2.resize(rgb, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
    print(f"[INFO] RGB shape for CAPTRA: {rgb.shape}")

    # 3) Predict depth and post-process to original resolution
    print("[INFO] Predicting depth with DepthAnything...")
    depth_raw = converter.predict_depth(image_tensor)
    depth_norm = converter.process_depth(depth_raw, original_width, original_height)
    print(f"[INFO] Depth map shape (normalized): {depth_norm.shape}")

    # 4) DINOv3 feature extraction and segmentation (steps 1–4)
    print("[INFO] Extracting DINO features...")
    dino_tensor = preprocess_for_dino(rgb, size=(224, 224))
    features = segmenter.extract_features(dino_tensor)
    patch_grid = segmenter.process_patch_tokens(features)
    patch_features_np, H_p, W_p = segmenter.prepare_features_for_clustering(patch_grid)

    print("[INFO] Clustering DINO patch features...")
    seg_small = segmenter.cluster_features(patch_features_np, H_p, W_p, k=k_clusters)
    print(f"[INFO] DINO patch segmentation shape: {seg_small.shape} (H_p={H_p}, W_p={W_p})")

    # 5) Upsample segmentation to image size for CAPTRA
    print("[INFO] Upsampling segmentation to full image size for CAPTRA...")
    seg_full = cv2.resize(
        seg_small.astype(np.uint8),
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST,
    )
    print(f"[INFO] Upsampled segmentation shape: {seg_full.shape}")

    # 6) Initialize CAPTRA with camera intrinsics
    if cx is None:
        cx = original_width / 2.0
    if cy is None:
        cy = original_height / 2.0

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    print("[INFO] Initializing CAPTRA...")
    captra = CAPTRA(camera_intrinsics=K, depth_scale=depth_scale)

    # 7) Run CAPTRA forward
    print("[INFO] Running CAPTRA.forward() on single frame...")
    out = captra.forward(
        rgb=rgb,
        depth=depth_norm,
        seg_or_mask=seg_full,
        target_label=target_label,
        previous_reference_state=None,
    )

    # 8) Summarize outputs
    print("\n=== DINO / CAPTRA Pipeline Output ===")
    print(f"Segmentation labels present (upsampled): {np.unique(seg_full)}")
    print_pose_summary(out)

    print("\n[INFO] Additional diagnostics:")
    region_diag = out.get("diagnostics", {}).get("region", {})
    print(f"  mask pixels: {region_diag.get('num_mask_pixels', 'NA')}")
    print(f"  valid depth points: {region_diag.get('num_valid_depth', 'NA')}")
    print(f"  valid flag: {out.get('valid')}")
    print(f"  message: {out.get('message')}")

    # 9) Optional visualization
    if show:
        print("[INFO] Visualization enabled (--show). Launching plots...")

        mask = out["mask"]
        masked_depth = out["masked_depth"]
        points = out["object_points"]
        centroid = out["object_centroid"]
        axes = out["principal_axes"]

        # Use a simple object mask for overlay: either target_label or the auto-selected mask
        if target_label is not None:
            obj_mask = seg_full == target_label
        else:
            obj_mask = mask

        print("[INFO] Showing mask overlay...")
        show_mask_overlay(rgb, obj_mask)

        print("[INFO] Showing masked depth...")
        show_masked_depth(masked_depth, mask)

        print("[INFO] Showing object point cloud (subsampled for visualization)...")
        show_pointcloud(points, title="Object Point Cloud", max_points=50000)

        if centroid is not None and axes is not None:
            print("[INFO] Showing reference frame on point cloud (subsampled)...")
            show_reference_frame(
                points, centroid, axes,
                title="Object Reference Frame",
                max_points=50000,
            )


# ------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BrownCSR DINO + DepthAnything + CAPTRA pipeline on a single image."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to RGB image.")
    parser.add_argument(
        "--weights",
        type=str,
        default=r"D:\Research Projects\exploreCSR-Research-Semantic-Context\RGed-research\checkpoints\depth_anything_v2_vitb.pth",  # noqa: E501
        help="Path to DepthAnything checkpoint (.pth). "
        "Defaults to the path used in RGed-research/main.py.",
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
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of clusters for DINO KMeans segmentation.",
    )
    parser.add_argument(
        "--target-label",
        type=int,
        default=None,
        help=(
            "Which segmentation label to treat as the target object. "
            "If omitted, CAPTRA will auto-select the most frequent non-zero label."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Enable matplotlib visualizations via captra_viz.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        image_path=args.image,
        weights_path=args.weights,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        depth_scale=args.depth_scale,
        k_clusters=args.k,
        target_label=args.target_label,
        show=args.show,
    )


if __name__ == "__main__":
    main()

