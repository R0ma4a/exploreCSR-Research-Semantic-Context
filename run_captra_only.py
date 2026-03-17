#!/usr/bin/env python
"""
Run CAPTRA on a single RGB image and visualize the result.

This script is a streamlined variant of run_full_pipeline.py that:
  - Takes an input JPG/JPEG (or other RGB image)
  - Internally uses DepthAnything to predict depth
  - Internally uses DINO to obtain a segmentation map
  - Runs CAPTRA on the RGB + depth + mask
  - Shows CAPTRA visualizations (mask overlay, depth, point cloud, reference frame)

The emphasis is on exercising and visualizing CAPTRA, not on printing
all intermediate details.
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


def run_captra_only(
    image_path: str,
    weights_path: str,
    fx: float,
    fy: float,
    cx: Optional[float],
    cy: Optional[float],
    depth_scale: float = 1.0,
    k_clusters: int = 5,
    target_label: Optional[int] = None,
) -> None:
    """
    Run CAPTRA (with internal depth + segmentation) on a single RGB image
    and show visualizations.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"DepthAnything weights not found: {weights_path}")

    print(f"[CAPTRA-ONLY] Using image: {image_path}")
    print(f"[CAPTRA-ONLY] Using DepthAnything checkpoint: {weights_path}")

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

    # 4) DINOv3 feature extraction and segmentation (steps 1–4)
    print("[CAPTRA-ONLY] Extracting DINO features...")
    dino_tensor = preprocess_for_dino(rgb, size=(224, 224))
    features = segmenter.extract_features(dino_tensor)
    patch_grid = segmenter.process_patch_tokens(features)
    patch_features_np, H_p, W_p = segmenter.prepare_features_for_clustering(patch_grid)

    print("[CAPTRA-ONLY] Clustering DINO patch features...")
    seg_small = segmenter.cluster_features(patch_features_np, H_p, W_p, k=k_clusters)
    print(f"[CAPTRA-ONLY] DINO patch segmentation shape: {seg_small.shape} (H_p={H_p}, W_p={W_p})")

    # 5) Upsample segmentation to image size for CAPTRA
    print("[CAPTRA-ONLY] Upsampling segmentation to full image size for CAPTRA...")
    seg_full = cv2.resize(
        seg_small.astype(np.uint8),
        (W, H),
        interpolation=cv2.INTER_NEAREST,
    )
    print(f"[CAPTRA-ONLY] Upsampled segmentation shape: {seg_full.shape}")

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
        seg_or_mask=seg_full,
        target_label=target_label,
        previous_reference_state=None,
    )

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

    # Use a simple object mask for overlay: either target_label or the auto-selected mask
    if target_label is not None:
        obj_mask = seg_full == target_label
    else:
        obj_mask = mask

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_captra_only(
        image_path=args.image,
        weights_path=args.weights,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        depth_scale=args.depth_scale,
        k_clusters=args.k,
        target_label=args.target_label,
    )


if __name__ == "__main__":
    main()

