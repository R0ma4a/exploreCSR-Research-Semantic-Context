#!/usr/bin/env python
"""
Helper script to debug DINO-based segmentation and object masking.

This script runs:
  - RGed-research/dino.dino
  - captra_viz.show_mask_overlay

on a single RGB image, and visualizes each segmentation label as a separate
mask overlay so you can decide which label should be used as the target
object (for the --target-label flag in run_full_pipeline.py).
"""

import argparse
import os
import sys
from typing import Optional

import cv2
import numpy as np
import torch


def _add_rged_to_path() -> None:
    """Allow importing RGed-research modules without installation."""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    rged_dir = os.path.join(root_dir, "RGed-research")
    if rged_dir not in sys.path:
        sys.path.insert(0, rged_dir)


_add_rged_to_path()

import dino  # type: ignore
from captra_viz import show_mask_overlay  # type: ignore


def preprocess_for_dino(rgb: np.ndarray, size: tuple[int, int] = (224, 224)) -> torch.Tensor:
    """
    Minimal preprocessing for DINO:
    - Resize to the model's expected image size (e.g. 224x224)
    - Convert to float32 in [0,1]
    - Normalize with ImageNet-like mean/std
    - Convert to (1, 3, H, W) torch tensor
    """
    rgb_resized = cv2.resize(rgb, size)
    img = rgb_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean[None, None, :]) / std[None, None, :]

    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return tensor


def load_rgb(image_path: str) -> np.ndarray:
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def run_debug(image_path: str, k_clusters: int = 5, show_all: bool = True) -> None:
    print(f"[DEBUG-DINO] Using image: {image_path}")
    rgb = load_rgb(image_path)
    H, W, _ = rgb.shape
    print(f"[DEBUG-DINO] RGB shape: {rgb.shape}")

    print("[DEBUG-DINO] Initializing DINO model...")
    segmenter = dino.dino()

    print("[DEBUG-DINO] Preprocessing image for DINO...")
    img_tensor = preprocess_for_dino(rgb)

    print("[DEBUG-DINO] Extracting DINO features...")
    feats = segmenter.extract_features(img_tensor)
    patch_grid = segmenter.process_patch_tokens(feats)
    patch_features_np, H_p, W_p = segmenter.prepare_features_for_clustering(patch_grid)

    print("[DEBUG-DINO] Clustering patch features...")
    seg_small = segmenter.cluster_features(patch_features_np, H_p, W_p, k=k_clusters)
    print(f"[DEBUG-DINO] Patch segmentation shape: {seg_small.shape} (H_p={H_p}, W_p={W_p})")

    print("[DEBUG-DINO] Upsampling segmentation to full image size...")
    seg_full = cv2.resize(
        seg_small.astype(np.uint8),
        (W, H),
        interpolation=cv2.INTER_NEAREST,
    )
    print(f"[DEBUG-DINO] Upsampled segmentation shape: {seg_full.shape}")

    labels = np.unique(seg_full)
    print(f"[DEBUG-DINO] Unique labels in segmentation: {labels}")

    if not show_all:
        print("[DEBUG-DINO] show_all=False; not launching matplotlib windows.")
        return

    # Visualize each label separately to understand which one matches the object.
    for lbl in labels:
        mask = seg_full == lbl
        print(f"[DEBUG-DINO] Label {lbl}: pixels={mask.sum()}")
        show_mask_overlay(rgb, mask, alpha=0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug DINO segmentation labels and object masks on a single image."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to RGB image.")
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of clusters for DINO KMeans segmentation (same as run_full_pipeline.py --k).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not show matplotlib windows; only print label stats.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_debug(
        image_path=args.image,
        k_clusters=args.k,
        show_all=not args.no_show,
    )


if __name__ == "__main__":
    main()

