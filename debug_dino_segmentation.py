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

import depth_anything  # type: ignore
import dino  # type: ignore
from captra_viz import show_mask_overlay  # type: ignore


def load_rgb(image_path: str) -> np.ndarray:
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def run_debug(image_path: str, show_all: bool = True) -> None:
    print(f"[DEBUG-DINO] Using image: {image_path}")
    rgb = load_rgb(image_path)
    H, W, _ = rgb.shape
    print(f"[DEBUG-DINO] RGB shape: {rgb.shape}")

    print("[DEBUG-DINO] Initializing DepthAnything...")
    # For debugging we don't expose weights via CLI; reuse the same default
    # as other scripts if needed by editing this path.
    # Alternatively, you can adapt this script to accept --weights.
    # Here we assume the checkpoint path is known to the user.
    # NOTE: To keep this script simple, you can temporarily hard-code your
    # own checkpoint path here if needed.
    raise_path_msg = (
        "debug_dino_segmentation.py currently expects you to edit the "
        "DepthAnything checkpoint path in the script if you want to use "
        "the CAPTRA-ready mask. Alternatively, run main.py for a full "
        "example. "
    )
    # The actual path should match RGed-research/main.py
    weights_path = r"C:\Users\roman\Downloads\depth_anything_v2_vitb.pth"
    converter = depth_anything.DepthAnything(weights_path)

    print("[DEBUG-DINO] Initializing DINO segmenter...")
    segmenter = dino.dino()

    print("[DEBUG-DINO] Preprocessing image for DepthAnything...")
    image_tensor, _, original_width, original_height = converter.image_to_tensor(image_path)

    print("[DEBUG-DINO] Predicting depth with DepthAnything...")
    depth_raw = converter.predict_depth(image_tensor)
    depth_norm = converter.process_depth(depth_raw, original_width, original_height)

    print("[DEBUG-DINO] Generating CAPTRA-ready object mask with DINO...")
    mask = segmenter.generate_object_mask(
        image_tensor,
        depth_norm,
        (original_height, original_width),
    )
    print(f"[DEBUG-DINO] Mask shape: {mask.shape}, unique values: {np.unique(mask)}")

    if not show_all:
        print("[DEBUG-DINO] show_all=False; not launching matplotlib windows.")
        return

    # Visualize the CAPTRA-ready mask overlay.
    show_mask_overlay(rgb, mask.astype(bool), alpha=0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug DINO segmentation labels and object masks on a single image."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to RGB image.")
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
        show_all=not args.no_show,
    )


if __name__ == "__main__":
    main()

