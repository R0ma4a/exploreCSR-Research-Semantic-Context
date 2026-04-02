#!/usr/bin/env python
"""
Debug DINO-based segmentation on a single image.

Visualizes the generated mask overlay to help choose target labels
and tune segmentation parameters.

Usage
-----
python -m exploreCSR.scripts.debug_segmentation \\
    --image path/to/image.jpg \\
    --weights path/to/depth_anything_v2_vitb.pth
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from exploreCSR.depth import DepthAnything
from exploreCSR.segmentation import DINOSegmenter
from exploreCSR.visualization import show_mask_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug DINO segmentation on a single image."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to RGB image.")
    parser.add_argument("--weights", type=str, required=True, help="DepthAnything checkpoint.")
    parser.add_argument("--no-show", action="store_true", help="Skip matplotlib windows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[DEBUG] Image: {args.image}")

    rgb = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    H, W, _ = rgb.shape
    print(f"[DEBUG] RGB shape: {rgb.shape}")

    converter = DepthAnything(args.weights)
    segmenter = DINOSegmenter()

    image_tensor, _, original_w, original_h = converter.preprocess(args.image)

    depth_raw = converter.predict(image_tensor)
    depth_norm = converter.postprocess(depth_raw, original_w, original_h)

    mask = segmenter.generate_object_mask(
        image_tensor, depth_norm, (original_h, original_w)
    )
    print(f"[DEBUG] Mask shape: {mask.shape}, unique: {np.unique(mask)}")

    if not args.no_show:
        show_mask_overlay(rgb, mask.astype(bool), alpha=0.5)


if __name__ == "__main__":
    main()
