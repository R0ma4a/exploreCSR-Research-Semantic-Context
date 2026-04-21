#!/usr/bin/env python
"""
Show DepthAnything depth, DINO mask, and object point cloud for a single image.

Usage
-----
python -m exploreCSR.scripts.run_single \
    --image path/to/image.jpg \
    --weights path/to/depth_anything_v2_vitb.pth \
    --prompt "mug" \
    --show
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from exploreCSR.depth import DepthAnything
from exploreCSR.pose.captra import build_point_cloud, depth_norm_to_metric, metric_to_uint16
from exploreCSR.segmentation import DINOSegmenter
from exploreCSR.visualization import show_mask_overlay, show_masked_depth, show_pointcloud


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show DepthAnything + DINO results on a single image."
    )
    parser.add_argument("--image",   type=str, required=True, help="Path to RGB image.")
    parser.add_argument("--weights", type=str, required=True, help="DepthAnything checkpoint.")
    parser.add_argument("--prompt",  type=str, default=None,  help="Segmentation prompt (None → unsupervised).")
    parser.add_argument("--fx", type=float, default=591.0)
    parser.add_argument("--fy", type=float, default=590.0)
    parser.add_argument("--cx", type=float, default=322.0)
    parser.add_argument("--cy", type=float, default=244.0)
    parser.add_argument("--show", action="store_true", help="Show matplotlib visualizations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    converter = DepthAnything(args.weights)
    segmenter = DINOSegmenter()

    image_tensor, rgb, original_w, original_h = converter.preprocess(args.image)

    depth_raw  = converter.predict(image_tensor)
    depth_norm = converter.postprocess(depth_raw, original_w, original_h)

    if args.prompt:
        mask = segmenter.segment_from_prompt(
            image_tensor, args.prompt,
            output_size=(original_h, original_w),
            depth_map=depth_norm,
        )
    else:
        mask = segmenter.generate_object_mask(
            image_tensor, depth_norm,
            output_size=(original_h, original_w),
        )

    print(f"Image size  : {original_w} x {original_h}")
    print(f"Depth range : [{depth_norm.min():.4f}, {depth_norm.max():.4f}]")
    print(f"Mask pixels : {mask.sum()} / {mask.size}  ({100*mask.sum()/mask.size:.1f}%)")

    if not args.show:
        return

    # 1 — full depth map
    plt.figure()
    plt.imshow(depth_norm, cmap="inferno")
    plt.colorbar(label="Normalized depth")
    plt.title("DepthAnything depth")
    plt.axis("off")
    plt.show()

    # 2 — DINO mask overlay on RGB
    show_mask_overlay(rgb, mask.astype(bool))

    # 3 — masked depth
    show_masked_depth(depth_norm, mask)

    # 4 — back-projected point cloud
    intrinsics = np.array(
        [[args.fx, 0.0, args.cx],
         [0.0, args.fy, args.cy],
         [0.0, 0.0,     1.0]],
        dtype=np.float64,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pcl = build_point_cloud(
        metric_to_uint16(depth_norm_to_metric(depth_norm)),
        mask,
        intrinsics,
        device=device,
    )
    if pcl is not None:
        show_pointcloud(pcl, title="Object Point Cloud")
    else:
        print("No valid depth points inside mask — skipping point cloud.")


if __name__ == "__main__":
    main()
