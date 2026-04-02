#!/usr/bin/env python
"""
Run the BrownCSR pipeline on a single RGB image.

Usage
-----
python -m exploreCSR.scripts.run_single \\
    --image path/to/image.jpg \\
    --weights path/to/depth_anything_v2_vitb.pth \\
    --prompt "bag" \\
    --show
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from exploreCSR.config import CameraConfig
from exploreCSR.pipeline import run_single
from exploreCSR.visualization import (
    show_mask_overlay,
    show_masked_depth,
    show_pointcloud,
    show_reference_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BrownCSR pipeline on a single image."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to RGB image.")
    parser.add_argument("--weights", type=str, required=True, help="DepthAnything checkpoint.")
    parser.add_argument("--prompt", type=str, default=None, help="Segmentation prompt (None → unsupervised).")
    parser.add_argument("--fx", type=float, default=500.0)
    parser.add_argument("--fy", type=float, default=500.0)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--depth-scale", type=float, default=1.0)
    parser.add_argument("--show", action="store_true", help="Show matplotlib visualizations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    camera = CameraConfig(
        fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
        depth_scale=args.depth_scale,
    )

    out = run_single(
        image_path=args.image,
        weights_path=args.weights,
        prompt=args.prompt,
        camera=camera,
    )

    if args.show and out.get("valid"):
        import cv2

        rgb = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
        mask = out["mask"].astype(bool)

        show_mask_overlay(rgb, mask)
        show_masked_depth(out["masked_depth"], out["mask"])
        show_pointcloud(out["object_points"], title="Object Point Cloud")

        if out["object_centroid"] is not None and out["principal_axes"] is not None:
            show_reference_frame(
                out["object_points"],
                out["object_centroid"],
                out["principal_axes"],
                title="Object Reference Frame",
            )


if __name__ == "__main__":
    main()
