#!/usr/bin/env python
"""
Run CAPTRA over multiple RGB images in temporal order.

Usage
-----
python -m exploreCSR.scripts.run_sequence \\
    --images frame1.jpg frame2.jpg frame3.jpg \\
    --weights path/to/depth_anything_v2_vitb.pth \\
    --prompt "bag"

python -m exploreCSR.scripts.run_sequence \\
    --glob "frames/*.jpg" \\
    --weights path/to/checkpoint.pth \\
    --prompt "person" --no-viz
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List, Optional

from exploreCSR.config import CameraConfig
from exploreCSR.pipeline import run_sequence
from exploreCSR.visualization import (
    print_pose_summary,
    show_mask_overlay,
    show_masked_depth,
    show_pointcloud,
    show_reference_frame,
)


def resolve_image_paths(
    images: Optional[List[str]], pattern: Optional[str]
) -> List[str]:
    """Merge --images and --glob into a deduplicated, ordered list."""
    paths: List[str] = []
    if pattern:
        paths.extend(sorted(glob.glob(pattern)))
    if images:
        paths.extend(images)
    seen, unique = set(), []
    for p in paths:
        ap = os.path.normpath(os.path.abspath(p))
        if ap not in seen:
            seen.add(ap)
            unique.append(p)
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CAPTRA on multiple images in temporal order."
    )
    parser.add_argument("--images", nargs="*", default=None, help="Image paths in order.")
    parser.add_argument("--glob", dest="glob_pattern", default=None, help='Glob pattern, e.g. "frames/*.jpg".')
    parser.add_argument("--weights", type=str, required=True, help="DepthAnything checkpoint.")
    parser.add_argument("--prompt", type=str, required=True, help="Segmentation prompt.")
    parser.add_argument("--fx", type=float, default=500.0)
    parser.add_argument("--fy", type=float, default=500.0)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--depth-scale", type=float, default=1.0)
    parser.add_argument("--no-viz", action="store_true", help="Skip visualizations.")
    parser.add_argument("--viz-last-only", action="store_true", help="Only visualize the final frame.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = resolve_image_paths(args.images, args.glob_pattern)

    if not paths:
        print("No images found. Use --images and/or --glob.")
        return

    camera = CameraConfig(
        fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
        depth_scale=args.depth_scale,
    )

    results = run_sequence(
        image_paths=paths,
        weights_path=args.weights,
        prompt=args.prompt,
        camera=camera,
    )

    if not args.no_viz:
        import cv2
        import numpy as np

        for idx, (path, out) in enumerate(zip(paths, results)):
            do_viz = not args.viz_last_only or idx == len(results) - 1
            if do_viz and out.get("valid"):
                rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                mask = out["mask"].astype(bool)

                show_mask_overlay(rgb, mask)
                show_masked_depth(out["masked_depth"], out["mask"])
                show_pointcloud(out["object_points"], title=f"Frame {idx}")

                if out["object_centroid"] is not None and out["principal_axes"] is not None:
                    show_reference_frame(
                        out["object_points"],
                        out["object_centroid"],
                        out["principal_axes"],
                        title=f"Frame {idx} Reference",
                    )


if __name__ == "__main__":
    main()
