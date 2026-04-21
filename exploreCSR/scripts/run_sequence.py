#!/usr/bin/env python
"""
Run the tracked pipeline over multiple RGB images in temporal order.

Usage
-----
python -m exploreCSR.scripts.run_sequence \
    --images frame1.jpg frame2.jpg frame3.jpg \
    --weights path/to/depth_anything_v2_vitb.pth \
    --prompt "mug" \
    --rot-dir runs/6_mug_rot \
    --coord-dir runs/6_mug_coord \
    --category 6

python -m exploreCSR.scripts.run_sequence \
    --glob "frames/*.jpg" \
    --weights path/to/checkpoint.pth \
    --prompt "bottle" \
    --rot-dir runs/1_bottle_rot \
    --coord-dir runs/1_bottle_coord \
    --category 1 \
    --no-viz \
    --out-csv results.csv
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List, Optional

CATEGORY_NAMES = {
    1: "bottle", 2: "bowl", 3: "camera",
    4: "can",    5: "laptop", 6: "mug",
}

import cv2
import numpy as np

from exploreCSR.pipeline import run_sequence_tracked
from exploreCSR.visualization import (
    print_pose_summary,
    show_mask_overlay,
    show_masked_depth,
    show_pointcloud,
)


def resolve_image_paths(
    images: Optional[List[str]], pattern: Optional[str]
) -> List[str]:
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
        description="Run tracked CAPTRA pipeline on multiple images."
    )
    parser.add_argument("--images", nargs="*", default=None, help="Image paths in temporal order.")
    parser.add_argument("--glob", dest="glob_pattern", default=None, help='Glob pattern, e.g. "frames/*.jpg".')
    parser.add_argument("--weights",    type=str, required=True, help="DepthAnything checkpoint.")
    parser.add_argument("--prompt",     type=str, required=True, help="Segmentation prompt.")
    parser.add_argument("--captra-weights-dir", type=str, default=None, metavar="DIR",
                        help="Base runs/ directory. Auto-derives --rot-dir/--coord-dir from --category.")
    parser.add_argument("--rot-dir",    type=str, default=None,  help="CAPTRA rotation net checkpoint dir.")
    parser.add_argument("--coord-dir",  type=str, default=None,  help="CAPTRA coordinate net checkpoint dir.")
    parser.add_argument("--category",   type=int, default=6,     help="NOCS category: 1=bottle 2=bowl 3=camera 4=can 5=laptop 6=mug.")
    parser.add_argument("--num-points", type=int, default=4096,  help="Point cloud size per frame.")
    parser.add_argument("--fx", type=float, default=591.0)
    parser.add_argument("--fy", type=float, default=590.0)
    parser.add_argument("--cx", type=float, default=322.0)
    parser.add_argument("--cy", type=float, default=244.0)
    parser.add_argument("--no-viz",       action="store_true", help="Skip per-frame visualizations.")
    parser.add_argument("--viz-last-only",action="store_true", help="Only visualize the final frame.")
    parser.add_argument("--out-csv",    type=str, default=None,  help="Save pose CSV to this path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.captra_weights_dir and not (args.rot_dir and args.coord_dir):
        cat_name = CATEGORY_NAMES.get(args.category, str(args.category))
        args.rot_dir   = os.path.join(args.captra_weights_dir, f"{args.category}_{cat_name}_rot")
        args.coord_dir = os.path.join(args.captra_weights_dir, f"{args.category}_{cat_name}_coord")
    if not args.rot_dir or not args.coord_dir:
        raise SystemExit(
            "Error: supply either --captra-weights-dir or both --rot-dir and --coord-dir."
        )

    paths = resolve_image_paths(args.images, args.glob_pattern)

    if not paths:
        print("No images found. Use --images and/or --glob.")
        return

    intrinsics = np.array(
        [[args.fx, 0.0, args.cx],
         [0.0, args.fy, args.cy],
         [0.0, 0.0,     1.0]],
        dtype=np.float64,
    )

    tracker = run_sequence_tracked(
        image_paths = paths,
        weights_path= args.weights,
        prompt      = args.prompt,
        rot_dir     = args.rot_dir,
        coord_dir   = args.coord_dir,
        category    = args.category,
        intrinsics  = intrinsics,
        num_points  = args.num_points,
    )

    if args.out_csv:
        tracker.to_csv(args.out_csv)
        print(f"Saved pose CSV: {args.out_csv}")

    if not args.no_viz:
        for i, frame in enumerate(tracker.frames):
            do_viz = not args.viz_last_only or i == len(tracker.frames) - 1
            if not (do_viz and frame.valid):
                continue

            print(f"\n--- Frame {frame.frame_idx} ---")
            print_pose_summary({
                "valid":           frame.valid,
                "translation":     frame.translation,
                "rotation_matrix": frame.rotation_matrix,
                "scale":           frame.scale,
            })

            if frame.image_path and os.path.isfile(frame.image_path):
                bgr = cv2.imread(frame.image_path)
                if bgr is not None:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    if frame.mask is not None:
                        show_mask_overlay(rgb, frame.mask.astype(bool))

            if frame.point_cloud is not None:
                show_pointcloud(frame.point_cloud, title=f"Frame {frame.frame_idx}")


if __name__ == "__main__":
    main()
