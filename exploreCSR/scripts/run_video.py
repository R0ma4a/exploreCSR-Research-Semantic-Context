#!/usr/bin/env python
"""
Run the tracked pipeline over sampled frames from a video file.

Usage
-----
python -m exploreCSR.scripts.run_video \
    --video path/to/video.mp4 \
    --weights path/to/depth_anything_v2_vitb.pth \
    --prompt "mug" \
    --rot-dir runs/6_mug_rot \
    --coord-dir runs/6_mug_coord \
    --category 6 \
    --fps 2.0 \
    --out-csv results.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from exploreCSR.pipeline import run_video_tracked
from exploreCSR.visualization import print_pose_summary, show_pointcloud

CATEGORY_NAMES = {
    1: "bottle", 2: "bowl", 3: "camera",
    4: "can",    5: "laptop", 6: "mug",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tracked CAPTRA pipeline on a video and export pose CSV."
    )
    parser.add_argument("--video",      type=str,   required=True, help="Input video file.")
    parser.add_argument("--weights",    type=str,   required=True, help="DepthAnything checkpoint.")
    parser.add_argument("--prompt",     type=str,   required=True, help="Segmentation prompt.")
    parser.add_argument("--captra-weights-dir", type=str, default=None, metavar="DIR",
                        help="Base runs/ directory. Auto-derives --rot-dir/--coord-dir from --category.")
    parser.add_argument("--rot-dir",    type=str,   default=None,  help="CAPTRA rotation net checkpoint dir.")
    parser.add_argument("--coord-dir",  type=str,   default=None,  help="CAPTRA coordinate net checkpoint dir.")
    parser.add_argument("--category",   type=int,   default=6,     help="NOCS category: 1=bottle 2=bowl 3=camera 4=can 5=laptop 6=mug.")
    parser.add_argument("--num-points", type=int,   default=4096,  help="Point cloud size per frame.")
    parser.add_argument("--fps",        type=float, default=2.0,   help="Target processing FPS.")
    parser.add_argument("--fx", type=float, default=591.0)
    parser.add_argument("--fy", type=float, default=590.0)
    parser.add_argument("--cx", type=float, default=322.0)
    parser.add_argument("--cy", type=float, default=244.0)
    parser.add_argument("--out-csv",    type=str,   default="captra_video_pose.csv", help="Output CSV path.")
    parser.add_argument("--no-viz",     action="store_true", help="Skip point cloud visualizations.")
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

    intrinsics = np.array(
        [[args.fx, 0.0, args.cx],
         [0.0, args.fy, args.cy],
         [0.0, 0.0,     1.0]],
        dtype=np.float64,
    )

    tracker = run_video_tracked(
        video_path  = args.video,
        weights_path= args.weights,
        prompt      = args.prompt,
        rot_dir     = args.rot_dir,
        coord_dir   = args.coord_dir,
        category    = args.category,
        target_fps  = args.fps,
        intrinsics  = intrinsics,
        num_points  = args.num_points,
    )

    tracker.to_csv(args.out_csv)
    print(f"Saved pose CSV: {args.out_csv}")

    if not args.no_viz:
        for frame in tracker.frames:
            if frame.valid and frame.point_cloud is not None:
                print_pose_summary({
                    "valid":           frame.valid,
                    "translation":     frame.translation,
                    "rotation_matrix": frame.rotation_matrix,
                    "scale":           frame.scale,
                })
                show_pointcloud(frame.point_cloud, title=f"Frame {frame.frame_idx}")


if __name__ == "__main__":
    main()
