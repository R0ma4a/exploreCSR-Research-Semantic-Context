#!/usr/bin/env python
"""
Run CAPTRA + DINO feature tracking over an image sequence,
then plot delta features vs delta pose.

Usage
-----
python -m exploreCSR.scripts.run_tracked \\
    --images frame1.jpg frame2.jpg frame3.jpg \\
    --weights path/to/depth_anything_v2_vitb.pth \\
    --prompt "bag"

python -m exploreCSR.scripts.run_tracked \\
    --glob "frames/*.jpg" \\
    --weights checkpoint.pth \\
    --prompt "person" \\
    --save-plot results.png \\
    --save-csv results.csv

python -m exploreCSR.scripts.run_tracked \\
    --video clip.mp4 \\
    --weights checkpoint.pth \\
    --prompt "bag" \\
    --fps 2.0
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List, Optional


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
        description="Run CAPTRA + DINO feature tracking and plot deltas."
    )

    # Input (images or video)
    parser.add_argument("--images", nargs="*", default=None, help="Image paths in order.")
    parser.add_argument("--glob", dest="glob_pattern", default=None, help='Glob pattern, e.g. "frames/*.jpg".')
    parser.add_argument("--video", type=str, default=None, help="Video file (alternative to --images).")
    parser.add_argument("--fps", type=float, default=2.0, help="Target FPS for video mode.")

    # Model
    parser.add_argument("--weights", type=str, required=True, help="DepthAnything checkpoint.")
    parser.add_argument("--prompt", type=str, required=True, help="Segmentation prompt.")

    # Camera
    parser.add_argument("--fx", type=float, default=500.0)
    parser.add_argument("--fy", type=float, default=500.0)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--depth-scale", type=float, default=1.0)

    # Feature config
    parser.add_argument(
        "--feature-key", type=str, default="object_mean",
        choices=["object_mean", "cls_token"],
        help="Which DINO feature to track (default: object_mean = masked patch average).",
    )

    # Output
    parser.add_argument("--save-plot", type=str, default=None, help="Save plot to file instead of showing.")
    parser.add_argument("--save-csv", type=str, default=None, help="Export per-frame data to CSV.")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting, only print summary.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from browncsr.config import CameraConfig
    from browncsr.pipeline import run_sequence_tracked, run_video_tracked

    camera = CameraConfig(
        fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
        depth_scale=args.depth_scale,
    )

    if args.video:
        # Video mode
        tracker = run_video_tracked(
            video_path=args.video,
            weights_path=args.weights,
            prompt=args.prompt,
            target_fps=args.fps,
            camera=camera,
            feature_key=args.feature_key,
        )
    else:
        # Image sequence mode
        paths = resolve_image_paths(args.images, args.glob_pattern)
        if not paths:
            print("No images found. Use --images, --glob, or --video.")
            return

        tracker = run_sequence_tracked(
            image_paths=paths,
            weights_path=args.weights,
            prompt=args.prompt,
            camera=camera,
            feature_key=args.feature_key,
        )

    # Summary
    tracker.summary()

    # CSV
    if args.save_csv:
        tracker.to_csv(args.save_csv)

    # Plot
    if not args.no_plot:
        tracker.plot(save_path=args.save_plot)


if __name__ == "__main__":
    main()
