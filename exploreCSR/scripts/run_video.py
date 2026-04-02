#!/usr/bin/env python
"""
Run CAPTRA over sampled frames from a video file.

Usage
-----
python -m exploreCSR.scripts.run_video \\
    --video path/to/video.mp4 \\
    --weights path/to/depth_anything_v2_vitb.pth \\
    --prompt "bag" \\
    --fps 2.0 \\
    --out-csv results.csv
"""

from __future__ import annotations

import argparse

from exploreCSR.config import CameraConfig
from exploreCSR.pipeline import run_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CAPTRA on sampled video frames and export pose CSV."
    )
    parser.add_argument("--video", type=str, required=True, help="Input video file.")
    parser.add_argument("--weights", type=str, required=True, help="DepthAnything checkpoint.")
    parser.add_argument("--prompt", type=str, required=True, help="Segmentation prompt.")
    parser.add_argument("--out-csv", type=str, default="captra_video_pose.csv", help="Output CSV path.")
    parser.add_argument("--fps", type=float, default=2.0, help="Target processing FPS.")
    parser.add_argument("--fx", type=float, default=500.0)
    parser.add_argument("--fy", type=float, default=500.0)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--depth-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    camera = CameraConfig(
        fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
        depth_scale=args.depth_scale,
    )

    run_video(
        video_path=args.video,
        weights_path=args.weights,
        prompt=args.prompt,
        out_csv=args.out_csv,
        target_fps=args.fps,
        camera=camera,
    )


if __name__ == "__main__":
    main()
