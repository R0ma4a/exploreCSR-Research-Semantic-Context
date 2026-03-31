#!/usr/bin/env python
"""
Run CAPTRA over multiple RGB images in sequence (temporal pipeline).

Like run_captra_only.py / main.py, each frame uses DepthAnything for depth and
DINO segment_from_prompt for prompt-guided masking, then CAPTRA.forward. After
the first frame, CAPTRA receives
`previous_reference_state` from the previous frame so pose outputs reflect *change*
(translation, rotation, scale) for the graph x-axis, not merely identity on every image.

First frame: translation ~0, rotation ~identity, scale ~1 (initialization).
Later frames: deltas vs. the stored object reference when geometry is valid.

Usage
-----
python captra_multiple_images.py \\
    --images path/to/a.jpg path/to/b.jpg path/to/c.jpg \\
    --weights "C:\\Users\\you\\Downloads\\depth_anything_v2_vitb.pth" \\
    --prompt "bag"

python captra_multiple_images.py \\
    --images img1.jpg img2.jpg \\
    --prompt "person" \\
    --no-viz
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List, Optional

import cv2
import numpy as np


def _add_rged_to_path() -> None:
    root_dir = os.path.dirname(os.path.abspath(__file__))
    rged_dir = os.path.join(root_dir, "RGed-research")
    if rged_dir not in sys.path:
        sys.path.insert(0, rged_dir)


_add_rged_to_path()

import depth_anything  # type: ignore
import dino  # type: ignore
from captra import CAPTRA, CAPTRAReferenceState  # type: ignore
from captra_viz import (  # type: ignore
    print_pose_summary,
    show_mask_overlay,
    show_masked_depth,
    show_pointcloud,
    show_reference_frame,
)


def apply_scale_relative_to_anchor(
    output_dict: dict,
    anchor_extent_mean: Optional[float],
) -> tuple[dict, Optional[float]]:
    """
    Re-express CAPTRA scale relative to the first valid reference frame.
    """
    ref = output_dict.get("reference_state")
    if ref is not None and anchor_extent_mean is None:
        anchor_extent_mean = float(np.mean(ref.extents))

    if ref is None or anchor_extent_mean is None:
        return output_dict, anchor_extent_mean

    curr_extent_mean = float(np.mean(ref.extents))
    denom = max(anchor_extent_mean, 1e-8)
    scale_anchor = curr_extent_mean / denom

    output_dict["scale"] = scale_anchor
    pose_vector = output_dict.get("pose_vector")
    if pose_vector is not None and len(pose_vector) >= 7:
        pose_vector = np.asarray(pose_vector).copy()
        pose_vector[6] = scale_anchor
        output_dict["pose_vector"] = pose_vector
    return output_dict, anchor_extent_mean


def load_rgb_for_captra(image_path: str) -> np.ndarray:
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def resolve_image_paths(images: Optional[List[str]], pattern: Optional[str]) -> List[str]:
    paths: List[str] = []
    if pattern:
        paths.extend(sorted(glob.glob(pattern)))
    if images:
        paths.extend(images)
    # De-dupe while preserving order
    seen = set()
    unique: List[str] = []
    for p in paths:
        ap = os.path.normpath(os.path.abspath(p))
        if ap not in seen:
            seen.add(ap)
            unique.append(p)
    return unique


def process_sequence(
    image_paths: List[str],
    weights_path: str,
    prompt: str,
    fx: float,
    fy: float,
    cx: Optional[float],
    cy: Optional[float],
    depth_scale: float,
    show_viz: bool,
    viz_last_only: bool,
) -> None:
    if not image_paths:
        raise ValueError("No images to process. Use --images and/or --glob.")

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"DepthAnything weights not found: {weights_path}")

    for p in image_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Image file not found: {p}")

    print(f"[CAPTRA-MULTI] {len(image_paths)} frame(s)")
    print(f"[CAPTRA-MULTI] DepthAnything checkpoint: {weights_path}")
    print(f"[CAPTRA-MULTI] Prompt (segment_from_prompt): {prompt!r}")

    converter = depth_anything.DepthAnything(weights_path)
    segmenter = dino.dino()

    # Intrinsics: use first image size as reference defaults (each frame may differ)
    prev_state: Optional[CAPTRAReferenceState] = None
    captra: Optional[CAPTRA] = None
    anchor_extent_mean: Optional[float] = None

    for idx, image_path in enumerate(image_paths):
        print(f"\n{'='*60}\n=== Frame {idx}: {image_path}\n{'='*60}")

        image_tensor, _, original_width, original_height = converter.image_to_tensor(
            image_path
        )

        rgb = load_rgb_for_captra(image_path)
        if rgb.shape[0] != original_height or rgb.shape[1] != original_width:
            rgb = cv2.resize(
                rgb,
                (original_width, original_height),
                interpolation=cv2.INTER_LINEAR,
            )
        H, W, _ = rgb.shape

        cx_use = cx if cx is not None else W / 2.0
        cy_use = cy if cy is not None else H / 2.0
        if captra is None:
            K = np.array(
                [[fx, 0.0, cx_use], [0.0, fy, cy_use], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            captra = CAPTRA(camera_intrinsics=K, depth_scale=depth_scale)
        else:
            captra.K[0, 2] = cx_use  # type: ignore[union-attr]
            captra.K[1, 2] = cy_use  # type: ignore[union-attr]

        depth_raw = converter.predict_depth(image_tensor)
        depth_norm = converter.process_depth(
            depth_raw, original_width, original_height
        )

        mask = segmenter.segment_from_prompt(
            image_tensor,
            prompt,
            output_size=(original_height, original_width),
        )

        assert captra is not None
        out = captra.forward(
            rgb=rgb,
            depth=depth_norm,
            seg_or_mask=mask,
            target_label=None,
            previous_reference_state=prev_state,
        )
        out, anchor_extent_mean = apply_scale_relative_to_anchor(out, anchor_extent_mean)

        print("\n=== CAPTRA Pose Summary ===")
        print_pose_summary(out)

        region_diag = out.get("diagnostics", {}).get("region", {})
        print("\n[CAPTRA-MULTI] Region diagnostics:")
        print(f"  mask pixels: {region_diag.get('num_mask_pixels', 'NA')}")
        print(f"  valid depth points: {region_diag.get('num_valid_depth', 'NA')}")
        print(f"  valid flag: {out.get('valid')}")
        print(f"  message: {out.get('message')}")

        ref = out.get("reference_state")
        if ref is not None:
            prev_state = ref
        else:
            print(
                "[CAPTRA-MULTI] No reference_state this frame; "
                "keeping previous reference for next frame."
            )

        do_viz = show_viz and (not viz_last_only or idx == len(image_paths) - 1)
        if do_viz and out.get("valid"):
            m = out["mask"].astype(bool)
            print("[CAPTRA-MULTI] Visualization (this frame)...")
            show_mask_overlay(rgb, m)
            show_masked_depth(out["masked_depth"], out["mask"])
            pts = out["object_points"]
            cen = out["object_centroid"]
            ax = out["principal_axes"]
            show_pointcloud(pts, title=f"Frame {idx} point cloud", max_points=50000)
            if cen is not None and ax is not None:
                show_reference_frame(
                    pts,
                    cen,
                    ax,
                    title=f"Frame {idx} reference frame",
                    max_points=50000,
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run CAPTRA on multiple images in order; chain reference state so "
            "pose summary shows inter-frame change after frame 0."
        )
    )
    parser.add_argument(
        "--images",
        nargs="*",
        default=None,
        help="One or more image paths, in temporal order.",
    )
    parser.add_argument(
        "--glob",
        dest="glob_pattern",
        default=None,
        metavar="PATTERN",
        help='Optional glob (e.g. "frames/*.jpg"); merged with --images, sorted.',
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=r"C:\Users\roman\Downloads\depth_anything_v2_vitb.pth",
        help="Path to DepthAnything .pth checkpoint.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help=(
            "Text prompt for dino.segment_from_prompt (same as RGed-research/main.py), "
            'e.g. "bag", "person".'
        ),
    )
    parser.add_argument("--fx", type=float, default=500.0)
    parser.add_argument("--fy", type=float, default=500.0)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--depth-scale", type=float, default=1.0)
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip matplotlib (recommended for many frames).",
    )
    parser.add_argument(
        "--viz-last-only",
        action="store_true",
        help="If visualizing, only show plots for the final frame.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = resolve_image_paths(args.images, args.glob_pattern)
    process_sequence(
        image_paths=paths,
        weights_path=args.weights,
        prompt=args.prompt,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        depth_scale=args.depth_scale,
        show_viz=not args.no_viz,
        viz_last_only=args.viz_last_only,
    )


if __name__ == "__main__":
    main()
