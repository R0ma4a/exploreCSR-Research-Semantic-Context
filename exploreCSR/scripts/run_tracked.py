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
import csv
import glob
import json
import os
from typing import List, Optional

import cv2
import numpy as np


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

    # Neural CAPTRA
    parser.add_argument("--rot-dir",   type=str, required=True,
                        help="CAPTRA rotation net checkpoint dir (e.g. RGed-research/captra/runs/6_mug_rot)")
    parser.add_argument("--coord-dir", type=str, required=True,
                        help="CAPTRA coordinate net checkpoint dir (e.g. RGed-research/captra/runs/6_mug_coord)")
    parser.add_argument("--category",  type=int, default=6,
                        help="NOCS category index: 1=bottle 2=bowl 3=camera 4=can 5=laptop 6=mug")

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
    parser.add_argument("--preview", action="store_true", help="Show live frame preview (mask overlay + pose) while processing.")
    parser.add_argument("--fit-coupling", action="store_true",
                        help="Fit and print the ICP rotation-translation coupling coefficient α. "
                             "Run this on a translation-only baseline, then pass the result as --coupling-alpha.")
    parser.add_argument("--coupling-alpha", type=float, default=0.0, metavar="ALPHA",
                        help="Coupling correction: subtract α × cumulative_translation from ICP "
                             "accumulated rotation (°/depth-unit). Get α from --fit-coupling on a "
                             "translation baseline.")
    parser.add_argument("--title", type=str, default=None, help="Custom plot title (e.g. 'Microphone 4fps').")

    # Axis limits (set these to the same values across runs for comparison)
    parser.add_argument("--ylim-feat", type=float, nargs=2, default=None, metavar=("MIN", "MAX"), help="Fixed y-axis for feature magnitude.")
    parser.add_argument("--ylim-trans", type=float, nargs=2, default=None, metavar=("MIN", "MAX"), help="Fixed y-axis for translation.")
    parser.add_argument("--ylim-rot", type=float, nargs=2, default=None, metavar=("MIN", "MAX"), help="Fixed y-axis for rotation.")
    parser.add_argument("--ylim-cos", type=float, nargs=2, default=None, metavar=("MIN", "MAX"), help="Fixed y-axis for cosine dissimilarity.")
    parser.add_argument("--xlim-trans", type=float, nargs=2, default=None, metavar=("MIN", "MAX"), help="Fixed x-axis for translation scatter.")
    parser.add_argument("--xlim-rot", type=float, nargs=2, default=None, metavar=("MIN", "MAX"), help="Fixed x-axis for rotation scatter.")
    parser.add_argument("--xlim-pose", type=float, nargs=2, default=None, metavar=("MIN", "MAX"), help="Fixed x-axis for combined pose scatter.")

    # Tracked correspondence analysis
    parser.add_argument("--track-world-points", action="store_true", help="Enable correspondence-based tracked-point analysis.")
    parser.add_argument("--max-track-points", type=int, default=300, help="Maximum tracked points initialized in frame 0 mask.")
    parser.add_argument("--track-min-distance", type=int, default=7, help="Min spacing between sampled track points.")
    parser.add_argument("--track-min-valid-ratio", type=float, default=0.7, help="Keep tracks valid for at least this fraction of frames.")
    parser.add_argument("--track-reference", type=str, choices=["prev", "first"], default="prev", help="Reference for feature deltas: previous valid frame or first valid frame.")
    parser.add_argument("--track-bidirectional-check", action="store_true", help="Use LK forward-backward consistency check.")
    parser.add_argument("--tracked-output-dir", type=str, default=None, help="Output directory for tracked-point artifacts.")

    return parser.parse_args()


def _load_video_frames(video_path: str, target_fps: float) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for tracking overlays: {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0 or np.isnan(native_fps):
        native_fps = 30.0
    step = max(1, int(round(native_fps / target_fps)))
    frames: List[np.ndarray] = []
    idx = -1
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        idx += 1
        if idx % step != 0:
            continue
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _derive_tracked_output_dir(args: argparse.Namespace) -> str:
    if args.tracked_output_dir:
        return args.tracked_output_dir
    safe_title = (args.title or args.prompt or "tracked_run").strip().replace(" ", "_")
    return os.path.join("outputs", safe_title, "tracked_points")


def _run_tracked_point_analysis(args: argparse.Namespace, tracker, frames_rgb: List[np.ndarray]) -> None:
    from exploreCSR.tracking import (
        compute_tracked_feature_deltas,
        extract_feature_vectors_at_tracked_points,
        filter_long_tracks,
        plot_track_overlay,
        plot_tracked_feature_metrics,
        plot_valid_track_counts,
        sample_track_points,
        summarize_tracked_feature_motion,
        track_points_through_video,
    )

    if len(tracker.frames) == 0:
        print("[TRACKED-POINTS] No frames found; skipping tracked-point analysis.")
        return

    masks = [np.asarray(fr.mask) if fr.mask is not None else None for fr in tracker.frames]
    if masks[0] is None or np.sum(masks[0]) == 0:
        print("[TRACKED-POINTS] Frame-0 mask missing/empty; skipping tracked-point analysis.")
        return
    if len(frames_rgb) != len(tracker.frames):
        print(f"[TRACKED-POINTS] Frame count mismatch ({len(frames_rgb)} images vs {len(tracker.frames)} tracker frames); skipping.")
        return

    init_points = sample_track_points(
        masks[0],
        max_points=args.max_track_points,
        min_distance=args.track_min_distance,
    )
    if init_points.shape[0] == 0:
        print("[TRACKED-POINTS] No points could be sampled from mask; skipping.")
        return

    tracks, valid_mask, diagnostics = track_points_through_video(
        frames=frames_rgb,
        init_points=init_points,
        object_masks=masks,
        bidirectional_check=args.track_bidirectional_check,
    )
    tracks_raw = tracks
    valid_mask_raw = valid_mask
    tracks, valid_mask, keep_mask = filter_long_tracks(
        tracks_raw, valid_mask_raw, min_valid_ratio=args.track_min_valid_ratio
    )
    if tracks.shape[1] == 0:
        valid_ratio = valid_mask_raw.mean(axis=0) if valid_mask_raw.size else np.array([])
        if valid_ratio.size == 0:
            print("[TRACKED-POINTS] No tracks available after tracking; skipping.")
            return
        fallback_min_ratio = 0.15
        fallback_keep = valid_ratio >= fallback_min_ratio
        if not np.any(fallback_keep):
            top_k = min(50, valid_ratio.shape[0])
            order = np.argsort(valid_ratio)[::-1]
            fallback_keep = np.zeros_like(valid_ratio, dtype=bool)
            fallback_keep[order[:top_k]] = True
        tracks = tracks_raw[:, fallback_keep]
        valid_mask = valid_mask_raw[:, fallback_keep]
        keep_mask = fallback_keep
        print(
            "[TRACKED-POINTS] No tracks met strict min-valid-ratio; "
            f"falling back to {tracks.shape[1]} tracks with ratio >= {fallback_min_ratio:.2f} or top survivors."
        )

    feature_maps = []
    image_size = tracker.frames[0].image_size
    for fr in tracker.frames:
        fmap = fr.patch_feature_map
        if fmap is None:
            print("[TRACKED-POINTS] Missing dense feature map in tracker records; skipping.")
            return
        feature_maps.append(np.asarray(fmap, dtype=np.float32))
    feature_maps_np = np.stack(feature_maps, axis=0)

    if image_size is None:
        print("[TRACKED-POINTS] Missing image_size metadata in tracker records; skipping.")
        return

    tracked_feats = extract_feature_vectors_at_tracked_points(
        feature_maps=feature_maps_np,
        tracks=tracks,
        mode="bilinear",
        image_size=image_size,
    )
    deltas = compute_tracked_feature_deltas(
        tracked_features=tracked_feats,
        valid_mask=valid_mask,
        reference=args.track_reference,
    )
    tracked_summary = summarize_tracked_feature_motion(deltas, valid_mask)

    output_dir = _derive_tracked_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)
    plot_track_overlay(
        frames=frames_rgb,
        tracks=tracks,
        valid_mask=valid_mask,
        save_path=os.path.join(output_dir, "track_overlay.png"),
    )
    plot_valid_track_counts(
        valid_mask=valid_mask,
        save_path=os.path.join(output_dir, "valid_track_counts.png"),
    )
    plot_tracked_feature_metrics(
        summary=tracked_summary,
        save_path=os.path.join(output_dir, "tracked_feature_metrics.png"),
    )

    with open(os.path.join(output_dir, "tracked_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "tracking_diagnostics": {
                    "retention_counts": diagnostics["retention_counts"].tolist(),
                    "round_trip_error_mean": diagnostics["round_trip_error_mean"].tolist(),
                    "round_trip_error_median": diagnostics["round_trip_error_median"].tolist(),
                    "num_init_points": int(init_points.shape[0]),
                    "num_tracks_after_filter": int(tracks.shape[1]),
                    "min_valid_ratio": float(args.track_min_valid_ratio),
                    "valid_ratio_after_filter_mean": float(valid_mask.mean()) if valid_mask.size else 0.0,
                    "kept_track_fraction": float(np.mean(keep_mask)) if keep_mask.size else 0.0,
                },
                "tracked_feature_summary": tracked_summary,
            },
            f,
            indent=2,
        )

    with open(os.path.join(output_dir, "tracked_per_frame.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_idx",
                "valid_tracks",
                "mean_cosine_dissimilarity",
                "median_cosine_dissimilarity",
                "mean_l2_delta",
                "median_l2_delta",
                "iqr_cosine_dissimilarity",
                "iqr_l2_delta",
            ],
        )
        writer.writeheader()
        writer.writerows(tracked_summary["per_frame"])

    region_deltas = tracker.compute_deltas()
    region_cos = region_deltas.get("cosine_similarity", np.array([]))
    region_feat = region_deltas.get("delta_feature_mag", np.array([]))
    valid_cos = region_cos[~np.isnan(region_cos)] if region_cos.size else np.array([])
    valid_feat = region_feat[~np.isnan(region_feat)] if region_feat.size else np.array([])
    seq_stats = tracked_summary["sequence"]
    comparison = {
        "region_mean": {
            "mean_cosine_dissimilarity": 1.0 - float(np.mean(valid_cos)) if valid_cos.size else np.nan,
            "mean_feature_delta_l2": float(np.mean(valid_feat)) if valid_feat.size else np.nan,
        },
        "tracked_correspondence": {
            "mean_cosine_dissimilarity": float(seq_stats.get("mean_cosine_dissimilarity", np.nan)),
            "median_cosine_dissimilarity": float(seq_stats.get("median_cosine_dissimilarity", np.nan)),
            "mean_l2_delta": float(seq_stats.get("mean_l2_delta", np.nan)),
            "median_l2_delta": float(seq_stats.get("median_l2_delta", np.nan)),
            "mean_valid_tracks_per_frame": float(seq_stats.get("mean_valid_tracks_per_frame", np.nan)),
        },
    }
    with open(os.path.join(output_dir, "comparison_summary.json"), "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("\n=== Region Mean vs Tracked Correspondence ===")
    print(f"Region-mean: mean L2 delta={comparison['region_mean']['mean_feature_delta_l2']:.6f}, "
          f"mean cosine dissim={comparison['region_mean']['mean_cosine_dissimilarity']:.6f}")
    print(f"Tracked:     mean L2 delta={comparison['tracked_correspondence']['mean_l2_delta']:.6f}, "
          f"median L2 delta={comparison['tracked_correspondence']['median_l2_delta']:.6f}, "
          f"mean cosine dissim={comparison['tracked_correspondence']['mean_cosine_dissimilarity']:.6f}, "
          f"valid tracks/frame={comparison['tracked_correspondence']['mean_valid_tracks_per_frame']:.2f}")
    print(f"[TRACKED-POINTS] Saved outputs to: {output_dir}")


def main() -> None:
    args = parse_args()

    from exploreCSR.pipeline import run_sequence_tracked, run_video_tracked

    paths: List[str] = []
    if args.video:
        # Video mode
        tracker = run_video_tracked(
            video_path   = args.video,
            weights_path = args.weights,
            prompt       = args.prompt,
            rot_dir      = args.rot_dir,
            coord_dir    = args.coord_dir,
            category     = args.category,
            target_fps   = args.fps,
            feature_key  = args.feature_key,
            show_preview = args.preview,
        )
    else:
        # Image sequence mode
        paths = resolve_image_paths(args.images, args.glob_pattern)
        if not paths:
            print("No images found. Use --images, --glob, or --video.")
            return

        tracker = run_sequence_tracked(
            image_paths  = paths,
            weights_path = args.weights,
            prompt       = args.prompt,
            rot_dir      = args.rot_dir,
            coord_dir    = args.coord_dir,
            category     = args.category,
            feature_key  = args.feature_key,
            show_preview = args.preview,
        )

    # Summary
    tracker.summary()

    # Coupling fit (translation baseline workflow)
    if args.fit_coupling:
        alpha = tracker.fit_coupling_alpha()
        print(f"\n[coupling] Use --coupling-alpha {alpha:.4f} on rotation runs to correct ICP.")

    # CSV
    if args.save_csv:
        tracker.to_csv(args.save_csv)

    # Plot
    if not args.no_plot:
        tracker.plot(
            save_path=args.save_plot,
            title=args.title,
            coupling_alpha=args.coupling_alpha,
            ylim_feat=tuple(args.ylim_feat) if args.ylim_feat else None,
            ylim_trans=tuple(args.ylim_trans) if args.ylim_trans else None,
            ylim_rot=tuple(args.ylim_rot) if args.ylim_rot else None,
        )

    if args.track_world_points:
        if args.video:
            frames_rgb = _load_video_frames(args.video, args.fps)
        else:
            frames_rgb = []
            for p in paths:
                bgr = cv2.imread(p, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise FileNotFoundError(f"Could not read image for tracking overlay: {p}")
                frames_rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        _run_tracked_point_analysis(args, tracker, frames_rgb)


if __name__ == "__main__":
    main()