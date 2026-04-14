"""
Tracked point correspondence analysis for object feature motion.

This module tracks 2D points across frames (inside object masks), samples dense
feature vectors at corresponding locations, and computes correspondence-aware
feature deltas over time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TrackDiagnostics:
    retention_counts: np.ndarray
    round_trip_error_mean: np.ndarray
    round_trip_error_median: np.ndarray


def _to_gray_u8(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        gray = frame
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def sample_track_points(
    mask: np.ndarray,
    max_points: int = 300,
    min_distance: int = 7,
    quality_level: float = 0.01,
) -> np.ndarray:
    """Sample 2D points inside a binary object mask."""
    if mask is None:
        return np.zeros((0, 2), dtype=np.float32)

    mask_u8 = (mask > 0).astype(np.uint8) * 255
    if mask_u8.sum() == 0:
        return np.zeros((0, 2), dtype=np.float32)

    corners = cv2.goodFeaturesToTrack(
        image=mask_u8,
        maxCorners=max_points,
        qualityLevel=quality_level,
        minDistance=min_distance,
        mask=mask_u8,
        blockSize=7,
        useHarrisDetector=False,
    )
    if corners is not None and len(corners) > 0:
        pts = corners.reshape(-1, 2).astype(np.float32)
        return pts[:max_points]

    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    rng = np.random.default_rng(0)
    order = rng.permutation(len(xs))
    selected: List[Tuple[float, float]] = []
    min_dist_sq = float(min_distance * min_distance)
    for idx in order:
        x = float(xs[idx])
        y = float(ys[idx])
        keep = True
        for sx, sy in selected:
            if (sx - x) ** 2 + (sy - y) ** 2 < min_dist_sq:
                keep = False
                break
        if keep:
            selected.append((x, y))
        if len(selected) >= max_points:
            break
    if not selected:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(selected, dtype=np.float32)


def track_points_through_video(
    frames: Sequence[np.ndarray],
    init_points: np.ndarray,
    object_masks: Optional[Sequence[np.ndarray]] = None,
    bidirectional_check: bool = True,
    fb_threshold: float = 1.5,
    mask_margin: int = 3,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Track points with pyramidal LK optical flow.

    Returns:
      tracks: [T, N, 2], NaN for invalid points
      valid_mask: [T, N] boolean validity mask
      diagnostics: retention and round-trip stats
    """
    if len(frames) == 0 or init_points.size == 0:
        return (
            np.full((len(frames), 0, 2), np.nan, dtype=np.float32),
            np.zeros((len(frames), 0), dtype=bool),
            {
                "retention_counts": np.zeros((len(frames),), dtype=np.int32),
                "round_trip_error_mean": np.full((max(0, len(frames) - 1),), np.nan),
                "round_trip_error_median": np.full((max(0, len(frames) - 1),), np.nan),
            },
        )

    gray_frames = [_to_gray_u8(f) for f in frames]
    t_count = len(frames)
    n_points = init_points.shape[0]

    tracks = np.full((t_count, n_points, 2), np.nan, dtype=np.float32)
    valid = np.zeros((t_count, n_points), dtype=bool)
    tracks[0] = init_points
    valid[0] = True

    retention_counts = np.zeros((t_count,), dtype=np.int32)
    retention_counts[0] = int(valid[0].sum())
    fb_mean = np.full((max(0, t_count - 1),), np.nan, dtype=np.float32)
    fb_median = np.full((max(0, t_count - 1),), np.nan, dtype=np.float32)

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    for t in range(t_count - 1):
        prev_pts = tracks[t].reshape(-1, 1, 2).astype(np.float32)
        prev_valid = valid[t].copy()
        if not np.any(prev_valid):
            retention_counts[t + 1] = 0
            continue

        p1, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
            gray_frames[t], gray_frames[t + 1], prev_pts, None, **lk_params
        )
        st_fwd = (st_fwd.reshape(-1) == 1)
        curr_valid = prev_valid & st_fwd
        curr_pts = p1.reshape(-1, 2)
        curr_valid = curr_valid & np.isfinite(curr_pts).all(axis=1)

        if bidirectional_check:
            p0_back, st_back, _ = cv2.calcOpticalFlowPyrLK(
                gray_frames[t + 1], gray_frames[t], p1, None, **lk_params
            )
            st_back = (st_back.reshape(-1) == 1)
            p0_back = p0_back.reshape(-1, 2)
            round_trip_err = np.linalg.norm(p0_back - prev_pts.reshape(-1, 2), axis=1)
            fb_ok = round_trip_err <= fb_threshold
            curr_valid = curr_valid & st_back & fb_ok

            vals = round_trip_err[curr_valid]
            if vals.size > 0:
                fb_mean[t] = float(np.mean(vals))
                fb_median[t] = float(np.median(vals))

        if object_masks is not None and t + 1 < len(object_masks):
            m = (object_masks[t + 1] > 0)
            if mask_margin > 0:
                ksz = 2 * int(mask_margin) + 1
                kernel = np.ones((ksz, ksz), dtype=np.uint8)
                m = cv2.dilate(m.astype(np.uint8), kernel, iterations=1).astype(bool)
            h, w = m.shape
            finite = np.isfinite(curr_pts).all(axis=1)
            x = np.zeros((curr_pts.shape[0],), dtype=int)
            y = np.zeros((curr_pts.shape[0],), dtype=int)
            if np.any(finite):
                x[finite] = np.round(curr_pts[finite, 0]).astype(int)
                y[finite] = np.round(curr_pts[finite, 1]).astype(int)
            in_bounds = finite & (x >= 0) & (x < w) & (y >= 0) & (y < h)
            inside = np.zeros_like(curr_valid)
            inside[in_bounds] = m[y[in_bounds], x[in_bounds]]
            curr_valid = curr_valid & inside

        tracks[t + 1, curr_valid] = curr_pts[curr_valid]
        valid[t + 1, curr_valid] = True
        retention_counts[t + 1] = int(curr_valid.sum())

    diagnostics = {
        "retention_counts": retention_counts,
        "round_trip_error_mean": fb_mean,
        "round_trip_error_median": fb_median,
    }
    return tracks, valid, diagnostics


def filter_long_tracks(
    tracks: np.ndarray,
    valid_mask: np.ndarray,
    min_valid_ratio: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep only points valid for at least min_valid_ratio of frames."""
    if valid_mask.size == 0:
        keep = np.zeros((0,), dtype=bool)
        return tracks, valid_mask, keep
    valid_ratio = valid_mask.mean(axis=0)
    keep = valid_ratio >= float(min_valid_ratio)
    return tracks[:, keep], valid_mask[:, keep], keep


def _bilinear_sample_feature_map(feature_map: np.ndarray, xy: np.ndarray) -> np.ndarray:
    c, hf, wf = feature_map.shape
    x = xy[:, 0]
    y = xy[:, 1]
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, wf - 1)
    y1 = np.clip(y0 + 1, 0, hf - 1)
    x0 = np.clip(x0, 0, wf - 1)
    y0 = np.clip(y0, 0, hf - 1)
    wx = np.clip(x - x0, 0.0, 1.0)
    wy = np.clip(y - y0, 0.0, 1.0)

    f00 = feature_map[:, y0, x0].T
    f01 = feature_map[:, y1, x0].T
    f10 = feature_map[:, y0, x1].T
    f11 = feature_map[:, y1, x1].T
    w00 = (1 - wx) * (1 - wy)
    w01 = (1 - wx) * wy
    w10 = wx * (1 - wy)
    w11 = wx * wy
    return (f00 * w00[:, None]) + (f01 * w01[:, None]) + (f10 * w10[:, None]) + (f11 * w11[:, None])


def extract_feature_vectors_at_tracked_points(
    feature_maps: np.ndarray,
    tracks: np.ndarray,
    mode: str = "bilinear",
    image_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Sample dense feature maps at tracked coordinates.

    feature_maps: [T, C, Hf, Wf]
    tracks: [T, N, 2] in image coordinates.
    returns: [T, N, C]
    """
    if mode != "bilinear":
        raise ValueError("Only bilinear mode is supported.")

    if feature_maps.ndim != 4:
        raise ValueError(f"feature_maps must be [T,C,Hf,Wf], got {feature_maps.shape}")
    t_count, c_dim, hf, wf = feature_maps.shape
    if tracks.shape[0] != t_count:
        raise ValueError("tracks/frame count mismatch with feature_maps.")

    if image_size is None:
        raise ValueError("image_size=(H,W) required for coordinate scaling.")
    img_h, img_w = image_size
    sx = (wf - 1) / max(img_w - 1, 1)
    sy = (hf - 1) / max(img_h - 1, 1)

    n_points = tracks.shape[1]
    out = np.full((t_count, n_points, c_dim), np.nan, dtype=np.float32)
    for t in range(t_count):
        xy = tracks[t].copy()
        valid = ~np.isnan(xy).any(axis=1)
        if not np.any(valid):
            continue
        xy_f = xy[valid]
        xy_f[:, 0] *= sx
        xy_f[:, 1] *= sy
        sampled = _bilinear_sample_feature_map(feature_maps[t], xy_f)
        out[t, valid] = sampled.astype(np.float32)
    return out


def compute_tracked_feature_deltas(
    tracked_features: np.ndarray,
    valid_mask: np.ndarray,
    reference: str = "prev",
) -> Dict[str, np.ndarray]:
    """Compute per-point cosine dissimilarity and L2 deltas."""
    if reference not in {"prev", "first"}:
        raise ValueError("reference must be 'prev' or 'first'")

    t_count, n_points, _ = tracked_features.shape
    cos_dis = np.full((t_count, n_points), np.nan, dtype=np.float32)
    l2_delta = np.full((t_count, n_points), np.nan, dtype=np.float32)

    first_valid_idx = np.full((n_points,), -1, dtype=np.int32)
    for n in range(n_points):
        idx = np.where(valid_mask[:, n])[0]
        if idx.size > 0:
            first_valid_idx[n] = int(idx[0])

    for t in range(1, t_count):
        for n in range(n_points):
            if not valid_mask[t, n]:
                continue
            if reference == "prev":
                prev = t - 1
                if not valid_mask[prev, n]:
                    continue
            else:
                prev = first_valid_idx[n]
                if prev < 0 or prev == t or (not valid_mask[prev, n]):
                    continue
            f0 = tracked_features[prev, n]
            f1 = tracked_features[t, n]
            if np.isnan(f0).any() or np.isnan(f1).any():
                continue
            denom = float(np.linalg.norm(f0) * np.linalg.norm(f1))
            cos_sim = float(np.dot(f0, f1) / denom) if denom > 1e-8 else np.nan
            cos_dis[t, n] = 1.0 - cos_sim if not np.isnan(cos_sim) else np.nan
            l2_delta[t, n] = float(np.linalg.norm(f1 - f0))

    return {"cosine_dissimilarity": cos_dis, "l2_delta": l2_delta}


def summarize_tracked_feature_motion(
    delta_metrics: Dict[str, np.ndarray],
    valid_mask: np.ndarray,
) -> Dict[str, object]:
    """Aggregate frame-level and sequence-level stats over valid correspondences."""
    cos = delta_metrics["cosine_dissimilarity"]
    l2 = delta_metrics["l2_delta"]
    t_count = cos.shape[0]

    per_frame: List[Dict[str, float]] = []
    for t in range(t_count):
        cos_vals = cos[t][~np.isnan(cos[t])]
        l2_vals = l2[t][~np.isnan(l2[t])]
        row: Dict[str, float] = {
            "frame_idx": float(t),
            "valid_tracks": float(np.sum(valid_mask[t])),
            "mean_cosine_dissimilarity": float(np.mean(cos_vals)) if cos_vals.size else np.nan,
            "median_cosine_dissimilarity": float(np.median(cos_vals)) if cos_vals.size else np.nan,
            "mean_l2_delta": float(np.mean(l2_vals)) if l2_vals.size else np.nan,
            "median_l2_delta": float(np.median(l2_vals)) if l2_vals.size else np.nan,
            "iqr_cosine_dissimilarity": float(np.percentile(cos_vals, 75) - np.percentile(cos_vals, 25))
            if cos_vals.size
            else np.nan,
            "iqr_l2_delta": float(np.percentile(l2_vals, 75) - np.percentile(l2_vals, 25))
            if l2_vals.size
            else np.nan,
        }
        per_frame.append(row)

    seq_cos = cos[~np.isnan(cos)]
    seq_l2 = l2[~np.isnan(l2)]
    sequence = {
        "num_frames": int(t_count),
        "num_points": int(valid_mask.shape[1]),
        "mean_valid_tracks_per_frame": float(np.mean(np.sum(valid_mask, axis=1))) if valid_mask.size else 0.0,
        "mean_cosine_dissimilarity": float(np.mean(seq_cos)) if seq_cos.size else np.nan,
        "median_cosine_dissimilarity": float(np.median(seq_cos)) if seq_cos.size else np.nan,
        "mean_l2_delta": float(np.mean(seq_l2)) if seq_l2.size else np.nan,
        "median_l2_delta": float(np.median(seq_l2)) if seq_l2.size else np.nan,
    }
    return {"per_frame": per_frame, "sequence": sequence}


def plot_track_overlay(
    frames: Sequence[np.ndarray],
    tracks: np.ndarray,
    valid_mask: np.ndarray,
    save_path: str,
    num_frames_to_show: int = 4,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    t_count = len(frames)
    if t_count == 0:
        return
    sample_ids = np.linspace(0, t_count - 1, num=min(num_frames_to_show, t_count), dtype=int)
    fig, axes = plt.subplots(1, len(sample_ids), figsize=(5 * len(sample_ids), 5))
    if len(sample_ids) == 1:
        axes = [axes]
    rng = np.random.default_rng(0)
    colors = rng.integers(0, 255, size=(tracks.shape[1], 3))

    for ax, t in zip(axes, sample_ids):
        img = frames[t].copy()
        ax.imshow(img)
        for n in range(tracks.shape[1]):
            valid_hist = valid_mask[: t + 1, n]
            if np.sum(valid_hist) < 2:
                continue
            path = tracks[: t + 1, n]
            path = path[valid_hist]
            col = colors[n] / 255.0
            ax.plot(path[:, 0], path[:, 1], "-", linewidth=1.2, color=col, alpha=0.8)
            ax.scatter(path[-1, 0], path[-1, 1], s=10, color=col)
        ax.set_title(f"Frame {t}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_valid_track_counts(valid_mask: np.ndarray, save_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    counts = np.sum(valid_mask, axis=1) if valid_mask.size else np.array([])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(len(counts)), counts, marker="o")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Valid tracks")
    ax.set_title("Valid tracked points over time")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tracked_feature_metrics(
    summary: Dict[str, object],
    save_path: str,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    rows = summary["per_frame"]
    x = np.array([int(r["frame_idx"]) for r in rows], dtype=int)
    mean_cos = np.array([r["mean_cosine_dissimilarity"] for r in rows], dtype=float)
    median_cos = np.array([r["median_cosine_dissimilarity"] for r in rows], dtype=float)
    mean_l2 = np.array([r["mean_l2_delta"] for r in rows], dtype=float)
    median_l2 = np.array([r["median_l2_delta"] for r in rows], dtype=float)
    valid = np.array([r["valid_tracks"] for r in rows], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(x, mean_cos, label="mean")
    axes[0].plot(x, median_cos, label="median")
    axes[0].set_ylabel("1 - cosine similarity")
    axes[0].set_title("Tracked feature cosine dissimilarity")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(x, mean_l2, label="mean")
    axes[1].plot(x, median_l2, label="median")
    axes[1].set_ylabel("L2 delta")
    axes[1].set_xlabel("Frame index")
    axes[1].set_title("Tracked feature L2 delta")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left")

    ax2 = axes[1].twinx()
    ax2.plot(x, valid, color="tab:gray", linestyle="--", label="valid tracks")
    ax2.set_ylabel("Valid tracks")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def backproject_points_from_depth(
    depth_map: np.ndarray,
    points_xy: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Optional helper: back-project 2D points to 3D using depth + intrinsics."""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    h, w = depth_map.shape
    x = np.clip(np.round(points_xy[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.round(points_xy[:, 1]).astype(int), 0, h - 1)
    z = depth_map[y, x]
    x3 = (points_xy[:, 0] - cx) * z / max(fx, 1e-8)
    y3 = (points_xy[:, 1] - cy) * z / max(fy, 1e-8)
    return np.stack([x3, y3, z], axis=1)

