"""
FeaturePoseTracker — tracks DINO features and neural CAPTRA poses across frames.

Neural CAPTRA outputs absolute poses per frame (not deltas), so this tracker:
  - Stores absolute translation (3,) and rotation_matrix (3,3) per frame
  - Computes frame-to-frame deltas as differences between consecutive absolute poses
  - Translation delta  = |t_curr - t_prev|
  - Rotation delta     = geodesic angle between R_curr and R_prev (degrees)
  - Absolute trajectory available via get_absolute_poses()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FrameRecord:
    """All tracked data for a single frame."""

    frame_idx: int
    object_feature: np.ndarray      # (D,) L2-normalised DINO object_mean
    cls_feature: np.ndarray         # (D,) L2-normalised DINO cls_token
    translation: np.ndarray         # (3,) absolute position from CAPTRA
    rotation_matrix: np.ndarray     # (3,3) absolute rotation from CAPTRA
    scale: float                    # absolute scale from CAPTRA
    valid: bool
    mask: Optional[np.ndarray] = None
    patch_feature_map: Optional[np.ndarray] = None
    image_size: Optional[Tuple[int, int]] = None
    image_path: Optional[str] = None
    timestamp: Optional[float] = None
    point_cloud: Optional[np.ndarray] = None  # (N, 3) camera-frame points from CAPTRA

    # legacy shim — computed on demand from rotation_matrix
    @property
    def rotation_euler(self) -> np.ndarray:
        return _rotation_matrix_to_euler_xyz(self.rotation_matrix)

    @property
    def pose_vector(self) -> np.ndarray:
        euler = self.rotation_euler
        return np.concatenate([self.translation, euler, [self.scale]])


# ── helpers ────────────────────────────────────────────────────────────────────

def _rotation_matrix_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy >= 1e-6:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0
    return np.array([x, y, z], dtype=np.float64)


def _geodesic_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic rotation angle in degrees between two rotation matrices."""
    R_rel = R1 @ R2.T
    trace = float(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(trace)))


# ── main class ─────────────────────────────────────────────────────────────────

class FeaturePoseTracker:

    def __init__(self, feature_key: str = "object_mean") -> None:
        self.feature_key = feature_key
        self.frames: List[FrameRecord] = []

    # ── adding frames ──────────────────────────────────────────────────────────

    def add_frame(
        self,
        frame_idx: int,
        features: Dict[str, Any],
        captra_output: Dict[str, Any],
        mask: Optional[np.ndarray] = None,
        image_path: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        if "object_mean" not in features:
            import warnings
            warnings.warn(
                f"[FeaturePoseTracker] frame {frame_idx}: 'object_mean' missing — "
                "delta_feature_mag will be 0. Check DINO feature extraction.",
                stacklevel=2,
            )

        obj_feat = np.asarray(features.get("object_mean", np.zeros(1)), dtype=np.float64)
        cls_feat = np.asarray(features.get("cls_token",   np.zeros(1)), dtype=np.float64)

        t = np.asarray(captra_output.get("translation",     np.zeros(3)),  dtype=np.float64)
        R = np.asarray(captra_output.get("rotation_matrix", np.eye(3)),    dtype=np.float64)
        s = float(captra_output.get("scale", 1.0))
        valid = bool(captra_output.get("valid", False))
        pcl_raw = captra_output.get("point_cloud", None)
        pcl = np.asarray(pcl_raw, dtype=np.float32) if pcl_raw is not None else None

        self.frames.append(FrameRecord(
            frame_idx       = frame_idx,
            object_feature  = obj_feat,
            cls_feature     = cls_feat,
            translation     = t,
            rotation_matrix = R,
            scale           = s,
            valid           = valid,
            mask            = mask,
            patch_feature_map = features.get("patch_feature_map"),
            image_size      = features.get("image_size"),
            image_path      = image_path,
            timestamp       = timestamp,
            point_cloud     = pcl,
        ))

    # ── delta computation ──────────────────────────────────────────────────────

    def compute_deltas(self) -> Dict[str, np.ndarray]:
        """
        Compute frame-to-frame changes from absolute CAPTRA poses.

        Returns
        -------
        delta_feature_mag     : L2 distance between consecutive feature vectors
        delta_translation_mag : |t_curr - t_prev|  (depth units)
        delta_rotation_deg    : geodesic angle between consecutive R matrices (degrees)
        delta_scale           : |s_curr - s_prev|
        cosine_similarity     : feature cosine similarity between consecutive frames
        frame_indices         : frame_idx of each delta (curr frame)
        """
        empty = {k: np.array([]) for k in [
            "delta_feature_mag", "delta_translation_mag",
            "delta_rotation_deg", "delta_scale",
            "cosine_similarity", "frame_indices",
        ]}
        if len(self.frames) < 2:
            return empty

        n = len(self.frames) - 1
        delta_feat  = np.zeros(n)
        delta_trans = np.zeros(n)
        delta_rot   = np.zeros(n)
        delta_scale = np.zeros(n)
        cosine_sim  = np.zeros(n)
        indices     = np.zeros(n, dtype=int)

        feat_attr = "object_feature" if self.feature_key == "object_mean" else "cls_feature"

        for i in range(n):
            prev, curr = self.frames[i], self.frames[i + 1]

            f_prev = getattr(prev, feat_attr)
            f_curr = getattr(curr, feat_attr)
            delta_feat[i] = np.linalg.norm(f_curr - f_prev)

            n_p, n_c = np.linalg.norm(f_prev), np.linalg.norm(f_curr)
            cosine_sim[i] = (
                float(np.dot(f_prev, f_curr) / (n_p * n_c))
                if n_p > 1e-8 and n_c > 1e-8 else np.nan
            )

            delta_trans[i] = float(np.linalg.norm(curr.translation - prev.translation))
            delta_rot[i]   = _geodesic_angle_deg(curr.rotation_matrix, prev.rotation_matrix)
            delta_scale[i] = abs(curr.scale - prev.scale)
            indices[i]     = curr.frame_idx

        return {
            "delta_feature_mag":     delta_feat,
            "delta_translation_mag": delta_trans,
            "delta_rotation_deg":    delta_rot,
            "delta_scale":           delta_scale,
            "cosine_similarity":     cosine_sim,
            "frame_indices":         indices,
        }

    # ── ICP-based rotation ─────────────────────────────────────────────────────

    @staticmethod
    def _icp_rotation_deg(
        src: np.ndarray,
        tgt: np.ndarray,
        n_pts: int = 256,
        max_iter: int = 30,
    ) -> float:
        """
        Estimate rotation angle (degrees) between two point clouds using ICP.

        Both clouds are centered before alignment so translation is removed.
        Returns NaN if ICP cannot converge (too few points or poor overlap).
        """
        try:
            from scipy.spatial import KDTree
        except ImportError:
            return np.nan

        src_c = (src - src.mean(0)).astype(np.float64)
        tgt_c = (tgt - tgt.mean(0)).astype(np.float64)

        rng = np.random.default_rng(42)
        if len(src_c) > n_pts:
            src_c = src_c[rng.choice(len(src_c), n_pts, replace=False)]
        if len(tgt_c) > n_pts:
            tgt_c = tgt_c[rng.choice(len(tgt_c), n_pts, replace=False)]

        if len(src_c) < 10 or len(tgt_c) < 10:
            return np.nan

        R_total = np.eye(3, dtype=np.float64)
        src_rot = src_c.copy()

        for _ in range(max_iter):
            tree = KDTree(tgt_c)
            dists, idx = tree.query(src_rot)

            threshold = np.percentile(dists, 80)
            mask = dists <= threshold
            if mask.sum() < 10:
                break

            A = src_rot[mask]
            B = tgt_c[idx[mask]]
            H = A.T @ B
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1] *= -1
                R = Vt.T @ U.T

            src_rot = (R @ src_rot.T).T
            R_total = R @ R_total

            angle_step = float(np.degrees(np.arccos(
                np.clip((np.trace(R) - 1) / 2, -1, 1)
            )))
            if angle_step < 0.05:
                break

        cos_total = float(np.clip((np.trace(R_total) - 1) / 2, -1, 1))
        return float(np.degrees(np.arccos(cos_total)))

    def fit_coupling_alpha(self) -> float:
        """
        Fit the linear coupling coefficient α (degrees / depth-unit of translation).

        Run this on a translation-only baseline to characterise how much spurious
        ICP rotation is induced by perspective distortion as the object translates.
        Use the returned α as --coupling-alpha on subsequent rotation runs.

        α is estimated as the least-squares slope through the origin between
        cumulative translation distance and accumulated ICP rotation angle.
        """
        icp    = self.compute_icp_rotation()
        deltas = self.compute_deltas()

        if not icp["available"]:
            print("[fit_coupling_alpha] No ICP data — run with point clouds stored.")
            return 0.0

        dt       = deltas["delta_translation_mag"]
        cum_trans = np.concatenate([[0.0], np.cumsum(dt)])   # (N,)
        cum_rot   = icp["accumulated_deg"]                    # (N,)

        mask = cum_trans > 1e-6
        if mask.sum() < 2:
            print("[fit_coupling_alpha] Not enough translation signal to fit α.")
            return 0.0

        alpha = float(
            np.dot(cum_trans[mask], cum_rot[mask]) /
            np.dot(cum_trans[mask], cum_trans[mask])
        )
        print(f"[fit_coupling_alpha] α = {alpha:.4f} °/depth-unit  "
              f"(total trans={cum_trans[-1]:.3f}, total rot={cum_rot[-1]:.2f}°)")
        return alpha

    def compute_icp_rotation(self, coupling_alpha: float = 0.0) -> Dict[str, np.ndarray]:
        """
        Compute ICP-based rotation for each consecutive frame pair and accumulate.

        Parameters
        ----------
        coupling_alpha : degrees of spurious rotation per depth-unit of cumulative
            translation (fitted via fit_coupling_alpha() on a translation baseline).
            When non-zero, the accumulated rotation is corrected by subtracting
            alpha × cumulative_translation from each frame's accumulated value.

        Returns
        -------
        delta_icp_deg      : (N-1,) frame-to-frame rotation angle via ICP
        accumulated_deg    : (N,)   cumulative rotation (corrected if alpha != 0)
        accumulated_raw_deg: (N,)   uncorrected cumulative rotation
        frame_indices      : (N,)   frame_idx of each frame
        available          : bool   False if no point clouds are stored
        corrected          : bool   True when coupling_alpha != 0 was applied
        """
        n = len(self.frames)
        empty = {
            "delta_icp_deg":       np.array([]),
            "accumulated_deg":     np.array([]),
            "accumulated_raw_deg": np.array([]),
            "frame_indices":       np.array([], dtype=int),
            "available":           False,
            "corrected":           False,
        }
        if n < 2:
            return empty

        has_pcl = any(f.point_cloud is not None for f in self.frames)
        if not has_pcl:
            return empty

        delta_icp = np.full(n - 1, np.nan)
        for i in range(n - 1):
            p, c = self.frames[i].point_cloud, self.frames[i + 1].point_cloud
            if p is not None and c is not None:
                delta_icp[i] = self._icp_rotation_deg(p, c)

        accumulated = np.zeros(n)
        for i in range(1, n):
            d = delta_icp[i - 1]
            accumulated[i] = accumulated[i - 1] + (d if not np.isnan(d) else 0.0)

        raw = accumulated.copy()

        if coupling_alpha != 0.0:
            deltas    = self.compute_deltas()
            dt        = deltas["delta_translation_mag"]
            cum_trans = np.concatenate([[0.0], np.cumsum(dt)])
            accumulated = np.maximum(0.0, accumulated - coupling_alpha * cum_trans)

        return {
            "delta_icp_deg":       delta_icp,
            "accumulated_deg":     accumulated,
            "accumulated_raw_deg": raw,
            "frame_indices":       np.array([f.frame_idx for f in self.frames], dtype=int),
            "available":           True,
            "corrected":           coupling_alpha != 0.0,
        }

    # ── absolute pose trajectory ───────────────────────────────────────────────

    def get_absolute_poses(self) -> Dict[str, np.ndarray]:
        """
        Return absolute pose trajectories across all frames.

        Returns
        -------
        translations      : (N, 3) absolute XYZ per frame
        rotation_angles   : (N,)   geodesic angle from frame-0 rotation (degrees)
        scales            : (N,)   absolute scale per frame
        frame_indices     : (N,)   frame_idx values
        valid             : (N,)   bool validity per frame
        """
        n = len(self.frames)
        if n == 0:
            return {k: np.array([]) for k in
                    ["translations", "rotation_angles", "scales", "frame_indices", "valid"]}

        translations    = np.stack([f.translation for f in self.frames])      # (N, 3)
        scales          = np.array([f.scale for f in self.frames])
        frame_indices   = np.array([f.frame_idx for f in self.frames], dtype=int)
        valid           = np.array([f.valid for f in self.frames], dtype=bool)

        R0 = self.frames[0].rotation_matrix
        rotation_angles = np.array([
            _geodesic_angle_deg(f.rotation_matrix, R0) for f in self.frames
        ])

        return {
            "translations":    translations,
            "rotation_angles": rotation_angles,
            "scales":          scales,
            "frame_indices":   frame_indices,
            "valid":           valid,
        }

    # ── CAPTRA health verification ─────────────────────────────────────────────

    def verify_captra(self) -> Dict[str, Any]:
        """
        Check per-frame rotation matrix health and translation bounds.

        Prints a diagnostic table and returns aggregate stats.
        """
        if not self.frames:
            print("[verify_captra] No frames recorded.")
            return {}

        det_errors  = []
        ortho_errors = []
        t_norms     = []
        invalid_frames = []

        for f in self.frames:
            if not f.valid:
                invalid_frames.append(f.frame_idx)
                continue
            R = f.rotation_matrix
            det_errors.append(abs(np.linalg.det(R) - 1.0))
            ortho_errors.append(float(np.linalg.norm(R.T @ R - np.eye(3), ord="fro")))
            t_norms.append(float(np.linalg.norm(f.translation)))

        valid_count   = len(self.frames) - len(invalid_frames)
        total         = len(self.frames)

        print("\n=== CAPTRA Verification ===")
        print(f"  Total frames     : {total}")
        print(f"  Valid frames     : {valid_count} / {total}  "
              f"({100*valid_count/max(total,1):.1f}%)")
        if invalid_frames:
            print(f"  Invalid frames   : {invalid_frames}")

        if det_errors:
            print(f"\n  Rotation matrix health (det should be 1.0, ortho error should be ~0):")
            print(f"    det error  — mean: {np.mean(det_errors):.2e}  "
                  f"max: {np.max(det_errors):.2e}")
            print(f"    ortho err  — mean: {np.mean(ortho_errors):.2e}  "
                  f"max: {np.max(ortho_errors):.2e}")

            if np.max(det_errors) < 1e-4 and np.max(ortho_errors) < 1e-4:
                print("    ✓  Rotation matrices are valid SO(3) matrices.")
            else:
                print("    ⚠  Some rotation matrices deviate from SO(3) — check CAPTRA output.")

            print(f"\n  Translation norms (depth units):")
            print(f"    mean: {np.mean(t_norms):.4f}  "
                  f"min: {np.min(t_norms):.4f}  "
                  f"max: {np.max(t_norms):.4f}")

        return {
            "valid_count":          valid_count,
            "total_frames":         total,
            "mean_det_error":       float(np.mean(det_errors))   if det_errors else np.nan,
            "max_det_error":        float(np.max(det_errors))    if det_errors else np.nan,
            "mean_ortho_error":     float(np.mean(ortho_errors)) if ortho_errors else np.nan,
            "max_ortho_error":      float(np.max(ortho_errors))  if ortho_errors else np.nan,
            "mean_translation_norm":float(np.mean(t_norms))      if t_norms else np.nan,
        }

    # ── plotting ───────────────────────────────────────────────────────────────

    def plot(
        self,
        figsize: Tuple[int, int] = (22, 14),
        save_path: Optional[str] = None,
        title: Optional[str] = None,
        smooth_window: int = 5,
        coupling_alpha: float = 0.0,
        ylim_feat:  Optional[Tuple[float, float]] = None,
        ylim_trans: Optional[Tuple[float, float]] = None,
        ylim_rot:   Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        3×3 figure — each panel has room to breathe.

        Row 1  (absolute translation, one axis per panel):
          tx  |  ty  |  tz

        Row 2  (absolute rotation & frame-to-frame deltas):
          Rotation ° from frame 0  |  Δ Feature (L2)  |  Δ Translation

        Row 3  (rotation delta & correlation scatter plots):
          Δ Rotation (°)  |  Scatter: Δtranslation vs Δfeature  |  Scatter: Δrotation vs Δfeature

        A summary text block appears below the figure.
        Rolling-mean trend lines (window = smooth_window) are overlaid on delta panels.
        Pearson r is annotated directly on each scatter plot.
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        deltas    = self.compute_deltas()
        abs_poses = self.get_absolute_poses()
        icp       = self.compute_icp_rotation(coupling_alpha=coupling_alpha)

        if len(deltas["delta_feature_mag"]) == 0:
            print("[FeaturePoseTracker] Need at least 2 frames to plot.")
            return

        df   = deltas["delta_feature_mag"]
        dt   = deltas["delta_translation_mag"]
        dr   = deltas["delta_rotation_deg"]
        cos  = deltas["cosine_similarity"]
        didx = deltas["frame_indices"].astype(float)

        trans = abs_poses["translations"]     # (N, 3)
        rang  = abs_poses["rotation_angles"]  # (N,) CAPTRA absolute
        aidx  = abs_poses["frame_indices"].astype(float)
        valid = abs_poses["valid"]

        # ICP rotation — use when available, fall back to CAPTRA
        icp_available   = icp["available"]
        icp_accum       = icp["accumulated_deg"]   if icp_available else rang
        icp_delta       = icp["delta_icp_deg"]     if icp_available else dr
        icp_aidx        = icp["frame_indices"].astype(float) if icp_available else aidx
        if icp_available:
            if icp.get("corrected"):
                rot_label = f"ICP Accumulated [α-corrected α={coupling_alpha:.3f}]"
            else:
                rot_label = "ICP Accumulated"
        else:
            rot_label = "CAPTRA (unreliable)"
        delta_rot_label = "Δ Rotation ICP (°)" if icp_available else "Δ Rotation CAPTRA (°)"

        # ── style helpers ────────────────────────────────────────────────────
        LABEL_FS, TICK_FS, TITLE_FS = 11, 10, 12

        def _style(a, xlabel="Frame", ylabel="", title_str=""):
            a.set_xlabel(xlabel, fontsize=LABEL_FS)
            a.set_ylabel(ylabel, fontsize=LABEL_FS)
            a.set_title(title_str, fontsize=TITLE_FS, fontweight="bold", pad=6)
            a.tick_params(labelsize=TICK_FS)
            a.grid(True, linestyle="--", alpha=0.3)
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)

        def _mark_invalid(a, aidx, valid):
            for xi in aidx[~valid]:
                a.axvline(xi, color="lightcoral", lw=1.2, alpha=0.5, zorder=0)

        def _smooth(vals):
            w = min(smooth_window, len(vals))
            if w < 2:
                return None, None
            kernel   = np.ones(w) / w
            smoothed = np.convolve(vals, kernel, mode="same")
            # mask edge artifacts (first and last w//2 values)
            half = w // 2
            mask = np.zeros(len(smoothed), dtype=bool)
            mask[half: len(smoothed) - half] = True
            return smoothed, mask

        def _pearson_r(x, y):
            """NaN-safe Pearson r — returns (r, mask) using only finite pairs."""
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 3 or np.std(x[mask]) < 1e-10 or np.std(y[mask]) < 1e-10:
                return np.nan, mask
            return float(np.corrcoef(x[mask], y[mask])[0, 1]), mask

        def _scatter_with_r(a, x, y, xlabel, ylabel, title_str, cvals):
            r, mask = _pearson_r(x, y)
            sc = a.scatter(x[mask], y[mask], c=cvals[mask], cmap="viridis",
                           edgecolors="k", linewidths=0.4, alpha=0.75, s=50)
            if not np.isnan(r):
                xm, ym = x[mask], y[mask]
                m, b = np.polyfit(xm, ym, 1)
                xs = np.linspace(xm.min(), xm.max(), 100)
                a.plot(xs, m * xs + b, "k--", lw=1.2, alpha=0.6)
                a.text(0.97, 0.96, f"r = {r:.3f}",
                       transform=a.transAxes, ha="right", va="top",
                       fontsize=11, fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))
            _style(a, xlabel=xlabel, ylabel=ylabel, title_str=title_str)
            return sc

        def _line_with_trend(a, x, y, color, ylabel, title_str, ylim=None):
            a.plot(x, y, "o-", color=color, markersize=4, linewidth=1.2,
                   alpha=0.6, label="per frame")
            smoothed, smask = _smooth(y)
            if smoothed is not None:
                a.plot(x[smask], smoothed[smask], "-", color=color,
                       linewidth=2.5, alpha=0.9, label=f"trend (w={smooth_window})")
            a.axhline(0, color="k", lw=0.6, alpha=0.25)
            if ylim:
                a.set_ylim(ylim)
            a.legend(fontsize=9, loc="upper right")
            _style(a, xlabel="Frame", ylabel=ylabel, title_str=title_str)

        # ── figure ───────────────────────────────────────────────────────────
        fig = plt.figure(figsize=figsize)
        fig.suptitle(
            title or "CAPTRA Pose Trajectory  ·  Feature Correlation",
            fontsize=15, fontweight="bold", y=0.99,
        )
        gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)
        axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(3)]

        # ── Row 0: absolute translation (one axis per panel) ─────────────────
        t0 = trans[0]
        for col, (axis_name, color, data) in enumerate([
            ("X", "#d62728", trans[:, 0] - t0[0]),
            ("Y", "#2ca02c", trans[:, 1] - t0[1]),
            ("Z", "#1f77b4", trans[:, 2] - t0[2]),
        ]):
            a = axes[0][col]
            a.plot(aidx, data, "o-", color=color, markersize=4, linewidth=1.5, alpha=0.7)
            _mark_invalid(a, aidx, valid)
            a.axhline(0, color="k", lw=0.7, alpha=0.3, linestyle=":")
            _style(a, ylabel="Displacement from start (depth units)",
                   title_str=f"Translation from Start  —  {axis_name} axis")

        # ── Row 1: rotation + deltas ──────────────────────────────────────────
        a = axes[1][0]
        a.plot(icp_aidx, icp_accum, "o-", color="#9467bd", markersize=4, linewidth=1.5, alpha=0.7)
        _mark_invalid(a, aidx, valid)
        a.axhline(0, color="k", lw=0.6, alpha=0.25)
        _style(a, ylabel="Degrees", title_str=f"Rotation from Frame 0  [{rot_label}]")

        _line_with_trend(axes[1][1], didx, df, "#1f77b4",
                         ylabel="Δ Feature (L2)", title_str="Feature Change per Frame",
                         ylim=ylim_feat)

        _line_with_trend(axes[1][2], didx, dt, "#d62728",
                         ylabel="Δ Translation (depth units)", title_str="Translation Change per Frame",
                         ylim=ylim_trans)

        # ── Row 2: rotation delta + scatter plots ─────────────────────────────
        # ICP delta uses aidx (same length as icp_delta = N-1)
        icp_didx = icp_aidx[1:] if icp_available else didx
        _line_with_trend(axes[2][0], icp_didx, icp_delta, "#2ca02c",
                         ylabel=delta_rot_label, title_str="Rotation Change per Frame",
                         ylim=ylim_rot)

        sc1 = _scatter_with_r(axes[2][1], dt, df,
                               xlabel="Δ Translation (depth units)",
                               ylabel="Δ Feature (L2)",
                               title_str="Translation vs Feature",
                               cvals=didx)
        fig.colorbar(sc1, ax=axes[2][1], label="Frame index", shrink=0.8)

        # Scatter uses ICP delta — align lengths with df
        dr_plot = icp_delta if icp_available else dr
        sc2 = _scatter_with_r(axes[2][2], dr_plot, df,
                               xlabel=delta_rot_label,
                               ylabel="Δ Feature (L2)",
                               title_str="Rotation vs Feature",
                               cvals=didx)
        fig.colorbar(sc2, ax=axes[2][2], label="Frame index", shrink=0.8)

        # ── Summary text below figure ─────────────────────────────────────────
        # Use ICP delta for rotation stats when available (matches scatter plot)
        dr_stat = icp_delta if icp_available else dr
        # dr_stat is length N-1; df is also N-1 — safe to correlate directly
        n_valid = int(valid.sum())
        vc = cos[~np.isnan(cos)]
        rot_mu = dr_stat[~np.isnan(dr_stat)].mean() if len(dr_stat) > 0 else 0.0
        rot_src = "ICP" if icp_available else "CAPTRA"
        parts = [
            f"Frames: {len(self.frames)}   Valid: {n_valid}   "
            f"Feature dim: {self.frames[0].object_feature.shape[0]}",
            f"Δ Feature μ={df.mean():.4f}   "
            f"Δ Trans μ={dt.mean():.4f}   "
            f"Δ Rot μ={rot_mu:.2f}° ({rot_src})",
        ]
        if len(vc):
            parts.append(f"Mean cosine similarity: {vc.mean():.4f}")
        if len(df) > 2:
            r_t,     _ = _pearson_r(dt, df)
            r_r,     _ = _pearson_r(dr_stat, df)
            max_t = float(np.nanmax(dt))      if np.nanmax(dt)      > 1e-10 else 1.0
            max_r = float(np.nanmax(dr_stat)) if np.nanmax(dr_stat) > 1e-10 else 1.0
            total_motion = dt / max_t + dr_stat / max_r
            r_total, _  = _pearson_r(total_motion, df)
            def _fmt(v): return f"{v:.3f}" if not np.isnan(v) else "n/a"
            parts.append(
                f"r(feat, trans)={_fmt(r_t)}   r(feat, rot {rot_src})={_fmt(r_r)}   "
                f"r(feat, total motion)={_fmt(r_total)}"
            )
        parts.append("Depth units: 0.3≈0 m  |  1.0≈0.7 m  |  2.0≈1.5 m  |  3.3≈3 m")
        fig.text(0.5, 0.005, "   ·   ".join(parts),
                 ha="center", va="bottom", fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="silver"))

        if save_path:
            import os as _os
            abs_path = _os.path.abspath(save_path)
            _os.makedirs(_os.path.dirname(abs_path), exist_ok=True)
            plt.savefig(abs_path, dpi=150, bbox_inches="tight")
            print(f"[FeaturePoseTracker] Plot saved to {abs_path}")
        else:
            plt.tight_layout(rect=[0, 0.04, 1, 0.98])
            plt.show()

    # ── spatial feature PCA (unchanged) ───────────────────────────────────────

    @staticmethod
    def _foreground_patches(
        pixels: np.ndarray,
        mask: Optional[np.ndarray],
        H: int,
        W: int,
    ) -> np.ndarray:
        if mask is None:
            return pixels
        import cv2 as _cv2
        msk = np.asarray(mask, dtype=np.float32)
        if msk.ndim != 2:
            return pixels
        if msk.shape != (H, W):
            msk = _cv2.resize(msk, (W, H), interpolation=_cv2.INTER_NEAREST)
        fg = msk.reshape(-1) > 0.5
        return pixels[fg] if fg.any() else pixels

    # ── summary ────────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        deltas = self.compute_deltas()
        icp    = self.compute_icp_rotation()
        n = len(self.frames)
        stats: Dict[str, Any] = {
            "num_frames":   n,
            "num_deltas":   max(n - 1, 0),
            "valid_frames": int(sum(f.valid for f in self.frames)),
            "feature_dim":  self.frames[0].object_feature.shape[0] if n > 0 else 0,
        }

        def _nanr(a, b):
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < 3 or np.std(a[mask]) < 1e-10 or np.std(b[mask]) < 1e-10:
                return np.nan
            return float(np.corrcoef(a[mask], b[mask])[0, 1])

        for key, label in [
            ("delta_feature_mag",     "delta_feature_mag"),
            ("delta_translation_mag", "delta_translation_mag"),
            ("delta_rotation_deg",    "delta_rotation_deg"),
        ]:
            vals = deltas.get(key, np.array([]))
            v = vals[np.isfinite(vals)] if vals.size else vals
            if len(v) > 0:
                stats[f"{label}_mean"] = float(np.mean(v))
                stats[f"{label}_std"]  = float(np.std(v))
                stats[f"{label}_max"]  = float(np.max(v))

        vc = deltas.get("cosine_similarity", np.array([]))
        vc = vc[np.isfinite(vc)] if vc.size else vc
        if len(vc) > 0:
            stats["mean_cosine_similarity"] = float(np.mean(vc))

        df = deltas.get("delta_feature_mag",     np.array([]))
        dt = deltas.get("delta_translation_mag", np.array([]))

        # Use ICP rotation when available — matches plot scatter and toolbar
        if icp["available"]:
            dr      = icp["delta_icp_deg"]
            rot_src = "icp"
        else:
            dr      = deltas.get("delta_rotation_deg", np.array([]))
            rot_src = "captra"

        if len(df) > 2:
            stats["correlation_feature_translation"]            = _nanr(df, dt)
            stats[f"correlation_feature_rotation_{rot_src}"]   = _nanr(df, dr)
            max_t = float(np.nanmax(dt))  if np.nanmax(dt)  > 1e-10 else 1.0
            max_r = float(np.nanmax(dr))  if np.nanmax(dr)  > 1e-10 else 1.0
            total_motion = dt / max_t + dr / max_r
            stats["correlation_feature_total_motion"]           = _nanr(df, total_motion)

        print("=== FeaturePoseTracker Summary ===")
        for k, v in stats.items():
            print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
        return stats

    # ── CSV export ─────────────────────────────────────────────────────────────

    def to_csv(self, path: str) -> None:
        import csv as _csv
        deltas   = self.compute_deltas()
        abs_pose = self.get_absolute_poses()

        n = len(self.frames)

        def _f(v) -> str:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return ""
            return f"{v:.6f}"

        import os as _os
        abs_path = _os.path.abspath(path)
        _os.makedirs(_os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow([
                "frame_idx", "valid", "image_path", "timestamp",
                "tx", "ty", "tz",
                "rotation_angle_from_ref_deg",
                "scale",
                "delta_feature_mag", "delta_translation_mag",
                "delta_rotation_deg", "cosine_similarity",
            ])
            for i, fr in enumerate(self.frames):
                ang = float(abs_pose["rotation_angles"][i]) if len(abs_pose["rotation_angles"]) > i else np.nan
                if i > 0 and i - 1 < len(deltas["delta_feature_mag"]):
                    j = i - 1
                    row_d = [
                        _f(deltas["delta_feature_mag"][j]),
                        _f(deltas["delta_translation_mag"][j]),
                        _f(deltas["delta_rotation_deg"][j]),
                        _f(deltas["cosine_similarity"][j]),
                    ]
                else:
                    row_d = [""] * 4

                w.writerow([
                    fr.frame_idx, fr.valid, fr.image_path or "",
                    _f(fr.timestamp),
                    _f(fr.translation[0]), _f(fr.translation[1]), _f(fr.translation[2]),
                    _f(ang),
                    _f(fr.scale),
                    *row_d,
                ])
        print(f"[FeaturePoseTracker] CSV written to {abs_path}")
