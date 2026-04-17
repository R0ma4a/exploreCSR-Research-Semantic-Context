"""
FeaturePoseTracker — tracks DINO features and CAPTRA poses across frames.

Translation is sourced from the 3D CAPTRA centroid shift (metric units).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FrameRecord:
    """All tracked data for a single frame."""

    frame_idx: int
    object_feature: np.ndarray
    cls_feature: np.ndarray
    pose_vector: np.ndarray
    translation: np.ndarray
    rotation_euler: np.ndarray
    scale: float
    valid: bool
    mask: Optional[np.ndarray] = None
    patch_feature_map: Optional[np.ndarray] = None
    image_size: Optional[Tuple[int, int]] = None
    image_path: Optional[str] = None
    timestamp: Optional[float] = None


class FeaturePoseTracker:

    def __init__(self, feature_key: str = "object_mean") -> None:
        self.feature_key = feature_key
        self.frames: List[FrameRecord] = []

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
                f"[FeaturePoseTracker] frame {frame_idx}: 'object_mean' key missing from features dict — "
                "delta_feature_mag will be 0. Check your DINO feature extraction.",
                stacklevel=2,
            )
        obj_feat = np.asarray(features.get("object_mean", np.zeros(1)), dtype=np.float64)
        cls_feat = np.asarray(features.get("cls_token",   np.zeros(1)), dtype=np.float64)

        pose_vector    = np.asarray(captra_output.get("pose_vector",    np.full(7, np.nan)), dtype=np.float64)
        translation    = np.asarray(captra_output.get("translation",    np.full(3, np.nan)), dtype=np.float64)
        rotation_euler = np.asarray(captra_output.get("rotation_euler", np.full(3, np.nan)), dtype=np.float64)
        scale = float(captra_output.get("scale", np.nan))
        valid = bool(captra_output.get("valid", False))

        self.frames.append(FrameRecord(
            frame_idx=frame_idx,
            object_feature=obj_feat,
            cls_feature=cls_feat,
            pose_vector=pose_vector,
            translation=translation,
            rotation_euler=rotation_euler,
            scale=scale,
            valid=valid,
            mask=mask,  # ✅ ADD THIS LINE
            patch_feature_map=features.get("patch_feature_map"),
            image_size=features.get("image_size"),
            image_path=image_path,
            timestamp=timestamp,
        ))

    # ------------------------------------------------------------------

    def compute_deltas(self) -> Dict[str, np.ndarray]:
        empty = {k: np.array([]) for k in [
            "delta_feature_mag", "delta_translation_mag",
            "delta_rotation_mag", "delta_pose_mag",
            "delta_scale", "frame_indices", "cosine_similarity",
        ]}
        if len(self.frames) < 2:
            return empty

        n = len(self.frames) - 1
        delta_feat  = np.zeros(n)
        delta_trans = np.zeros(n)
        delta_rot   = np.zeros(n)
        delta_pose  = np.zeros(n)
        delta_scale = np.zeros(n)
        cosine_sim  = np.zeros(n)
        indices     = np.zeros(n, dtype=int)

        for i in range(n):
            prev, curr = self.frames[i], self.frames[i + 1]

            # Feature delta
            feat_key = "object_feature" if self.feature_key == "object_mean" else "cls_feature"
            f_prev, f_curr = getattr(prev, feat_key), getattr(curr, feat_key)
            delta_feat[i] = np.linalg.norm(f_curr - f_prev)

            # Cosine similarity
            n_prev, n_curr = np.linalg.norm(f_prev), np.linalg.norm(f_curr)
            cosine_sim[i] = (
                np.dot(f_prev, f_curr) / (n_prev * n_curr)
                if n_prev > 1e-8 and n_curr > 1e-8 else np.nan
            )

            # Translation (3D centroid shift from CAPTRA)
            delta_trans[i] = np.linalg.norm(curr.translation)

            # Rotation
            delta_rot[i] = np.linalg.norm(curr.rotation_euler)

            # Combined pose
            delta_pose[i] = np.linalg.norm(np.concatenate([curr.translation, curr.rotation_euler]))

            # Scale deviation from 1
            delta_scale[i] = abs(curr.scale - 1.0)

            indices[i] = curr.frame_idx

        return {
            "delta_feature_mag":     delta_feat,
            "delta_translation_mag": delta_trans,
            "delta_rotation_mag":    delta_rot,
            "delta_pose_mag":        delta_pose,
            "delta_scale":           delta_scale,
            "frame_indices":         indices,
            "cosine_similarity":     cosine_sim,
        }

    # ------------------------------------------------------------------

    def plot(
        self,
        figsize: Tuple[int, int] = (20, 10),
        save_path: Optional[str] = None,
        xlim_trans: Optional[Tuple[float, float]] = None,
        xlim_rot:   Optional[Tuple[float, float]] = None,
        xlim_pose:  Optional[Tuple[float, float]] = None,
        ylim_feat:  Optional[Tuple[float, float]] = None,
        ylim_trans: Optional[Tuple[float, float]] = None,
        ylim_rot:   Optional[Tuple[float, float]] = None,
        ylim_cos:   Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
    ) -> None:
        """
        2x4 figure:
          Row 1: scatter (translation / rotation / combined pose / summary)
          Row 2: time series (feature / translation / rotation / cosine dissimilarity)
        """
        import matplotlib.pyplot as plt

        deltas = self.compute_deltas()
        if len(deltas["delta_feature_mag"]) == 0:
            print("[FeaturePoseTracker] Need at least 2 frames to plot.")
            return

        df  = deltas["delta_feature_mag"]
        dt  = deltas["delta_translation_mag"]
        dr  = deltas["delta_rotation_mag"]
        dp  = deltas["delta_pose_mag"]
        idx = deltas["frame_indices"]
        cos = deltas["cosine_similarity"]

        fig, ax = plt.subplots(2, 4, figsize=figsize)
        fig.suptitle(title or "Motion Change (x) vs Feature Change (y)", fontsize=14, fontweight="bold")

        # --- Row 1: Scatter ---
        ax[0, 0].scatter(dt, df, c=idx, cmap="viridis", edgecolors="k", alpha=0.7)
        ax[0, 0].set_xlabel("Δ Translation")
        ax[0, 0].set_ylabel("Δ Feature Magnitude")
        ax[0, 0].set_title("Translation vs Feature")
        if xlim_trans: ax[0, 0].set_xlim(xlim_trans)
        if ylim_feat:  ax[0, 0].set_ylim(ylim_feat)

        ax[0, 1].scatter(dr, df, c=idx, cmap="viridis", edgecolors="k", alpha=0.7)
        ax[0, 1].set_xlabel("Δ Rotation (rad)")
        ax[0, 1].set_ylabel("Δ Feature Magnitude")
        ax[0, 1].set_title("Rotation vs Feature")
        if xlim_rot:  ax[0, 1].set_xlim(xlim_rot)
        if ylim_feat: ax[0, 1].set_ylim(ylim_feat)

        sc = ax[0, 2].scatter(dp, df, c=idx, cmap="viridis", edgecolors="k", alpha=0.7)
        ax[0, 2].set_xlabel("Δ Combined Pose")
        ax[0, 2].set_ylabel("Δ Feature Magnitude")
        ax[0, 2].set_title("Combined Pose vs Feature")
        fig.colorbar(sc, ax=ax[0, 2], label="Frame Index")
        if xlim_pose: ax[0, 2].set_xlim(xlim_pose)
        if ylim_feat: ax[0, 2].set_ylim(ylim_feat)

        # Summary
        ax[0, 3].axis("off")
        lines = [
            f"Frames:    {len(self.frames)}",
            f"Valid:     {sum(1 for f in self.frames if f.valid)}",
            f"Feat dim:  {self.frames[0].object_feature.shape[0]}",
            "",
            f"Δ Feature  mean: {df.mean():.4f}",
            f"Δ Trans    mean: {dt.mean():.4f}",
            f"Δ Rot      mean: {dr.mean():.4f}",
        ]
        vc = cos[~np.isnan(cos)]
        if len(vc) > 0:
            lines.append(f"Cos sim    mean: {vc.mean():.4f}")
        if len(df) > 2 and np.std(df) > 1e-10 and np.std(dp) > 1e-10:
            lines.append(f"Corr(feat,pose): {float(np.corrcoef(df, dp)[0, 1]):.4f}")
        ax[0, 3].text(0.1, 0.95, "\n".join(lines), transform=ax[0, 3].transAxes,
                      fontsize=11, verticalalignment="top", fontfamily="monospace")
        ax[0, 3].set_title("Summary")

        # --- Row 2: Time series ---
        ax[1, 0].plot(idx, df, "o-", color="tab:blue", markersize=3, linewidth=1)
        ax[1, 0].set_xlabel("Frame Index")
        ax[1, 0].set_ylabel("Δ Feature Magnitude")
        ax[1, 0].set_title("Feature Over Time")
        if ylim_feat: ax[1, 0].set_ylim(ylim_feat)

        ax[1, 1].plot(idx, dt, "s-", color="tab:red", markersize=3, linewidth=1)
        ax[1, 1].set_xlabel("Frame Index")
        ax[1, 1].set_ylabel("Δ Translation")
        ax[1, 1].set_title("Translation Over Time")
        if ylim_trans: ax[1, 1].set_ylim(ylim_trans)

        ax[1, 2].plot(idx, dr, "^-", color="tab:green", markersize=3, linewidth=1)
        ax[1, 2].set_xlabel("Frame Index")
        ax[1, 2].set_ylabel("Δ Rotation (rad)")
        ax[1, 2].set_title("Rotation Over Time")
        if ylim_rot: ax[1, 2].set_ylim(ylim_rot)

        ax[1, 3].plot(idx, 1.0 - cos, "D-", color="tab:purple", markersize=3, linewidth=1)
        ax[1, 3].set_xlabel("Frame Index")
        ax[1, 3].set_ylabel("1 − cos(sim)")
        ax[1, 3].set_title("Cosine Dissimilarity Over Time")
        if ylim_cos: ax[1, 3].set_ylim(ylim_cos)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[FeaturePoseTracker] Saved to {save_path}")
        else:
            plt.show()

    # ------------------------------------------------------------------

    @staticmethod
    def _foreground_patches(
        pixels: np.ndarray,          # [H*W, D]
        mask: Optional[np.ndarray],  # full-res binary mask, or None
        H: int,
        W: int,
    ) -> np.ndarray:
        """
        Return only the rows of ``pixels`` that correspond to foreground
        patch tokens.  If no mask is stored, all patches are returned.

        The pipeline mask is full-resolution (e.g. 480×640); it is resized
        to the patch grid (H, W) with nearest-neighbour interpolation so
        that a patch token is foreground if *any* of its source pixels were
        labelled foreground.
        """
        if mask is None:
            return pixels
        import cv2 as _cv2
        msk = np.asarray(mask, dtype=np.float32)
        if msk.ndim != 2:
            return pixels
        if msk.shape != (H, W):
            msk = _cv2.resize(msk, (W, H), interpolation=_cv2.INTER_NEAREST)
        fg = msk.reshape(-1) > 0.5          # [H*W] bool
        return pixels[fg] if fg.any() else pixels

    # ------------------------------------------------------------------

    def visualize_feature_pca(
        self,
        frame_indices: Optional[List[int]] = None,
        n_cols: int = 4,
        figsize_per_cell: Tuple[float, float] = (3.0, 3.0),
        save_path: Optional[str] = None,
        title: Optional[str] = None,
        mask_alpha: float = 0.3,
    ) -> None:
        """
        Project each frame's patch feature map from D → 3 via PCA and display
        it as a colour image.  This reveals *spatial* variation in the features,
        letting you judge whether mean-pooling discards important structure.

        The PCA is fitted jointly across all selected frames so that colours are
        comparable between frames.

        Parameters
        ----------
        frame_indices : list of ints, optional
            Which frames to visualise (by position in self.frames, not frame_idx).
            Defaults to all frames.
        n_cols : int
            Number of columns in the subplot grid.
        figsize_per_cell : (w, h)
            Size in inches for each subplot cell.
        save_path : str, optional
            If given, save the figure instead of showing it.
        title : str, optional
            Overall figure title.
        mask_alpha : float
            If a binary mask is stored, overlay it at this opacity (0 = off).
        """
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        # ── select frames that actually have a patch map ──────────────────
        candidates = (
            [self.frames[i] for i in frame_indices]
            if frame_indices is not None
            else self.frames
        )
        frames_with_map = [f for f in candidates if f.patch_feature_map is not None]
        if not frames_with_map:
            print("[FeaturePoseTracker] No patch_feature_map found in selected frames.")
            return

        # ── reshape each map: [D, H, W] → [H*W, D] ───────────────────────
        def _hw_pixels(f: FrameRecord) -> Tuple[np.ndarray, int, int]:
            m = np.asarray(f.patch_feature_map, dtype=np.float64)
            if m.ndim == 2:          # [D, N] — flat spatial tokens
                D, N = m.shape
                H = W = int(round(N ** 0.5))
                if H * W != N:
                    raise ValueError(
                        f"Frame {f.frame_idx}: cannot infer (H, W) from flat token count {N}. "
                        "Store patch_feature_map as [D, H, W]."
                    )
                m = m.reshape(D, H, W)
            D, H, W = m.shape
            return m.reshape(D, H * W).T, H, W   # → [H*W, D]

        parsed = [_hw_pixels(f) for f in frames_with_map]
        all_pixels = np.concatenate([p for p, _, _ in parsed], axis=0)   # [N_total, D]

        # ── fit PCA on the joint pixel set ────────────────────────────────
        pca = PCA(n_components=3, random_state=0)
        pca.fit(all_pixels)
        explained = pca.explained_variance_ratio_

        # ── layout ────────────────────────────────────────────────────────
        n = len(frames_with_map)
        n_cols = min(n_cols, n)
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
        )
        axes = np.array(axes).reshape(-1)   # always 1-D

        fig.suptitle(
            title or (
                f"PCA Feature Visualisation  —  "
                f"PC1-3 explain {100*explained.sum():.1f}% of variance  "
                f"({100*explained[0]:.1f} / {100*explained[1]:.1f} / {100*explained[2]:.1f}%)"
            ),
            fontsize=11, fontweight="bold",
        )

        for ax, frame, (pixels, H, W) in zip(axes, frames_with_map, parsed):
            rgb_flat = pca.transform(pixels)           # [H*W, 3]

            # Normalise each channel to [0, 1] independently for display
            lo, hi = rgb_flat.min(axis=0), rgb_flat.max(axis=0)
            denom = np.where(hi - lo > 1e-10, hi - lo, 1.0)
            rgb_flat = (rgb_flat - lo) / denom
            rgb_img = rgb_flat.reshape(H, W, 3).clip(0, 1)

            ax.imshow(rgb_img, interpolation="nearest")

            # Optionally overlay the stored binary mask.
            # The pipeline stores a full-resolution mask; resize it to the
            # patch grid (H, W) so the shape check always passes.
            if frame.mask is not None and mask_alpha > 0:
                msk = np.asarray(frame.mask, dtype=np.float32)
                if msk.ndim == 2:
                    if msk.shape != (H, W):
                        import cv2 as _cv2
                        msk = _cv2.resize(msk, (W, H), interpolation=_cv2.INTER_NEAREST)
                    overlay = np.zeros((H, W, 4), dtype=np.float32)
                    overlay[..., 3] = (1.0 - msk) * mask_alpha   # darken background
                    ax.imshow(overlay, interpolation="nearest")

            ax.set_title(f"Frame {frame.frame_idx}", fontsize=8)
            ax.axis("off")

        # Hide any unused subplot cells
        for ax in axes[n:]:
            ax.axis("off")

        # ── per-frame spatial-variation annotation ─────────────────────────
        #    mean_reconstruction_error = mean(||p_i - mu||) over foreground patches.
        #    Directly answers: on average, how far is each patch from the mean used
        #    to represent it?  Compare against delta_feature_mag — if they are
        #    similar in magnitude, frame-to-frame deltas are drowned in pooling noise.
        print("\n[FeaturePoseTracker] Mean-pool reconstruction error per frame "
              "(compare against delta_feature_mag — if similar, pooling is lossy):")
        print(f"  {'frame_idx':>9}  {'patches':>7}  {'mean_recon_err':>14}")
        for frame, (pixels, H, W) in zip(frames_with_map, parsed):
            fg_pixels = FeaturePoseTracker._foreground_patches(pixels, frame.mask, H, W)
            mu = fg_pixels.mean(axis=0)
            recon_err = float(np.linalg.norm(fg_pixels - mu, axis=1).mean())
            print(f"  {frame.frame_idx:>9}  {len(fg_pixels):>7}  {recon_err:>14.4f}")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[FeaturePoseTracker] PCA visualisation saved to {save_path}")
        else:
            plt.show()

    # ------------------------------------------------------------------

    def assess_spatial_variation(self) -> Dict[str, Any]:
        """
        Quantify how valid the spatial mean-pool is, without plotting.

        For every frame that has a patch_feature_map, compute (over foreground
        patches only):

          - ``mean_reconstruction_error`` : mean(||p_i - mu||) where mu is the
                                            spatial mean.  This is the primary
                                            validity metric: it lives in the same
                                            feature-space units as delta_feature_mag,
                                            so you can directly compare them.  If
                                            mean_reconstruction_error >> delta_feature_mag,
                                            frame-to-frame deltas are dominated by
                                            pooling noise rather than actual motion.
          - ``pixel_std_mean``            : mean per-dimension std (secondary diagnostic)
          - ``pixel_std_max``             : max per-dimension std
          - ``pca_top3_variance``         : fraction of variance in the first 3 PCs
                                            (high -> PCA colour image is representative)

        Returns a list of per-frame dicts plus an ``aggregate`` entry.
        """
        from sklearn.decomposition import PCA

        frames_with_map = [f for f in self.frames if f.patch_feature_map is not None]
        if not frames_with_map:
            print("[FeaturePoseTracker] No patch_feature_map found; skipping spatial variation.")
            return {}

        records: List[Dict[str, Any]] = []
        all_pixels_list = []

        for frame in frames_with_map:
            m = np.asarray(frame.patch_feature_map, dtype=np.float64)
            if m.ndim == 2:
                D, N = m.shape
                H = W = int(round(N ** 0.5))
                m = m.reshape(D, H, W)
            D, H, W = m.shape
            pixels    = m.reshape(D, H * W).T                                    # [H*W, D]
            fg_pixels = FeaturePoseTracker._foreground_patches(pixels, frame.mask, H, W)
            all_pixels_list.append(fg_pixels)

            mu          = fg_pixels.mean(axis=0)
            per_dim_std = fg_pixels.std(axis=0)
            recon_err   = float(np.linalg.norm(fg_pixels - mu, axis=1).mean())

            pca = PCA(n_components=min(3, D), random_state=0).fit(fg_pixels)
            top3_var = float(pca.explained_variance_ratio_.sum())

            rec = {
                "frame_idx":                 frame.frame_idx,
                "spatial_h":                 H,
                "spatial_w":                 W,
                "feature_dim":               D,
                "num_fg_patches":            int(len(fg_pixels)),
                "mean_reconstruction_error": recon_err,
                "pixel_std_mean":            float(per_dim_std.mean()),
                "pixel_std_max":             float(per_dim_std.max()),
                "pca_top3_variance":         top3_var,
            }
            records.append(rec)

        # Aggregate across frames
        all_pixels = np.concatenate(all_pixels_list, axis=0)
        global_pca  = PCA(n_components=min(3, all_pixels.shape[1]), random_state=0).fit(all_pixels)
        aggregate = {
            "num_frames_with_map":             len(frames_with_map),
            "mean_reconstruction_error":       float(np.mean([r["mean_reconstruction_error"] for r in records])),
            "mean_pca_top3_variance":          float(np.mean([r["pca_top3_variance"] for r in records])),
            "global_pca_top3_variance":        float(global_pca.explained_variance_ratio_.sum()),
        }

        print("\n=== Spatial Feature Variation Assessment ===")
        print(f"  {'frame_idx':>9}  {'H×W':>7}  {'D':>5}  {'fg_patches':>10}  "
              f"{'mean_recon_err':>14}  {'PCA top-3%':>11}")
        for r in records:
            print(
                f"  {r['frame_idx']:>9}  "
                f"{r['spatial_h']}×{r['spatial_w']:>3}  "
                f"{r['feature_dim']:>5}  "
                f"{r['num_fg_patches']:>10}  "
                f"{r['mean_reconstruction_error']:>14.4f}  "
                f"{100*r['pca_top3_variance']:>10.1f}%"
            )
        print(f"\n  Aggregate:")
        for k, v in aggregate.items():
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

        hint = aggregate["mean_reconstruction_error"]
        if hint > 0.3:
            print("\n  ⚠  High reconstruction error (>0.3): mean-pooling discards significant"
                  " spatial structure. Compare against your delta_feature_mag — if they are"
                  " similar, frame-to-frame deltas are dominated by pooling noise.")
        else:
            print("\n  ✓  Low reconstruction error: mean-pooling is a reasonable summary for these frames.")

        return {"per_frame": records, "aggregate": aggregate}

    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        deltas = self.compute_deltas()
        n = len(self.frames)
        stats: Dict[str, Any] = {
            "num_frames":   n,
            "num_deltas":   max(n - 1, 0),
            "valid_frames": sum(1 for f in self.frames if f.valid),
            "feature_dim":  self.frames[0].object_feature.shape[0] if n > 0 else 0,
        }
        if len(deltas["delta_feature_mag"]) > 0:
            for key in ["delta_feature_mag", "delta_translation_mag", "delta_rotation_mag", "delta_pose_mag"]:
                vals = deltas[key]
                v = vals[~np.isnan(vals)] if np.any(np.isnan(vals)) else vals
                if len(v) > 0:
                    stats[f"{key}_mean"] = float(np.mean(v))
                    stats[f"{key}_std"]  = float(np.std(v))
                    stats[f"{key}_max"]  = float(np.max(v))
            vc = deltas["cosine_similarity"]
            vc = vc[~np.isnan(vc)]
            if len(vc) > 0:
                stats["mean_cosine_similarity"] = float(np.mean(vc))
            df, dp = deltas["delta_feature_mag"], deltas["delta_pose_mag"]
            if len(df) > 2 and np.std(df) > 1e-10 and np.std(dp) > 1e-10:
                stats["correlation_feature_pose"] = float(np.corrcoef(df, dp)[0, 1])

        # Mean-pool validity: mean(||p_i - mu||) over foreground patches.
        # Computed here without the full assess_spatial_variation overhead.
        frames_with_map = [f for f in self.frames if f.patch_feature_map is not None]
        if frames_with_map:
            errs = []
            for frame in frames_with_map:
                m = np.asarray(frame.patch_feature_map, dtype=np.float64)
                if m.ndim == 2:
                    D, N = m.shape
                    S = int(round(N ** 0.5))
                    m = m.reshape(D, S, S)
                D, H, W = m.shape
                pixels    = m.reshape(D, H * W).T
                fg_pixels = FeaturePoseTracker._foreground_patches(pixels, frame.mask, H, W)
                mu        = fg_pixels.mean(axis=0)
                errs.append(float(np.linalg.norm(fg_pixels - mu, axis=1).mean()))
            stats["mean_reconstruction_error"] = float(np.mean(errs))

        print("=== FeaturePoseTracker Summary ===")
        for k, v in stats.items():
            print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
        if "mean_reconstruction_error" in stats and "delta_feature_mag_mean" in stats:
            ratio = stats["mean_reconstruction_error"] / max(stats["delta_feature_mag_mean"], 1e-10)
            flag  = "⚠  pooling noise may dominate deltas" if ratio > 1.0 else "✓  deltas are meaningful"
            print(f"  recon_err / delta_feat_mean = {ratio:.3f}  →  {flag}")
        return stats

    # ------------------------------------------------------------------

    def to_csv(self, path: str) -> None:
        import csv
        deltas = self.compute_deltas()
        n = len(self.frames)

        def _fmt(v: Any) -> str:
            return "" if v == "" or (isinstance(v, float) and np.isnan(v)) else f"{v:.6f}"

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "frame_idx", "valid", "image_path",
                "tx", "ty", "tz", "rx_deg", "ry_deg", "rz_deg", "scale",
                "delta_feature_mag", "delta_translation_mag",
                "delta_rotation_mag", "delta_pose_mag", "cosine_similarity",
            ])
            for i in range(n):
                fr = self.frames[i]
                rd = np.degrees(fr.rotation_euler)
                if i > 0 and i - 1 < len(deltas["delta_feature_mag"]):
                    j = i - 1
                    row_deltas = [
                        deltas["delta_feature_mag"][j],
                        deltas["delta_translation_mag"][j],
                        deltas["delta_rotation_mag"][j],
                        deltas["delta_pose_mag"][j],
                        deltas["cosine_similarity"][j],
                    ]
                else:
                    row_deltas = [""] * 5
                w.writerow([
                    fr.frame_idx, fr.valid, fr.image_path or "",
                    f"{fr.translation[0]:.6f}", f"{fr.translation[1]:.6f}", f"{fr.translation[2]:.6f}",
                    f"{rd[0]:.4f}", f"{rd[1]:.4f}", f"{rd[2]:.4f}", f"{fr.scale:.6f}",
                    *[_fmt(d) for d in row_deltas],
                ])
        print(f"[FeaturePoseTracker] CSV written to {path}")
