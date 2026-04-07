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

        print("=== FeaturePoseTracker Summary ===")
        for k, v in stats.items():
            print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
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
