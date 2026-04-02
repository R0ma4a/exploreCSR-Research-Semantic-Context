"""
FeaturePoseTracker — tracks DINO features and CAPTRA poses across frames.

This is the "Combination Module" from the pipeline architecture:
  DINO features + CAPTRA pose → delta analysis → visual feedback

Stores per-frame data, computes inter-frame deltas, and produces
scatter plots of delta_pose_magnitude vs delta_feature_magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FrameRecord:
    """All tracked data for a single frame."""

    frame_idx: int
    # DINO features (L2-normalized)
    object_feature: np.ndarray     # (D,) masked object patch mean
    cls_feature: np.ndarray        # (D,) CLS token
    # CAPTRA pose
    pose_vector: np.ndarray        # (7,) [tx, ty, tz, rx, ry, rz, scale]
    translation: np.ndarray        # (3,)
    rotation_euler: np.ndarray     # (3,)
    scale: float
    valid: bool
    # Optional metadata
    image_path: Optional[str] = None
    timestamp: Optional[float] = None


class FeaturePoseTracker:
    """
    Accumulates DINO features and CAPTRA poses across a sequence of frames,
    computes inter-frame deltas, and produces analysis / plots.

    Usage
    -----
    tracker = FeaturePoseTracker()

    for frame_idx, (image_tensor, captra_output, mask) in enumerate(frames):
        features = extract_object_features(segmenter, image_tensor, mask)
        tracker.add_frame(frame_idx, features, captra_output)

    tracker.plot()
    tracker.summary()
    """

    def __init__(self, feature_key: str = "object_mean") -> None:
        """
        Parameters
        ----------
        feature_key : which feature to use for delta computation.
            "object_mean" — mean of masked patch tokens (recommended)
            "cls_token"   — CLS token (global image summary)
        """
        self.feature_key = feature_key
        self.frames: List[FrameRecord] = []

    def add_frame(
        self,
        frame_idx: int,
        features: Dict[str, Any],
        captra_output: Dict[str, Any],
        image_path: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Record one frame's DINO features and CAPTRA pose output.

        Parameters
        ----------
        frame_idx     : sequential frame number
        features      : dict from ``extract_object_features``
        captra_output : dict from ``CAPTRA.forward`` or ``pipeline.run_*``
        image_path    : optional source image path
        timestamp     : optional timestamp (seconds)
        """
        # Extract the chosen feature vector
        obj_feat = np.asarray(features.get("object_mean", np.zeros(1)), dtype=np.float64)
        cls_feat = np.asarray(features.get("cls_token", np.zeros(1)), dtype=np.float64)

        # Extract pose
        pose_vector = np.asarray(
            captra_output.get("pose_vector", np.full(7, np.nan)), dtype=np.float64
        )
        translation = np.asarray(
            captra_output.get("translation", np.full(3, np.nan)), dtype=np.float64
        )
        rotation_euler = np.asarray(
            captra_output.get("rotation_euler", np.full(3, np.nan)), dtype=np.float64
        )
        scale = float(captra_output.get("scale", np.nan))
        valid = bool(captra_output.get("valid", False))

        self.frames.append(
            FrameRecord(
                frame_idx=frame_idx,
                object_feature=obj_feat,
                cls_feature=cls_feat,
                pose_vector=pose_vector,
                translation=translation,
                rotation_euler=rotation_euler,
                scale=scale,
                valid=valid,
                image_path=image_path,
                timestamp=timestamp,
            )
        )

    # ==================================================================
    # Delta computation
    # ==================================================================

    def compute_deltas(self) -> Dict[str, np.ndarray]:
        """
        Compute inter-frame deltas for features and pose.

        Returns dict with arrays of length (N-1) where N = number of frames:
        - delta_feature_mag  : ||feature_t - feature_{t-1}||
        - delta_translation_mag : ||translation_t||  (already a delta from CAPTRA)
        - delta_rotation_mag : ||rotation_euler_t||
        - delta_pose_mag     : ||[tx, ty, tz, rx, ry, rz]||  (combined)
        - delta_scale        : |scale_t - 1.0|
        - frame_indices      : frame index of each delta (t, not t-1)
        - cosine_similarity  : cosine sim between consecutive feature vectors
        """
        if len(self.frames) < 2:
            return {
                "delta_feature_mag": np.array([]),
                "delta_translation_mag": np.array([]),
                "delta_rotation_mag": np.array([]),
                "delta_pose_mag": np.array([]),
                "delta_scale": np.array([]),
                "frame_indices": np.array([]),
                "cosine_similarity": np.array([]),
            }

        n = len(self.frames) - 1
        delta_feat = np.zeros(n)
        delta_trans = np.zeros(n)
        delta_rot = np.zeros(n)
        delta_pose = np.zeros(n)
        delta_scale = np.zeros(n)
        cosine_sim = np.zeros(n)
        indices = np.zeros(n, dtype=int)

        for i in range(n):
            prev = self.frames[i]
            curr = self.frames[i + 1]

            # Feature delta
            feat_key = "object_feature" if self.feature_key == "object_mean" else "cls_feature"
            f_prev = getattr(prev, feat_key)
            f_curr = getattr(curr, feat_key)
            diff = f_curr - f_prev
            delta_feat[i] = np.linalg.norm(diff)

            # Cosine similarity
            norm_prev = np.linalg.norm(f_prev)
            norm_curr = np.linalg.norm(f_curr)
            if norm_prev > 1e-8 and norm_curr > 1e-8:
                cosine_sim[i] = np.dot(f_prev, f_curr) / (norm_prev * norm_curr)
            else:
                cosine_sim[i] = np.nan

            # Pose deltas (CAPTRA already reports these as inter-frame changes)
            delta_trans[i] = np.linalg.norm(curr.translation)
            delta_rot[i] = np.linalg.norm(curr.rotation_euler)
            delta_pose[i] = np.linalg.norm(
                np.concatenate([curr.translation, curr.rotation_euler])
            )
            delta_scale[i] = abs(curr.scale - 1.0)
            indices[i] = curr.frame_idx

        return {
            "delta_feature_mag": delta_feat,
            "delta_translation_mag": delta_trans,
            "delta_rotation_mag": delta_rot,
            "delta_pose_mag": delta_pose,
            "delta_scale": delta_scale,
            "frame_indices": indices,
            "cosine_similarity": cosine_sim,
        }

    # ==================================================================
    # Plotting
    # ==================================================================

    def plot(
        self,
        alpha: float = 0.1,
        figsize: Tuple[int, int] = (16, 10),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Generate scatter and time-series plots of delta features vs delta pose.

        Creates a 2×3 figure:
          Row 1: scatter plots (delta_feature vs translation / rotation / combined)
          Row 2: time series  (feature + translation + rotation over frame index)

        Parameters
        ----------
        alpha     : rotation scaling factor for combined magnitude
        figsize   : figure size
        save_path : if given, save figure to this path instead of showing
        """
        import matplotlib.pyplot as plt

        deltas = self.compute_deltas()

        if len(deltas["delta_feature_mag"]) == 0:
            print("[FeaturePoseTracker] Need at least 2 frames to plot deltas.")
            return

        df = deltas["delta_feature_mag"]
        dt = deltas["delta_translation_mag"]
        dr = deltas["delta_rotation_mag"]
        dp = deltas["delta_pose_mag"]
        idx = deltas["frame_indices"]
        cos = deltas["cosine_similarity"]

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle("Delta Features vs Delta Pose", fontsize=14, fontweight="bold")

        # --- Row 1: Scatter plots ---
        axes[0, 0].scatter(df, dt, c=idx, cmap="viridis", edgecolors="k", alpha=0.7)
        axes[0, 0].set_xlabel("Δ Feature Magnitude")
        axes[0, 0].set_ylabel("Δ Translation Magnitude")
        axes[0, 0].set_title("Feature Change vs Translation")

        axes[0, 1].scatter(df, dr, c=idx, cmap="viridis", edgecolors="k", alpha=0.7)
        axes[0, 1].set_xlabel("Δ Feature Magnitude")
        axes[0, 1].set_ylabel("Δ Rotation Magnitude (rad)")
        axes[0, 1].set_title("Feature Change vs Rotation")

        sc = axes[0, 2].scatter(df, dp, c=idx, cmap="viridis", edgecolors="k", alpha=0.7)
        axes[0, 2].set_xlabel("Δ Feature Magnitude")
        axes[0, 2].set_ylabel("Δ Combined Pose Magnitude")
        axes[0, 2].set_title("Feature Change vs Combined Pose")
        fig.colorbar(sc, ax=axes[0, 2], label="Frame Index")

        # --- Row 2: Time series ---
        axes[1, 0].plot(idx, df, "o-", color="tab:blue", label="Δ Feature")
        axes[1, 0].set_xlabel("Frame Index")
        axes[1, 0].set_ylabel("Magnitude")
        axes[1, 0].set_title("Feature Change Over Time")
        axes[1, 0].legend()

        axes[1, 1].plot(idx, dt, "s-", color="tab:red", label="Δ Translation")
        axes[1, 1].plot(idx, dr, "^-", color="tab:green", label="Δ Rotation")
        axes[1, 1].set_xlabel("Frame Index")
        axes[1, 1].set_ylabel("Magnitude")
        axes[1, 1].set_title("Pose Change Over Time")
        axes[1, 1].legend()

        axes[1, 2].plot(idx, 1.0 - cos, "D-", color="tab:purple", label="1 - cos(sim)")
        axes[1, 2].set_xlabel("Frame Index")
        axes[1, 2].set_ylabel("Feature Dissimilarity")
        axes[1, 2].set_title("Cosine Dissimilarity Over Time")
        axes[1, 2].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[FeaturePoseTracker] Saved plot to {save_path}")
        else:
            plt.show()

    # ==================================================================
    # Summary
    # ==================================================================

    def summary(self) -> Dict[str, Any]:
        """Print and return summary statistics."""
        deltas = self.compute_deltas()
        n = len(self.frames)

        stats = {
            "num_frames": n,
            "num_deltas": max(n - 1, 0),
            "valid_frames": sum(1 for f in self.frames if f.valid),
            "feature_dim": self.frames[0].object_feature.shape[0] if n > 0 else 0,
        }

        if len(deltas["delta_feature_mag"]) > 0:
            for key in ["delta_feature_mag", "delta_translation_mag", "delta_rotation_mag", "delta_pose_mag"]:
                vals = deltas[key]
                stats[f"{key}_mean"] = float(np.mean(vals))
                stats[f"{key}_std"] = float(np.std(vals))
                stats[f"{key}_max"] = float(np.max(vals))

            cos = deltas["cosine_similarity"]
            valid_cos = cos[~np.isnan(cos)]
            if len(valid_cos) > 0:
                stats["mean_cosine_similarity"] = float(np.mean(valid_cos))

            # Correlation between feature change and pose change
            df = deltas["delta_feature_mag"]
            dp = deltas["delta_pose_mag"]
            if len(df) > 2 and np.std(df) > 1e-10 and np.std(dp) > 1e-10:
                stats["correlation_feature_pose"] = float(np.corrcoef(df, dp)[0, 1])

        print("=== FeaturePoseTracker Summary ===")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")

        return stats

    # ==================================================================
    # Export
    # ==================================================================

    def to_csv(self, path: str) -> None:
        """Export per-frame data and deltas to CSV."""
        import csv

        deltas = self.compute_deltas()
        n = len(self.frames)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame_idx", "valid", "image_path",
                "tx", "ty", "tz", "rx_deg", "ry_deg", "rz_deg", "scale",
                "delta_feature_mag", "delta_translation_mag",
                "delta_rotation_mag", "delta_pose_mag",
                "cosine_similarity",
            ])

            for i in range(n):
                fr = self.frames[i]
                r_deg = np.degrees(fr.rotation_euler)

                # Deltas are offset by 1 (first frame has no delta)
                if i > 0 and i - 1 < len(deltas["delta_feature_mag"]):
                    j = i - 1
                    d_feat = deltas["delta_feature_mag"][j]
                    d_trans = deltas["delta_translation_mag"][j]
                    d_rot = deltas["delta_rotation_mag"][j]
                    d_pose = deltas["delta_pose_mag"][j]
                    cos_sim = deltas["cosine_similarity"][j]
                else:
                    d_feat = d_trans = d_rot = d_pose = cos_sim = ""

                writer.writerow([
                    fr.frame_idx, fr.valid, fr.image_path or "",
                    f"{fr.translation[0]:.6f}", f"{fr.translation[1]:.6f}", f"{fr.translation[2]:.6f}",
                    f"{r_deg[0]:.4f}", f"{r_deg[1]:.4f}", f"{r_deg[2]:.4f}",
                    f"{fr.scale:.6f}",
                    f"{d_feat:.6f}" if d_feat != "" else "",
                    f"{d_trans:.6f}" if d_trans != "" else "",
                    f"{d_rot:.6f}" if d_rot != "" else "",
                    f"{d_pose:.6f}" if d_pose != "" else "",
                    f"{cos_sim:.6f}" if cos_sim != "" else "",
                ])

        print(f"[FeaturePoseTracker] CSV written to {path}")
