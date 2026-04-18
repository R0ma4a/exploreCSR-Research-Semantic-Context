from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class CAPTRAReferenceState:
    """
    Lightweight container for the object-centered reference frame.

    This state is meant to be passed between frames so that CAPTRA can
    estimate pose changes over time without re-deriving everything from scratch.
    """

    centroid: np.ndarray  # (3,)
    axes: np.ndarray  # (3, 3) principal axes as column vectors
    extents: np.ndarray  # (3,) spread along each axis
    points_sample: Optional[np.ndarray] = None  # (N, 3) optional downsampled point cloud
    meta: Optional[Dict[str, Any]] = None


class CAPTRA:
    """
    CAPTRA: Object-centered pose estimation from RGB-D and mask.

    This class focuses on the geometric / pose side of the pipeline:

    - Takes RGB, depth, and an object mask or segmentation map
    - Extracts the object region
    - Converts masked RGB-D into a 3D point cloud
    - Estimates a stable object-centered reference frame via PCA
    - Compares current frame to a previous reference to estimate pose change

    Feature changes (for the graph's y-axis) are handled separately upstream;
    this module only returns geometric and reference-frame information plus
    sufficient intermediates for debugging and visualization.
    """

    def __init__(
        self,
        camera_intrinsics: np.ndarray,
        depth_scale: float = 1.0,
        min_points: int = 50,
        pca_eps: float = 1e-6,
        max_points_for_state: int = 10_000,
    ) -> None:
        """
        Initialize CAPTRA.

        Parameters
        ----------
        camera_intrinsics:
            \(3 \times 3\) pinhole intrinsics matrix \(\mathbf{K}\) with
            fx, fy, cx, cy in the usual positions.
        depth_scale:
            Optional multiplicative scale factor to convert raw depth units
            into metric units. Set to 1.0 if depth is already in meters.
        min_points:
            Minimum number of valid depth points required to compute a
            meaningful reference frame.
        pca_eps:
            Small regularization term used to guard against degenerate
            covariance matrices during PCA.
        max_points_for_state:
            Maximum number of points to store inside the reference state
            (for memory control). If exceeded, a random subset is kept.
        """
        self.K = np.asarray(camera_intrinsics, dtype=np.float64)
        if self.K.shape != (3, 3):
            raise ValueError(f"camera_intrinsics must be 3x3, got {self.K.shape}")

        self.depth_scale = float(depth_scale)
        self.min_points = int(min_points)
        self.pca_eps = float(pca_eps)
        self.max_points_for_state = int(max_points_for_state)

    # ------------------------------------------------------------------
    # Stage 1: Extract object region
    # ------------------------------------------------------------------
    def extract_object_region(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        seg_or_mask: np.ndarray,
        target_label: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Extract the object-specific region from RGB-D using a mask or segmentation.

        Parameters
        ----------
        rgb:
            RGB image as H x W x 3 array (uint8 or float).
        depth:
            Depth map as H x W array. NaNs or non-positive values are treated
            as invalid and removed.
        seg_or_mask:
            Either a binary mask (H x W, {0, 1}) or a label map (H x W, ints).
        target_label:
            If `seg_or_mask` is a label map, this specifies which label index
            to use as the object mask. If None and values are not boolean,
            the largest non-zero label is selected by default.

        Returns
        -------
        dict
            Contains:
            - `mask`             : binary mask (H, W), dtype=bool
            - `masked_rgb`       : RGB values where mask is True, zeros elsewhere
            - `masked_depth`     : depth values where mask is True, 0 elsewhere
            - `valid_indices`    : (N, 2) array of (v, u) coordinates with valid depth
            - `valid_depth`      : (N,) array of valid depth values
            - `diagnostics`      : dict with simple stats
        """
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"rgb must be HxWx3, got {rgb.shape}")
        if depth.shape != rgb.shape[:2]:
            raise ValueError("depth must have shape HxW matching rgb")
        if seg_or_mask.shape != rgb.shape[:2]:
            raise ValueError("seg_or_mask must have shape HxW matching rgb")

        # Derive binary mask
        if seg_or_mask.dtype == bool:
            mask = seg_or_mask.copy()
        else:
            unique_vals = np.unique(seg_or_mask)
            if target_label is not None:
                mask = seg_or_mask == target_label
            else:
                # Heuristic: choose the most frequent non-zero label
                non_zero = unique_vals[unique_vals != 0]
                if non_zero.size == 0:
                    mask = np.zeros_like(seg_or_mask, dtype=bool)
                else:
                    # Pick label with max count
                    counts = [(lbl, np.count_nonzero(seg_or_mask == lbl)) for lbl in non_zero]
                    chosen_label = max(counts, key=lambda x: x[1])[0]
                    mask = seg_or_mask == chosen_label

        # Masked RGB / depth
        masked_rgb = np.zeros_like(rgb)
        masked_rgb[mask] = rgb[mask]

        masked_depth = np.zeros_like(depth, dtype=np.float64)
        depth_float = np.asarray(depth, dtype=np.float64) * self.depth_scale
        masked_depth[mask] = depth_float[mask]

        # Valid depth points: within mask, positive, finite
        valid = mask & np.isfinite(depth_float) & (depth_float > 0.0)
        vs, us = np.where(valid)
        valid_indices = np.stack([vs, us], axis=-1) if vs.size > 0 else np.empty((0, 2), dtype=int)
        valid_depth = depth_float[valid]

        diagnostics = {
            "num_mask_pixels": int(mask.sum()),
            "num_valid_depth": int(valid_depth.size),
        }

        return {
            "mask": mask,
            "masked_rgb": masked_rgb,
            "masked_depth": masked_depth,
            "valid_indices": valid_indices,
            "valid_depth": valid_depth,
            "diagnostics": diagnostics,
        }

    # ------------------------------------------------------------------
    # Stage 2: RGB-D to point cloud
    # ------------------------------------------------------------------
    def rgbd_to_pointcloud(
        self,
        valid_indices: np.ndarray,
        valid_depth: np.ndarray,
        rgb: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Convert valid masked RGB-D pixels into a 3D point cloud.

        Parameters
        ----------
        valid_indices:
            (N, 2) pixel coordinates as (v, u) pairs.
        valid_depth:
            (N,) depth values corresponding to `valid_indices`.
        rgb:
            Optional full RGB image H x W x 3. If provided, per-point colors
            will be extracted.

        Returns
        -------
        dict
            - `points` : (N, 3) 3D points in camera coordinates
            - `colors` : (N, 3) uint8 colors (if rgb provided), else None
        """
        if valid_indices.size == 0 or valid_depth.size == 0:
            return {"points": np.empty((0, 3), dtype=np.float64), "colors": None}

        if valid_indices.shape[0] != valid_depth.shape[0]:
            raise ValueError("valid_indices and valid_depth must have same length")

        # Camera intrinsics
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        vs = valid_indices[:, 0].astype(np.float64)
        us = valid_indices[:, 1].astype(np.float64)
        z = valid_depth.astype(np.float64)

        # Pinhole back-projection
        x = (us - cx) * z / fx
        y = (vs - cy) * z / fy

        points = np.stack([x, y, z], axis=-1)

        colors = None
        if rgb is not None:
            colors = rgb[valid_indices[:, 0], valid_indices[:, 1]].astype(np.uint8)

        return {"points": points, "colors": colors}

    # ------------------------------------------------------------------
    # Stage 3: Estimate object reference frame via PCA
    # ------------------------------------------------------------------
    def estimate_object_reference(
        self,
        points: np.ndarray,
    ) -> Tuple[Optional[CAPTRAReferenceState], Dict[str, Any]]:
        """
        Estimate an object-centered reference frame from a point cloud.

        The reference frame is defined by:
        - centroid (mean of points)
        - principal axes (via PCA on centered coordinates)
        - extents (standard deviation along each principal axis)

        Parameters
        ----------
        points:
            (N, 3) array of 3D points in camera coordinates.

        Returns
        -------
        (reference_state, diagnostics)
            reference_state:
                `CAPTRAReferenceState` instance, or None if estimation failed.
            diagnostics:
                dict with flags and basic statistics.
        """
        diagnostics: Dict[str, Any] = {
            "num_points": int(points.shape[0]),
            "valid": False,
            "message": "",
        }

        if points.shape[0] < self.min_points:
            diagnostics["message"] = (
                f"Insufficient points for PCA: {points.shape[0]} < {self.min_points}"
            )
            return None, diagnostics

        # Center the data
        centroid = points.mean(axis=0)
        centered = points - centroid

        # Covariance and PCA via SVD (more stable than eig on covariance)
        cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
        cov += self.pca_eps * np.eye(3, dtype=np.float64)

        try:
            U, S, _ = np.linalg.svd(cov)
        except np.linalg.LinAlgError:
            diagnostics["message"] = "SVD failed during PCA"
            return None, diagnostics

        axes = U  # columns are principal directions

        # Ensure right-handed coordinate system (determinant +1)
        if np.linalg.det(axes) < 0:
            axes[:, -1] *= -1.0

        # Extents: standard deviation along each principal axis
        proj = centered @ axes
        extents = proj.std(axis=0)

        # Guard against degenerate PCA (very small singular values)
        if np.all(S < self.pca_eps):
            diagnostics["message"] = "Degenerate covariance; all singular values tiny"
            return None, diagnostics

        # Downsample stored points if necessary to keep memory in check
        points_sample = points
        if points_sample.shape[0] > self.max_points_for_state:
            idx = np.random.choice(points_sample.shape[0], self.max_points_for_state, replace=False)
            points_sample = points_sample[idx]

        ref_state = CAPTRAReferenceState(
            centroid=centroid,
            axes=axes,
            extents=extents,
            points_sample=points_sample,
            meta=None,
        )

        diagnostics.update(
            {
                "valid": True,
                "message": "OK",
                "centroid": centroid,
                "axes": axes,
                "extents": extents,
                "singular_values": S,
            }
        )

        return ref_state, diagnostics

    # ------------------------------------------------------------------
    # Stage 4: Pose change estimation between reference frames
    # ------------------------------------------------------------------
    def estimate_pose_change(
        self,
        current: CAPTRAReferenceState,
        previous: Optional[CAPTRAReferenceState] = None,
    ) -> Dict[str, Any]:
        """
        Estimate pose change (translation, rotation, scale) between frames.

        This uses a simple, robust approximation:
        - Translation: difference between centroids
        - Rotation: rotation that aligns previous axes to current axes
          (R ≈ A_curr @ A_prev^T)
        - Scale: ratio of average extents (current / previous)

        If `previous` is None, the pose is initialized with:
        - zero translation
        - identity rotation
        - unit scale

        Parameters
        ----------
        current:
            Current `CAPTRAReferenceState`.
        previous:
            Previous `CAPTRAReferenceState` (or None on first frame).

        Returns
        -------
        dict
            - `translation`      : (3,) translation vector
            - `rotation_matrix`  : (3, 3) rotation from previous to current
            - `rotation_euler`   : (3,) XYZ Euler angles in radians
            - `scale`            : scalar scale ratio
            - `pose_vector`      : concatenated pose [tx, ty, tz, rx, ry, rz, s]
            - `diagnostics`      : dict with basic info
        """
        if previous is None:
            translation = np.zeros(3, dtype=np.float64)
            rotation_matrix = np.eye(3, dtype=np.float64)
            rotation_euler = self._rotation_matrix_to_euler_xyz(rotation_matrix)
            scale = 1.0

            pose_vector = np.concatenate([translation, rotation_euler, np.array([scale])])

            return {
                "translation": translation,
                "rotation_matrix": rotation_matrix,
                "rotation_euler": rotation_euler,
                "scale": scale,
                "pose_vector": pose_vector,
                "diagnostics": {
                    "initialized": True,
                    "message": "No previous reference; initialized pose to identity",
                },
            }

        # Translation: centroid shift (current - previous)
        translation = current.centroid - previous.centroid

        # Rotation: best-fit aligning previous axes to current axes
        R = current.axes @ previous.axes.T
        # Orthonormalize via SVD for numerical stability
        U, _, Vt = np.linalg.svd(R)
        rotation_matrix = U @ Vt
        if np.linalg.det(rotation_matrix) < 0:
            U[:, -1] *= -1.0
            rotation_matrix = U @ Vt

        rotation_euler = self._rotation_matrix_to_euler_xyz(rotation_matrix)

        # Scale: ratio of mean extents (guard from divide-by-zero)
        prev_extent_mean = max(float(previous.extents.mean()), self.pca_eps)
        curr_extent_mean = float(current.extents.mean())
        scale = curr_extent_mean / prev_extent_mean

        pose_vector = np.concatenate([translation, rotation_euler, np.array([scale])])

        diagnostics = {
            "initialized": False,
            "message": "Pose estimated from previous and current reference frames",
            "prev_centroid": previous.centroid,
            "curr_centroid": current.centroid,
            "prev_extents": previous.extents,
            "curr_extents": current.extents,
        }

        return {
            "translation": translation,
            "rotation_matrix": rotation_matrix,
            "rotation_euler": rotation_euler,
            "scale": scale,
            "pose_vector": pose_vector,
            "diagnostics": diagnostics,
        }

    # ------------------------------------------------------------------
    # High-level forward API
    # ------------------------------------------------------------------
    def forward(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        seg_or_mask: np.ndarray,
        target_label: Optional[int] = None,
        previous_reference_state: Optional[CAPTRAReferenceState] = None,
    ) -> Dict[str, Any]:
        """
        Full CAPTRA forward pass for a single frame.

        Parameters
        ----------
        rgb:
            RGB image as H x W x 3 array.
        depth:
            Depth map as H x W array.
        seg_or_mask:
            Binary mask or segmentation label map (H x W).
        target_label:
            Optional label index if `seg_or_mask` is a label map.
        previous_reference_state:
            Optional `CAPTRAReferenceState` from the previous frame.

        Returns
        -------
        dict
            Core pose outputs (for graph x-axis):
            - `pose_vector`
            - `translation`
            - `rotation_matrix`
            - `rotation_euler`
            - `scale`
            - `reference_state`

            Plus a rich set of intermediates for debugging and visualization:
            - `mask`
            - `masked_rgb`
            - `masked_depth`
            - `object_points`
            - `object_centroid`
            - `principal_axes`
            - `object_extents`
            - `previous_reference_frame`
            - `current_reference_frame`
            - `valid` / `message`
            - any additional diagnostics
        """
        region = self.extract_object_region(rgb, depth, seg_or_mask, target_label=target_label)

        mask = region["mask"]
        masked_rgb = region["masked_rgb"]
        masked_depth = region["masked_depth"]
        valid_indices = region["valid_indices"]
        valid_depth = region["valid_depth"]

        if valid_depth.size < self.min_points:
            # Not enough points to do any meaningful geometry
            pose_init = self.estimate_pose_change(
                current=CAPTRAReferenceState(
                    centroid=np.zeros(3, dtype=np.float64),
                    axes=np.eye(3, dtype=np.float64),
                    extents=np.ones(3, dtype=np.float64),
                    points_sample=None,
                    meta=None,
                ),
                previous=None,
            )

            return {
                "pose_vector": pose_init["pose_vector"],
                "translation": pose_init["translation"],
                "rotation_matrix": pose_init["rotation_matrix"],
                "rotation_euler": pose_init["rotation_euler"],
                "scale": pose_init["scale"],
                "reference_state": None,
                "mask": mask,
                "masked_rgb": masked_rgb,
                "masked_depth": masked_depth,
                "object_points": np.empty((0, 3), dtype=np.float64),
                "object_centroid": None,
                "principal_axes": None,
                "object_extents": None,
                "previous_reference_frame": previous_reference_state,
                "current_reference_frame": None,
                "valid": False,
                "message": f"Insufficient valid depth points: {valid_depth.size} < {self.min_points}",
                "diagnostics": {
                    "region": region["diagnostics"],
                    "pose": pose_init["diagnostics"],
                },
            }

        pc = self.rgbd_to_pointcloud(valid_indices, valid_depth, rgb=rgb)
        points = pc["points"]

        ref_state, ref_diag = self.estimate_object_reference(points)
        if ref_state is None:
            # Geometry failed; propagate diagnostics
            pose_init = self.estimate_pose_change(
                current=CAPTRAReferenceState(
                    centroid=np.zeros(3, dtype=np.float64),
                    axes=np.eye(3, dtype=np.float64),
                    extents=np.ones(3, dtype=np.float64),
                    points_sample=None,
                    meta=None,
                ),
                previous=None,
            )

            return {
                "pose_vector": pose_init["pose_vector"],
                "translation": pose_init["translation"],
                "rotation_matrix": pose_init["rotation_matrix"],
                "rotation_euler": pose_init["rotation_euler"],
                "scale": pose_init["scale"],
                "reference_state": None,
                "mask": mask,
                "masked_rgb": masked_rgb,
                "masked_depth": masked_depth,
                "object_points": points,
                "object_centroid": None,
                "principal_axes": None,
                "object_extents": None,
                "previous_reference_frame": previous_reference_state,
                "current_reference_frame": None,
                "valid": False,
                "message": ref_diag.get("message", "Reference estimation failed"),
                "diagnostics": {
                    "region": region["diagnostics"],
                    "reference": ref_diag,
                    "pose": pose_init["diagnostics"],
                },
            }

        pose_out = self.estimate_pose_change(ref_state, previous_reference_state)

        output: Dict[str, Any] = {
            "pose_vector": pose_out["pose_vector"],
            "translation": pose_out["translation"],
            "rotation_matrix": pose_out["rotation_matrix"],
            "rotation_euler": pose_out["rotation_euler"],
            "scale": pose_out["scale"],
            "reference_state": ref_state,
            "mask": mask,
            "masked_rgb": masked_rgb,
            "masked_depth": masked_depth,
            "object_points": points,
            "object_centroid": ref_state.centroid,
            "principal_axes": ref_state.axes,
            "object_extents": ref_state.extents,
            "previous_reference_frame": previous_reference_state,
            "current_reference_frame": ref_state,
            "valid": True,
            "message": "OK",
            "diagnostics": {
                "region": region["diagnostics"],
                "reference": ref_diag,
                "pose": pose_out["diagnostics"],
            },
        }

        return output

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _rotation_matrix_to_euler_xyz(R: np.ndarray) -> np.ndarray:
        """
        Convert a 3x3 rotation matrix to XYZ Euler angles (radians).

        The convention is:
        - First rotate about X (roll)
        - Then about Y (pitch)
        - Then about Z (yaw)

        This is a standard aerospace / robotics convention and is mostly
        used here so that the pose vector has interpretable components.
        """
        if R.shape != (3, 3):
            raise ValueError(f"R must be 3x3, got {R.shape}")

        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            # Gimbal lock: fall back to alternative representation
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0.0

        return np.array([x, y, z], dtype=np.float64)
