"""
Surface-relative delta pose computation.

Computes pose changes relative to a fitted surface plane, which is useful
for tracking objects moving along a table, floor, or other planar surface.
Also includes camera calibration and plotting utilities.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def compute_surface_delta_poses(
    depth: np.ndarray,
    delta_xyz_seq: np.ndarray,
    delta_theta_seq: np.ndarray,
    K: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute delta poses relative to the object's surface plane.

    Parameters
    ----------
    depth          : (H, W) depth map
    delta_xyz_seq  : (N, 3) per-frame translation deltas
    delta_theta_seq: (N, 3) per-frame rotation deltas (radians)
    K              : (3, 3) camera intrinsic matrix
    mask           : (H, W) bool, object region (None → all pixels)

    Returns
    -------
    delta_poses : (N, 4) — (delta_u, delta_v, delta_h, delta_theta_inplane)
    """
    H, W = depth.shape

    if mask is None:
        mask = np.ones_like(depth, dtype=bool)

    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    us, vs, ds = us[mask], vs[mask], depth[mask]

    # Back-project to 3D
    pixels_h = np.stack([us, vs, np.ones_like(ds)], axis=0)  # (3, N)
    X = (np.linalg.inv(K) @ pixels_h) * ds  # (3, N)
    X = X.T  # (N, 3)

    # Fit plane via PCA
    centroid = X.mean(axis=0)
    cov = np.cov((X - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, np.argmin(eigvals)]

    # Ensure normal points toward camera
    if normal[2] > 0:
        normal = -normal

    # Build plane frame
    u_vec = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
    u_vec = u_vec - np.dot(u_vec, normal) * normal
    u_vec /= np.linalg.norm(u_vec)
    v_vec = np.cross(normal, u_vec)
    F = np.stack([u_vec, v_vec, normal], axis=1)  # (3, 3) plane frame

    # Process sequence
    N = delta_xyz_seq.shape[0]
    delta_poses = np.zeros((N, 4))

    for i in range(N):
        # Translation in plane frame
        delta_t_plane = F.T @ delta_xyz_seq[i]
        du, dv, dh = delta_t_plane

        # Small-angle rotation in plane frame
        dx, dy, dz = delta_theta_seq[i]
        R_delta = np.array([[1, -dz, dy], [dz, 1, -dx], [-dy, dx, 1]])
        R_delta_plane = F.T @ R_delta @ F
        dtheta_inplane = np.arctan2(R_delta_plane[1, 0], R_delta_plane[0, 0])

        delta_poses[i] = [du, dv, dh, dtheta_inplane]

    return delta_poses


def compute_delta_pose_magnitudes(
    delta_poses: np.ndarray, alpha: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scalar magnitudes of delta poses for plotting.

    Parameters
    ----------
    delta_poses : (N, 4) from ``compute_surface_delta_poses``
    alpha       : rotation scaling factor for combined magnitude

    Returns
    -------
    translation_magnitude, rotation_magnitude, combined_magnitude
    """
    translation_mag = np.linalg.norm(delta_poses[:, :3], axis=1)
    rotation_mag = np.abs(delta_poses[:, 3])
    combined_mag = np.sqrt(translation_mag**2 + (alpha * rotation_mag) ** 2)
    return translation_mag, rotation_mag, combined_mag


def calibrate_camera(
    calib_images,
    checkerboard_size: Tuple[int, int] = (9, 6),
    square_size: float = 0.025,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute camera intrinsics from checkerboard calibration images.

    Parameters
    ----------
    calib_images     : list of file paths or BGR images
    checkerboard_size: (cols, rows) inner corners
    square_size      : size of one square in meters

    Returns
    -------
    K    : (3, 3) intrinsic matrix
    dist : distortion coefficients
    """
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = (
        np.mgrid[0 : checkerboard_size[0], 0 : checkerboard_size[1]]
        .T.reshape(-1, 2)
    )
    objp *= square_size

    objpoints, imgpoints = [], []
    gray_shape = None

    for img in calib_images:
        if isinstance(img, str):
            img = cv2.imread(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_shape = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    if not objpoints:
        raise ValueError("No checkerboard corners found in any calibration image.")

    _, K, dist, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, gray_shape, None, None
    )
    return K, dist


def plot_delta_pose_vs_features(
    delta_poses: np.ndarray, features: np.ndarray, alpha: float = 0.1
) -> None:
    """Plot delta pose magnitudes vs feature values."""
    import matplotlib.pyplot as plt

    t_mag, r_mag, c_mag = compute_delta_pose_magnitudes(delta_poses, alpha)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].scatter(features, t_mag)
    axes[0].set_xlabel("Delta Feature")
    axes[0].set_ylabel("Translation Magnitude (m)")
    axes[0].set_title("Translation vs Feature")

    axes[1].scatter(features, r_mag)
    axes[1].set_xlabel("Delta Feature")
    axes[1].set_ylabel("Rotation Magnitude (rad)")
    axes[1].set_title("Rotation vs Feature")

    axes[2].scatter(features, c_mag)
    axes[2].set_xlabel("Delta Feature")
    axes[2].set_ylabel("Combined Pose Magnitude")
    axes[2].set_title("Combined Pose vs Feature")

    plt.tight_layout()
    plt.show()