from __future__ import annotations

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

try:
    import cv2  # optional
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None


def show_mask_overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> None:
    """
    Display an RGB image with a semi-transparent mask overlay.

    Parameters
    ----------
    rgb:
        H x W x 3 RGB image (uint8 or float).
    mask:
        H x W binary mask.
    alpha:
        Transparency of the mask overlay.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must be HxWx3, got {rgb.shape}")
    if mask.shape != rgb.shape[:2]:
        raise ValueError("mask must have shape HxW matching rgb")

    rgb_disp = rgb.astype(np.float32)
    if rgb_disp.max() > 1.0:
        rgb_disp /= 255.0

    mask_color = np.zeros_like(rgb_disp)
    mask_color[..., 0] = mask.astype(float)  # red channel

    overlay = (1 - alpha) * rgb_disp + alpha * mask_color

    plt.figure()
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Mask Overlay")
    plt.show()


def show_masked_depth(depth: np.ndarray, mask: np.ndarray) -> None:
    """
    Visualize depth values inside the mask.

    Parameters
    ----------
    depth:
        H x W depth map.
    mask:
        H x W binary mask.
    """
    if depth.shape != mask.shape:
        raise ValueError("depth and mask must have the same shape")

    masked = np.where(mask, depth, np.nan)

    plt.figure()
    im = plt.imshow(masked, cmap="viridis")
    plt.colorbar(im, label="Depth")
    plt.axis("off")
    plt.title("Masked Depth")
    plt.show()


def show_pointcloud(
    points: np.ndarray,
    title: Optional[str] = None,
    s: int = 2,
    max_points: int = 50000,
    cmap: str = "viridis",
) -> None:
    """
    3D scatter plot of a point cloud with points colored by camera-frame depth Z.

    Color encodes ``points[:, 2]`` (same convention as CAPTRA back-projection).

    Parameters
    ----------
    points:
        (N, 3) array of 3D points.
    title:
        Optional plot title.
    s:
        Marker size.
    cmap:
        Colormap for depth (Z) values.
    """
    if points.size == 0:
        print("show_pointcloud: no points to display.")
        return

    # Subsample for visualization if there are too many points
    if points.shape[0] > max_points:
        idx = np.random.choice(points.shape[0], max_points, replace=False)
        points = points[idx]

    z_depth = points[:, 2].astype(np.float64)

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=s,
        c=z_depth,
        cmap=cmap,
        depthshade=False,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if title:
        ax.set_title(title)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.08)
    cbar.set_label("Depth Z (camera frame)")
    plt.show()


def show_reference_frame(
    points: np.ndarray,
    centroid: np.ndarray,
    axes: np.ndarray,
    axis_length: float = 0.1,
    title: Optional[str] = None,
    max_points: int = 50000,
    cmap: str = "viridis",
) -> None:
    """
    Plot a point cloud with an overlaid reference frame (centroid + axes).

    Points are colored by ``points[:, 2]`` (camera-frame depth Z), matching
    the masked-depth intuition in the image plane.

    Parameters
    ----------
    points:
        (N, 3) point cloud.
    centroid:
        (3,) centroid of the object.
    axes:
        (3, 3) principal axes as columns.
    axis_length:
        Length of the rendered axes.
    title:
        Optional plot title.
    max_points:
        Maximum number of points to draw; cloud is subsampled if larger
        to avoid hanging/crashing on huge point clouds.
    cmap:
        Colormap for depth (Z) values.
    """
    if points.size == 0:
        print("show_reference_frame: no points to display.")
        return

    # Subsample for visualization if there are too many points
    if points.shape[0] > max_points:
        idx = np.random.choice(points.shape[0], max_points, replace=False)
        points = points[idx]

    z_depth = points[:, 2].astype(np.float64)

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=2,
        alpha=0.45,
        c=z_depth,
        cmap=cmap,
        depthshade=False,
    )

    c = centroid.reshape(3)
    A = axes

    colors = ["r", "g", "b"]
    labels = ["x", "y", "z"]
    for i in range(3):
        axis_vec = A[:, i] * axis_length
        ax.quiver(
            c[0],
            c[1],
            c[2],
            axis_vec[0],
            axis_vec[1],
            axis_vec[2],
            color=colors[i],
            label=f"{labels[i]}-axis",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if title:
        ax.set_title(title)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.08)
    cbar.set_label("Depth Z (camera frame)")
    plt.legend()
    plt.show()


def compare_pointclouds(
    prev_points: np.ndarray,
    curr_points: np.ndarray,
    title: Optional[str] = None,
    s: int = 2,
) -> None:
    """
    Compare two point clouds in a shared 3D view.

    Parameters
    ----------
    prev_points:
        (N1, 3) previous point cloud.
    curr_points:
        (N2, 3) current point cloud.
    title:
        Optional plot title.
    s:
        Marker size.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    if prev_points.size > 0:
        ax.scatter(prev_points[:, 0], prev_points[:, 1], prev_points[:, 2], s=s, c="b", label="prev")
    if curr_points.size > 0:
        ax.scatter(curr_points[:, 0], curr_points[:, 1], curr_points[:, 2], s=s, c="r", label="curr")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if title:
        ax.set_title(title)
    plt.legend()
    plt.show()


def print_pose_summary(output_dict: Dict[str, Any]) -> None:
    """Print a concise summary of neural CAPTRA pose output for one frame."""
    valid = output_dict.get("valid", True)
    msg   = output_dict.get("message", "")
    t     = output_dict.get("translation")
    R     = output_dict.get("rotation_matrix")
    s     = output_dict.get("scale")

    status = "OK" if valid else "INVALID"
    print(f"  [{status}] {msg}")
    if t is not None:
        print(f"  translation : [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")
    if R is not None:
        det   = float(np.linalg.det(R))
        ortho = float(np.linalg.norm(R.T @ R - np.eye(3), ord="fro"))
        print(f"  rotation    : det={det:.6f}  ortho_err={ortho:.2e}")
    if s is not None:
        print(f"  scale       : {s:.4f}")

