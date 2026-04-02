"""
Lightweight numerical validation for CAPTRA's geometric computations.

All checks use synthetic point clouds — no RGB, depth, or real data needed.
Validates identity behavior, known translations/rotations/scales, axis
orthonormality, and alignment residuals.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .captra import CAPTRA, CAPTRAReferenceState


def _random_rotation_matrix(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(3, 3))
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def _apply_transform(
    points: np.ndarray, R: np.ndarray, t: np.ndarray, s: float
) -> np.ndarray:
    return (s * (R @ points.T)).T + t.reshape(1, 3)


def check_identity_behavior(captra: CAPTRA, points: np.ndarray) -> Dict[str, float]:
    """Same state twice → near-zero pose change."""
    ref1, _ = captra.estimate_object_reference(points)
    ref2, _ = captra.estimate_object_reference(points)
    pose = captra.estimate_pose_change(ref2, ref1)
    return {
        "translation_norm": float(np.linalg.norm(pose["translation"])),
        "rotation_angle_rad": float(np.linalg.norm(pose["rotation_euler"])),
        "scale_error": float(abs(pose["scale"] - 1.0)),
    }


def check_known_translation(captra: CAPTRA, points: np.ndarray) -> Dict[str, float]:
    """Apply a known translation and verify recovery."""
    t_true = np.array([0.1, -0.05, 0.02], dtype=np.float64)
    ref_prev, _ = captra.estimate_object_reference(points)
    moved = _apply_transform(points, np.eye(3), t_true, 1.0)
    ref_curr, _ = captra.estimate_object_reference(moved)
    pose = captra.estimate_pose_change(ref_curr, ref_prev)
    return {
        "translation_error": float(np.linalg.norm(pose["translation"] - t_true)),
        "rotation_angle_rad": float(np.linalg.norm(pose["rotation_euler"])),
        "scale_error": float(abs(pose["scale"] - 1.0)),
    }


def check_known_rotation(captra: CAPTRA, points: np.ndarray) -> Dict[str, float]:
    """Apply a known rotation and verify recovery."""
    R_true = _random_rotation_matrix(seed=123)
    ref_prev, _ = captra.estimate_object_reference(points)
    rotated = _apply_transform(points, R_true, np.zeros(3), 1.0)
    ref_curr, _ = captra.estimate_object_reference(rotated)
    pose = captra.estimate_pose_change(ref_curr, ref_prev)

    R_rel = R_true.T @ pose["rotation_matrix"]
    angle = float(np.arccos(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)))

    return {
        "rotation_angle_error_rad": angle,
        "translation_norm": float(np.linalg.norm(pose["translation"])),
        "scale_error": float(abs(pose["scale"] - 1.0)),
    }


def check_known_scale(
    captra: CAPTRA, points: np.ndarray, scale_factor: float = 1.5
) -> Dict[str, float]:
    """Apply known uniform scaling and verify recovery."""
    ref_prev, _ = captra.estimate_object_reference(points)
    scaled = _apply_transform(points, np.eye(3), np.zeros(3), scale_factor)
    ref_curr, _ = captra.estimate_object_reference(scaled)
    pose = captra.estimate_pose_change(ref_curr, ref_prev)
    return {
        "scale_error": float(abs(pose["scale"] - scale_factor)),
        "translation_norm": float(np.linalg.norm(pose["translation"])),
        "rotation_angle_rad": float(np.linalg.norm(pose["rotation_euler"])),
    }


def check_axes_orthonormal(axes: np.ndarray) -> float:
    """Orthonormality check: ||A^T A - I||_F."""
    return float(np.linalg.norm(axes.T @ axes - np.eye(3), ord="fro"))


def check_alignment_residual(
    prev_state: CAPTRAReferenceState,
    curr_state: CAPTRAReferenceState,
    captra: CAPTRA,
) -> float:
    """RMS residual between states after applying the estimated pose."""
    pose = captra.estimate_pose_change(curr_state, prev_state)
    R, t, s = pose["rotation_matrix"], pose["translation"], pose["scale"]

    if prev_state.points_sample is None or curr_state.points_sample is None:
        return float("nan")

    n = min(prev_state.points_sample.shape[0], curr_state.points_sample.shape[0])
    if n == 0:
        return float("nan")

    P_prev = prev_state.points_sample[:n]
    P_curr = curr_state.points_sample[:n]
    P_aligned = (s * (R @ P_prev.T)).T + t.reshape(1, 3)
    residuals = P_curr - P_aligned
    return float(np.sqrt((residuals**2).sum(axis=1).mean()))


def run_all_synthetic_checks(num_points: int = 500) -> Dict[str, Dict[str, float]]:
    """
    Run a suite of lightweight numerical validation checks on synthetic data.

    Returns a dict of test_name → metrics for each check.
    """
    rng = np.random.default_rng(42)
    base = rng.normal(size=(num_points, 3))
    base *= np.array([0.3, 0.1, 0.05])[None, :]  # anisotropic cloud

    K = np.array(
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    captra = CAPTRA(camera_intrinsics=K)

    ref, _ = captra.estimate_object_reference(base)
    axes_err = check_axes_orthonormal(ref.axes) if ref is not None else float("nan")

    return {
        "identity": check_identity_behavior(captra, base),
        "translation": check_known_translation(captra, base),
        "rotation": check_known_rotation(captra, base),
        "scale": check_known_scale(captra, base, scale_factor=1.5),
        "axes_orthonormality": {"frobenius_error": axes_err},
    }


if __name__ == "__main__":
    results = run_all_synthetic_checks()
    print("=== CAPTRA Synthetic Validation ===")
    for name, metrics in results.items():
        print(f"[{name}]")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
