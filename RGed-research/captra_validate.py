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


def _apply_transform(points: np.ndarray, R: np.ndarray, t: np.ndarray, s: float) -> np.ndarray:
    return (s * (R @ points.T)).T + t.reshape(1, 3)


def check_identity_behavior(captra: CAPTRA, points: np.ndarray) -> Dict[str, float]:
    """
    Identity: same state twice should yield near-zero pose.
    """
    ref1, _ = captra.estimate_object_reference(points)
    ref2, _ = captra.estimate_object_reference(points)
    pose = captra.estimate_pose_change(ref2, ref1)

    t_norm = float(np.linalg.norm(pose["translation"]))
    r_angle = float(np.linalg.norm(pose["rotation_euler"]))
    s_err = float(abs(pose["scale"] - 1.0))

    return {
        "translation_norm": t_norm,
        "rotation_angle_rad": r_angle,
        "scale_error": s_err,
    }


def check_known_translation(captra: CAPTRA, points: np.ndarray) -> Dict[str, float]:
    """
    Known translation: apply a pure translation and verify recovery.
    """
    t_true = np.array([0.1, -0.05, 0.02], dtype=np.float64)
    R_id = np.eye(3, dtype=np.float64)

    ref_prev, _ = captra.estimate_object_reference(points)
    moved_points = _apply_transform(points, R_id, t_true, 1.0)
    ref_curr, _ = captra.estimate_object_reference(moved_points)

    pose = captra.estimate_pose_change(ref_curr, ref_prev)

    t_err = float(np.linalg.norm(pose["translation"] - t_true))
    r_angle = float(np.linalg.norm(pose["rotation_euler"]))
    s_err = float(abs(pose["scale"] - 1.0))

    return {
        "translation_error": t_err,
        "rotation_angle_rad": r_angle,
        "scale_error": s_err,
    }


def check_known_rotation(captra: CAPTRA, points: np.ndarray) -> Dict[str, float]:
    """
    Known rotation: apply a pure rotation about a random axis.
    """
    R_true = _random_rotation_matrix(seed=123)
    t_zero = np.zeros(3, dtype=np.float64)

    ref_prev, _ = captra.estimate_object_reference(points)
    rotated_points = _apply_transform(points, R_true, t_zero, 1.0)
    ref_curr, _ = captra.estimate_object_reference(rotated_points)

    pose = captra.estimate_pose_change(ref_curr, ref_prev)

    # Compare R_est ~ R_true by measuring log of R_true^T * R_est
    R_est = pose["rotation_matrix"]
    R_rel = R_true.T @ R_est
    angle = float(np.arccos(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)))

    t_norm = float(np.linalg.norm(pose["translation"]))
    s_err = float(abs(pose["scale"] - 1.0))

    return {
        "rotation_angle_error_rad": angle,
        "translation_norm": t_norm,
        "scale_error": s_err,
    }


def check_known_scale(captra: CAPTRA, points: np.ndarray, scale_factor: float = 1.5) -> Dict[str, float]:
    """
    Known scale: apply uniform scaling and verify ratio.
    """
    R_id = np.eye(3, dtype=np.float64)
    t_zero = np.zeros(3, dtype=np.float64)

    ref_prev, _ = captra.estimate_object_reference(points)
    scaled_points = _apply_transform(points, R_id, t_zero, scale_factor)
    ref_curr, _ = captra.estimate_object_reference(scaled_points)

    pose = captra.estimate_pose_change(ref_curr, ref_prev)

    s_err = float(abs(pose["scale"] - scale_factor))
    t_norm = float(np.linalg.norm(pose["translation"]))
    r_angle = float(np.linalg.norm(pose["rotation_euler"]))

    return {
        "scale_error": s_err,
        "translation_norm": t_norm,
        "rotation_angle_rad": r_angle,
    }


def check_axes_orthonormal(axes: np.ndarray) -> float:
    """
    Orthonormality check: ||A^T A - I||_F.
    """
    I = np.eye(3, dtype=np.float64)
    err = float(np.linalg.norm(axes.T @ axes - I, ord="fro"))
    return err


def check_alignment_residual(
    prev_state: CAPTRAReferenceState,
    curr_state: CAPTRAReferenceState,
    captra: CAPTRA,
) -> float:
    """
    Alignment residual between previous and current states.

    Uses stored point samples from each state if available and measures
    the RMS residual after applying the estimated pose.
    """
    pose = captra.estimate_pose_change(curr_state, prev_state)
    R = pose["rotation_matrix"]
    t = pose["translation"]
    s = pose["scale"]

    if prev_state.points_sample is None or curr_state.points_sample is None:
        return float("nan")

    P_prev = prev_state.points_sample
    P_curr = curr_state.points_sample

    # Subsample to common size for simplicity
    n = min(P_prev.shape[0], P_curr.shape[0])
    if n == 0:
        return float("nan")
    idx_prev = np.arange(n)
    idx_curr = np.arange(n)

    P_prev_s = P_prev[idx_prev]
    P_curr_s = P_curr[idx_curr]

    P_prev_aligned = (s * (R @ P_prev_s.T)).T + t.reshape(1, 3)
    residuals = P_curr_s - P_prev_aligned
    rms = float(np.sqrt((residuals ** 2).sum(axis=1).mean()))
    return rms


def run_all_synthetic_checks(num_points: int = 500) -> Dict[str, Dict[str, float]]:
    """
    Run a suite of lightweight numerical validation checks on synthetic data.

    This does not depend on RGB or real depth; it only validates the
    geometric behavior of reference estimation and pose change.
    """
    # Create a synthetic, anisotropic cloud so PCA has a preferred orientation
    rng = np.random.default_rng(42)
    base = rng.normal(size=(num_points, 3))
    base *= np.array([0.3, 0.1, 0.05])[None, :]

    # Dummy intrinsics (only needed to construct CAPTRA; not used here)
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    captra = CAPTRA(camera_intrinsics=K)

    id_res = check_identity_behavior(captra, base)
    t_res = check_known_translation(captra, base)
    r_res = check_known_rotation(captra, base)
    s_res = check_known_scale(captra, base, scale_factor=1.5)

    ref, _ = captra.estimate_object_reference(base)
    axes_err = check_axes_orthonormal(ref.axes) if ref is not None else float("nan")

    return {
        "identity": id_res,
        "translation": t_res,
        "rotation": r_res,
        "scale": s_res,
        "axes_orthonormality": {"frobenius_error": axes_err},
    }


if __name__ == "__main__":
    # Simple CLI-like entry point for quick manual testing
    results = run_all_synthetic_checks()
    print("=== CAPTRA Synthetic Validation ===")
    for name, metrics in results.items():
        print(f"[{name}]")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

