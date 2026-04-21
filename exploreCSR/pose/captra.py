"""
Neural CAPTRA: category-level 6-DoF pose tracker using pretrained PointNet2 + NOCS networks.

All network code lives in exploreCSR/pose/captra_network/ (self-contained, no cross-repo paths).
Pretrained checkpoints are loaded from whichever rot_dir / coord_dir you pass in.

Supported NOCS categories: 1=bottle  2=bowl  3=camera  4=can  5=laptop  6=mug
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

import numpy as np

# ── add captra_network/ to sys.path so its absolute imports resolve ───────────
_CAPTRA_NET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captra_network")
if _CAPTRA_NET not in sys.path:
    sys.path.insert(0, _CAPTRA_NET)

from configs.config import get_config           # type: ignore[import]
from network.trainer import Trainer             # type: ignore[import]
from datasets.data_utils import farthest_point_sample  # type: ignore[import]

# ── constants ─────────────────────────────────────────────────────────────────

# NOCS real camera intrinsics (from nocs_data_process.py)
_NOCS_INTRINSICS = np.array(
    [[591.0, 0.0, 322.0],
     [0.0,   590.0, 244.0],
     [0.0,   0.0,   1.0]],
    dtype=np.float64,
)

# DepthAnything outputs relative depth [0, 1].
# Invert (closer objects get lower value), scale and shift to approximate meters.
_DEPTH_SCALE = 3.0
_DEPTH_SHIFT  = 0.3

CATEGORY_NAMES: Dict[int, str] = {
    1: "bottle", 2: "bowl", 3: "camera",
    4: "can",    5: "laptop", 6: "mug",
}


# ── depth utilities ────────────────────────────────────────────────────────────

def depth_norm_to_metric(depth_norm: np.ndarray) -> np.ndarray:
    """Convert DepthAnything normalized [0, 1] depth to approximate metric depth (meters)."""
    inverted = 1.0 - depth_norm          # invert: higher value = farther away
    return (inverted * _DEPTH_SCALE + _DEPTH_SHIFT).astype(np.float32)


def metric_to_uint16(depth_m: np.ndarray) -> np.ndarray:
    """Convert metric depth (meters) to uint16 millimetres for CAPTRA's point cloud builder."""
    return (depth_m * 1000).astype(np.uint16)


# ── point cloud builder ────────────────────────────────────────────────────────

def build_point_cloud(
    depth_uint16: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    num_points: int = 4096,
    device=None,
) -> Optional[np.ndarray]:
    """
    Back-project masked depth pixels to a fixed-size (num_points, 3) point cloud.

    Parameters
    ----------
    depth_uint16 : (H, W) uint16 depth in mm
    mask         : (H, W) binary mask — nonzero = object foreground
    intrinsics   : (3, 3) camera matrix K
    num_points   : target point count; pads with replacement if too few
    device       : torch device (passed to farthest_point_sample)

    Returns None if fewer than 2 valid points are found.
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    depth_m = depth_uint16.astype(np.float32) / 1000.0

    h, w = depth_m.shape
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth_m
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy

    pts = np.stack([X, Y, Z], axis=-1)[mask > 0]  # (N, 3)
    pts = pts[pts[:, 2] > 0.01]                    # remove zero / near-zero depth

    if len(pts) < 2:
        return None

    if len(pts) >= num_points:
        idx = farthest_point_sample(pts, num_points, device)
        pts = pts[idx]
    else:
        idx = np.random.choice(len(pts), num_points, replace=True)
        pts = pts[idx]

    return pts.astype(np.float32)


# ── CAPTRA model loader ────────────────────────────────────────────────────────

def _load_captra_model(rot_dir: str, coord_dir: str, category: int, device) -> "Trainer":
    """Load CAPTRA CoordinateNet + RotationNet from checkpoint directories."""
    import types
    args = types.SimpleNamespace(
        config         = "config_track.yml",
        obj_config     = "obj_info_nocs.yml",
        obj_category   = str(category),
        experiment_dir = rot_dir,
        resume_epoch   = -1,
        num_workers    = 0,
        save           = False,
        eval_train     = False,
        no_eval        = True,
        num_iter       = 1,
        mode_name      = "test",
    )
    cfg = get_config(args, save=False)
    cfg["coord_exp"] = {"dir": coord_dir, "resume_epoch": -1}
    cfg["device"]    = device
    cfg["init_frame"]["gt"] = True   # clean pose init on first frame

    trainer = Trainer(cfg, logger=None)
    trainer.resume()                 # load pretrained weights
    return trainer


# ── single-frame inference ─────────────────────────────────────────────────────

def _estimate_pose_single(
    model: "Trainer",
    point_cloud: np.ndarray,
    prev_pose: Optional[Dict[str, Any]],
    device,
    prev_point_cloud: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Run one CAPTRA tracking step.

    CAPTRA takes a two-frame sequence: [init_frame, current_frame].
    prev_point_cloud is used as frame_0000 geometry when available so the
    network can detect inter-frame change; falls back to current cloud on
    the first frame.

    Returns dict with rotation (3,3), translation (3,), scale (float).
    """
    import torch
    N   = point_cloud.shape[0]
    pts = torch.from_numpy(point_cloud).float().to(device)    # (N, 3)
    pts_mean  = pts.mean(0, keepdim=True)                     # (1, 3)
    pts_t     = (pts - pts_mean).T.unsqueeze(0)               # (1, 3, N)
    pts_mean_t = pts_mean.T.unsqueeze(0)                      # (1, 3, 1)

    if prev_pose is None:
        R = np.eye(3, dtype=np.float32)[np.newaxis]
        t = pts_mean.cpu().numpy().reshape(3, 1).astype(np.float32)[np.newaxis]
        s = np.array([0.2], dtype=np.float32)
    else:
        R = prev_pose["rotation"].astype(np.float32)[np.newaxis]
        t = prev_pose["translation"].reshape(3, 1).astype(np.float32)[np.newaxis]
        s = np.array([prev_pose["scale"]], dtype=np.float32)

    pose_list = [{"rotation": R, "translation": t, "scale": s}]
    labels    = torch.zeros(1, N, dtype=torch.long, device=device)
    nocs      = torch.zeros(1, 3, N, dtype=torch.float32, device=device)

    # Build the init-frame tensor from the previous point cloud when available.
    if prev_point_cloud is not None:
        prev_pts = torch.from_numpy(prev_point_cloud).float().to(device)
        prev_mean = prev_pts.mean(0, keepdim=True)
        prev_pts_t = (prev_pts - prev_mean).T.unsqueeze(0)
        prev_mean_t = prev_mean.T.unsqueeze(0)
    else:
        prev_pts_t  = pts_t
        prev_mean_t = pts_mean_t

    def _init_frame() -> dict:
        return {
            "meta": {
                "nocs2camera": pose_list,
                "points_mean": prev_mean_t,
                "path": ["synthetic/obj/0/frame_0000.npz"],
            },
            "labels": labels,
            "points": prev_pts_t,
            "nocs":   nocs,
        }

    def _curr_frame() -> dict:
        return {
            "meta": {
                "nocs2camera": pose_list,
                "points_mean": pts_mean_t,
                "path": ["synthetic/obj/0/frame_0001.npz"],
            },
            "labels": labels,
            "points": pts_t,
            "nocs":   nocs,
        }

    model.test([_init_frame(), _curr_frame()], no_eval=True)

    pred = model.model.pred_dict["poses"][1]   # refined current frame
    return {
        "rotation":    pred["rotation"][0, 0].cpu().numpy(),
        "translation": pred["translation"][0, 0].cpu().numpy().reshape(3),
        "scale":       float(pred["scale"][0, 0].cpu()),
    }


# ── Euler angle helper ─────────────────────────────────────────────────────────

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


# ── main class ────────────────────────────────────────────────────────────────

class CAPTRA:
    """
    Frame-by-frame neural 6-DoF object pose tracker.

    Wraps pretrained PointNet2 + NOCS coordinate networks.  Returns the same
    pose dict format that FeaturePoseTracker.add_frame() consumes.

    Usage
    -----
    >>> import torch
    >>> captra = CAPTRA(
    ...     rot_dir   = "/path/to/runs/6_mug_rot",
    ...     coord_dir = "/path/to/runs/6_mug_coord",
    ...     category  = 6,
    ...     device    = torch.device("cuda"),
    ... )
    >>> for rgb, depth_norm, mask in frames:
    ...     out = captra.forward(rgb, depth_norm, mask)
    ...     tracker.add_frame(idx, features, out, mask)
    """

    def __init__(
        self,
        rot_dir: str,
        coord_dir: str,
        category: int,
        device,
        intrinsics: Optional[np.ndarray] = None,
        num_points: int = 4096,
    ) -> None:
        """
        Parameters
        ----------
        rot_dir    : path to the rotation net checkpoint directory
        coord_dir  : path to the coordinate net checkpoint directory
        category   : NOCS category index  1=bottle 2=bowl 3=camera 4=can 5=laptop 6=mug
        device     : torch.device
        intrinsics : (3, 3) camera matrix K.
                     Defaults to the NOCS real camera [[591,0,322],[0,590,244],[0,0,1]].
        num_points : number of points per frame point cloud (default 4096)
        """
        if category not in CATEGORY_NAMES:
            raise ValueError(
                f"category must be one of {list(CATEGORY_NAMES)} — got {category}"
            )
        self.device = device
        self.num_points = int(num_points)
        self.intrinsics = (
            np.asarray(intrinsics, dtype=np.float64)
            if intrinsics is not None
            else _NOCS_INTRINSICS.copy()
        )
        self._prev_pose:     Optional[Dict[str, Any]] = None
        self._prev_pcl:      Optional[np.ndarray]     = None
        self._prev_centroid:  Optional[np.ndarray] = None
        self._prev_pcl_scale: Optional[float]     = None
        self._frozen_count:   int                  = 0
        self._scale_frozen_count: int              = 0

        print(f"[CAPTRA] Loading {CATEGORY_NAMES[category]} (category {category}) on {device} …")
        self._model = _load_captra_model(rot_dir, coord_dir, category, device)
        print("[CAPTRA] Model ready.")

    def reset(self) -> None:
        """Clear tracked state — call this between independent sequences."""
        self._prev_pose       = None
        self._prev_pcl        = None
        self._prev_centroid   = None
        self._prev_pcl_scale  = None
        self._frozen_count    = 0
        self._scale_frozen_count = 0

    def forward(
        self,
        rgb: np.ndarray,
        depth_norm: np.ndarray,
        mask: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Estimate pose for the current frame.

        Parameters
        ----------
        rgb        : (H, W, 3) uint8 RGB image (kept for interface parity; not used by CAPTRA)
        depth_norm : (H, W) float normalized depth in [0, 1] from DepthAnything
        mask       : (H, W) binary object mask (nonzero = foreground)

        Returns
        -------
        dict with keys:
          translation     (3,)   object translation in approximate metric space
          rotation_matrix (3, 3) rotation from canonical to camera frame
          rotation_euler  (3,)   XYZ Euler angles in radians
          pose_vector     (7,)   [tx, ty, tz, rx, ry, rz, scale]
          scale           float  predicted object scale
          valid           bool
          message         str
        """
        depth_m      = depth_norm_to_metric(depth_norm)
        depth_uint16 = metric_to_uint16(depth_m)

        pcl = build_point_cloud(
            depth_uint16,
            mask,
            self.intrinsics,
            num_points=self.num_points,
            device=self.device,
        )
        if pcl is None:
            return self._invalid("No valid depth points inside mask", point_cloud=None)

        try:
            pose = _estimate_pose_single(self._model, pcl, self._prev_pose, self.device, self._prev_pcl)
        except Exception as exc:
            return self._invalid(f"CAPTRA inference error: {exc}", point_cloud=pcl)

        centroid = pcl.mean(0).astype(np.float64)   # camera-frame geometric centre

        # Centroid-vs-CAPTRA comparison: if the point cloud centroid moved but
        # CAPTRA's translation didn't, the network is frozen on stale context.
        # Override just the translation with the centroid; keep rotation/scale so
        # we don't destabilise CAPTRA's pose context for the next frame.
        msg = "CAPTRA OK"
        t   = np.asarray(pose["translation"], dtype=np.float64)

        if self._prev_pose is not None and self._prev_centroid is not None:
            centroid_delta = float(np.linalg.norm(centroid - self._prev_centroid))
            captra_delta   = float(np.linalg.norm(
                t - np.asarray(self._prev_pose["translation"], dtype=np.float64)
            ))
            # Centroid moved but CAPTRA didn't → frozen
            if centroid_delta > 0.01 and captra_delta < centroid_delta * 0.25:
                self._frozen_count += 1
                if self._frozen_count >= 3:
                    t   = centroid
                    msg = f"centroid fallback (frozen×{self._frozen_count})"
                    print(f"[CAPTRA] {msg} — centroid Δ={centroid_delta:.4f}, CAPTRA Δ={captra_delta:.4f}")
            else:
                self._frozen_count = 0
        else:
            self._frozen_count = 0

        self._prev_centroid = centroid

        # Scale fallback: mean point distance from centroid is a direct geometric
        # proxy for object size in camera space.  If this changes but CAPTRA's
        # predicted scale doesn't, CAPTRA's scale is frozen.
        pcl_scale = float(np.mean(np.linalg.norm(pcl.astype(np.float64) - centroid, axis=1)))
        s = float(pose["scale"])

        if self._prev_pcl_scale is not None and self._prev_pose is not None:
            pcl_scale_delta    = abs(pcl_scale - self._prev_pcl_scale)
            captra_scale_delta = abs(s - float(self._prev_pose["scale"]))
            if pcl_scale_delta > 0.005 and captra_scale_delta < pcl_scale_delta * 0.25:
                self._scale_frozen_count += 1
                if self._scale_frozen_count >= 3:
                    s   = pcl_scale
                    msg = msg.replace("CAPTRA OK", "").strip(" ,") or "OK"
                    msg = f"{msg}, scale fallback (frozen×{self._scale_frozen_count})"
                    print(f"[CAPTRA] scale fallback — PCL scale Δ={pcl_scale_delta:.4f}, CAPTRA Δ={captra_scale_delta:.4f}")
            else:
                self._scale_frozen_count = 0
        else:
            self._scale_frozen_count = 0

        self._prev_pcl_scale = pcl_scale

        # Store pose with corrected translation and scale so next frame's prior is consistent.
        pose_stored = dict(pose)
        pose_stored["translation"] = t
        pose_stored["scale"] = s
        self._prev_pose = pose_stored
        self._prev_pcl  = pcl

        R     = np.asarray(pose["rotation"], dtype=np.float64)
        euler = _rotation_matrix_to_euler_xyz(R)

        return {
            "translation":     t,
            "rotation_matrix": R,
            "rotation_euler":  euler,
            "pose_vector":     np.concatenate([t, euler, [s]]),
            "scale":           s,
            "valid":           True,
            "message":         msg,
            "point_cloud":     pcl,
        }

    @staticmethod
    def _invalid(msg: str, point_cloud=None) -> Dict[str, Any]:
        return {
            "translation":     np.zeros(3, dtype=np.float64),
            "rotation_matrix": np.eye(3,  dtype=np.float64),
            "rotation_euler":  np.zeros(3, dtype=np.float64),
            "pose_vector":     np.zeros(7, dtype=np.float64),
            "scale":           1.0,
            "valid":           False,
            "message":         msg,
            "point_cloud":     point_cloud,
        }
