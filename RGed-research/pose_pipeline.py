"""
pose_pipeline.py

Full pipeline: RGB frame -> DepthAnything -> DINOSegmenter -> CAPTRA -> 6DoF pose

Usage:
    python pose_pipeline.py --frames_dir path/to/frames --category 1 --prompt "bottle"
"""

import sys
import os
import argparse
import pickle
import cv2
import numpy as np
import torch

# ── path setup ────────────────────────────────────────────────────────────────
CAPTRA_ROOT = os.path.join(os.path.dirname(__file__), "captra", "CAPTRA")
sys.path.insert(0, CAPTRA_ROOT)

from depth_anything import DepthAnything
from dino_segmenter import DINOSegmenter

# CAPTRA imports
from captra.CAPTRA.configs.config import get_config
from captra.CAPTRA.network.trainer import Trainer
from captra.CAPTRA.datasets.nocs_data.nocs_data_process import (
    crop_ball_from_depth_image,
    base_generate_data,
    nocs_real_cam_intrinsics,
)
from captra.CAPTRA.datasets.data_utils import farthest_point_sample

# ── constants ─────────────────────────────────────────────────────────────────

# NOCS real camera intrinsics (change if using your own camera)
INTRINSICS = nocs_real_cam_intrinsics  # [[591, 0, 322], [0, 590, 244], [0,0,1]]

# DepthAnything outputs relative depth [0,1].
# This scale converts it to approximate metric depth in meters.
# Tune this for your camera/scene. Typical indoor range: 0.5–4m
DEPTH_SCALE = 3.0   # multiply normalized depth by this to get ~meters
DEPTH_SHIFT = 0.3   # add this to prevent zero-depth near objects

CATEGORY_NAMES = {
    1: "bottle", 2: "bowl", 3: "camera",
    4: "can",    5: "laptop", 6: "mug"
}

# ── depth conversion ───────────────────────────────────────────────────────────

def depth_norm_to_metric(depth_norm: np.ndarray) -> np.ndarray:
    """
    Convert DepthAnything normalized [0,1] relative depth to approximate
    metric depth in meters.

    DepthAnything uses inverse depth (higher value = closer).
    We invert it so higher value = farther, then scale to meters.
    """
    inverted = 1.0 - depth_norm                      # closer objects now have lower values
    metric   = inverted * DEPTH_SCALE + DEPTH_SHIFT  # shift away from zero
    return metric.astype(np.float32)


def metric_to_uint16(depth_m: np.ndarray) -> np.ndarray:
    """Convert metric float depth (meters) to uint16 millimetres for CAPTRA."""
    return (depth_m * 1000).astype(np.uint16)


# ── point cloud builder ────────────────────────────────────────────────────────

def build_point_cloud(
    depth_uint16: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    num_points: int = 4096,
    device=None,
) -> np.ndarray:
    """
    Back-project masked depth pixels to a (num_points, 3) point cloud.

    depth_uint16 : (H, W) uint16 depth in mm
    mask         : (H, W) uint8 binary — 1 = object
    intrinsics   : 3x3 camera matrix
    """
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    depth_m = depth_uint16.astype(np.float32) / 1000.0

    h, w = depth_m.shape
    u = np.arange(w)
    v = np.arange(h)
    uu, vv = np.meshgrid(u, v)

    Z = depth_m
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy

    xyz = np.stack([X, Y, Z], axis=-1)          # (H, W, 3)
    pts = xyz[mask > 0]                          # (N, 3) masked points

    valid = pts[:, 2] > 0.01                     # remove zero-depth
    pts = pts[valid]

    if len(pts) == 0:
        return None

    # subsample to fixed size via farthest point sampling
    if len(pts) >= num_points:
        idx = farthest_point_sample(pts, num_points, device)
        pts = pts[idx]
    else:
        idx = np.random.choice(len(pts), num_points, replace=True)
        pts = pts[idx]

    return pts.astype(np.float32)


# ── CAPTRA model loader ────────────────────────────────────────────────────────

def load_captra(rot_dir: str, coord_dir: str, category: int, device) -> Trainer:
    """Load CAPTRA CoordinateNet + RotationNet from checkpoint dirs."""
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
    cfg  = get_config(args, save=False)
    cfg["coord_exp"]           = {"dir": coord_dir, "resume_epoch": -1}
    cfg["device"]              = device
    cfg["init_frame"]["gt"]    = True   # use clean pose init (no noise on first frame)

    trainer = Trainer(cfg, logger=None)
    trainer.resume()                    # load checkpoint weights — required before inference
    return trainer


# ── single-frame pose ──────────────────────────────────────────────────────────

def estimate_pose_single(
    captra: Trainer,
    point_cloud: np.ndarray,
    prev_pose: dict | None,
    device,
) -> dict:
    """
    Run one CAPTRA tracking step via Trainer.test().

    CAPTRA's EvalTrackModel takes a two-frame sequence: [init_frame, current_frame].
    The init_frame provides the starting pose; the network refines it on current_frame.

    Returns dict with keys: rotation (3,3), translation (3,), scale (float)
    """
    N = point_cloud.shape[0]
    pts       = torch.from_numpy(point_cloud).float().to(device)  # (N, 3)
    pts_mean  = pts.mean(0, keepdim=True)                         # (1, 3)
    pts_t     = (pts - pts_mean).T.unsqueeze(0)                   # (1, 3, N) — centered
    pts_mean_t = pts_mean.T.unsqueeze(0)                          # (1, 3, 1)

    if prev_pose is None:
        R = np.eye(3, dtype=np.float32)[np.newaxis]                         # (1, 3, 3)
        t = pts_mean.cpu().numpy().reshape(3, 1).astype(np.float32)[np.newaxis]  # (1, 3, 1)
        s = np.array([0.2], dtype=np.float32)                               # (1,)
    else:
        R = prev_pose["rotation"].astype(np.float32)[np.newaxis]
        t = prev_pose["translation"].reshape(3, 1).astype(np.float32)[np.newaxis]
        s = np.array([prev_pose["scale"]], dtype=np.float32)

    # nocs2camera: list of one pose-dict (one part) with batched numpy arrays
    pose_list = [{"rotation": R, "translation": t, "scale": s}]

    labels = torch.zeros(1, N, dtype=torch.long, device=device)
    nocs   = torch.zeros(1, 3, N, dtype=torch.float32, device=device)

    def _frame(name: str) -> dict:
        return {
            "meta": {
                "nocs2camera": pose_list,
                "points_mean": pts_mean_t,
                "path": [f"synthetic/obj/0/{name}.npz"],
            },
            "labels": labels,
            "points": pts_t,  # (1, 3, N) — CoordNet / PartCanonNet expect [B, 3, N]
            "nocs":   nocs,
        }

    # Trainer.test sets model to eval, calls set_data then model.test (→ forward)
    captra.test([_frame("frame_0000"), _frame("frame_0001")], no_eval=True)

    # pred_dict['poses'][0] = init, [1] = refined current frame
    pred_part = captra.model.pred_dict["poses"][1]   # shapes: (B, P, ...)
    pose = {
        "rotation":    pred_part["rotation"][0, 0].cpu().numpy(),             # (3, 3)
        "translation": pred_part["translation"][0, 0].cpu().numpy().reshape(3),  # (3,)
        "scale":       float(pred_part["scale"][0, 0].cpu()),
    }
    return pose


# ── frame pre-processing ───────────────────────────────────────────────────────

def preprocess_frame(
    frame_bgr: np.ndarray,
    depth_model: DepthAnything,
    dino_model: DINOSegmenter,
    prompt: str,
    use_depth_fusion: bool = True,
) -> tuple[np.ndarray | None, np.ndarray]:
    """
    Run DepthAnything + DINO on one BGR frame.

    Returns
    -------
    point_cloud : (4096, 3) float32 or None if no valid points
    mask        : (H, W) uint8 for visualization
    """
    H, W = frame_bgr.shape[:2]

    # 1. Depth estimation
    tensor_d, orig_w, orig_h = depth_model.frame_to_tensor(frame_bgr)
    raw_depth   = depth_model.predict(tensor_d)
    depth_norm  = depth_model.postprocess(raw_depth, orig_w, orig_h)   # [0,1] H×W
    depth_m     = depth_norm_to_metric(depth_norm)                      # meters H×W
    depth_uint16 = metric_to_uint16(depth_m)                            # mm uint16

    # 2. DINO segmentation
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb_f = rgb.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(
        rgb_f.transpose(2, 0, 1)
    ).unsqueeze(0).to(dino_model.device)

    mask = dino_model.segment_from_prompt(
        image_tensor   = img_tensor,
        prompt         = prompt,
        output_size    = (H, W),
        depth_map      = depth_norm if use_depth_fusion else None,
        depth_tolerance= 0.15,
    )

    # 3. Build point cloud
    pcl = build_point_cloud(
        depth_uint16, mask, INTRINSICS, num_points=4096,
        device=dino_model.device,
    )

    return pcl, mask, depth_norm


# ── main pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(
    frames_dir: str,
    depth_checkpoint: str,
    rot_dir: str,
    coord_dir: str,
    category: int,
    prompt: str,
    output_pkl: str,
    max_frames: int | None = None,
) -> list[dict]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # load models
    print("Loading DepthAnything...")
    depth_model = DepthAnything(checkpoint_path=depth_checkpoint)

    print("Loading DINO...")
    dino_model  = DINOSegmenter()

    print("Loading CAPTRA...")
    captra      = load_captra(rot_dir, coord_dir, category, device)

    # collect frames
    exts   = (".png", ".jpg", ".jpeg")
    frames = sorted([
        f for f in os.listdir(frames_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])
    if max_frames:
        frames = frames[:max_frames]

    print(f"Tracking {len(frames)} frames for category {category} ({CATEGORY_NAMES.get(category)})")

    results   = []
    prev_pose = None

    for i, fname in enumerate(frames):
        frame_path = os.path.join(frames_dir, fname)
        frame_bgr  = cv2.imread(frame_path)
        if frame_bgr is None:
            print(f"  [skip] could not read {fname}")
            continue

        # preprocess
        pcl, mask, depth_norm = preprocess_frame(
            frame_bgr, depth_model, dino_model, prompt
        )

        if pcl is None:
            print(f"  [skip] frame {i} — no valid points in mask")
            continue

        # CAPTRA tracking
        try:
            pose = estimate_pose_single(captra, pcl, prev_pose, device)
        except Exception as e:
            print(f"  [skip] frame {i} — CAPTRA error: {e}")
            continue

        prev_pose = pose

        results.append({
            "frame":       i,
            "file":        fname,
            "rotation":    pose["rotation"],
            "translation": pose["translation"],
            "scale":       pose["scale"],
        })

        print(
            f"  frame {i:04d} | "
            f"t=[{pose['translation'][0]:.3f}, "
            f"{pose['translation'][1]:.3f}, "
            f"{pose['translation'][2]:.3f}] | "
            f"s={pose['scale']:.3f}"
        )

    # save
    os.makedirs(os.path.dirname(os.path.abspath(output_pkl)), exist_ok=True)
    with open(output_pkl, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved {len(results)} poses → {output_pkl}")

    return results


# ── pose matrix helper ─────────────────────────────────────────────────────────

def pose_to_matrix(pose: dict) -> np.ndarray:
    """Build a 4x4 transform from a pose dict."""
    T = np.eye(4)
    T[:3, :3] = pose["scale"] * pose["rotation"]
    T[:3,  3] = pose["translation"]
    return T




def run_pipeline_video(
    video_path: str,
    depth_checkpoint: str,
    rot_dir: str,
    coord_dir: str,
    category: int,
    prompt: str,
    output_pkl: str,
    max_frames: int | None = None,
    frame_skip: int = 1,        # process every Nth frame
    display: bool = True,       # show live preview window
) -> list[dict]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # load models
    print("Loading DepthAnything...")
    depth_model = DepthAnything(checkpoint_path=depth_checkpoint)
    print("Loading DINO...")
    dino_model  = DINOSegmenter()
    print("Loading CAPTRA...")
    captra      = load_captra(rot_dir, coord_dir, category, device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames @ {fps:.1f} fps")
    print(f"Tracking every {frame_skip} frame(s) for '{CATEGORY_NAMES.get(category)}'")

    results   = []
    prev_pose = None
    frame_idx = 0
    processed = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # skip frames
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        if max_frames and processed >= max_frames:
            break

        # preprocess
        pcl, mask, depth_norm = preprocess_frame(
            frame_bgr, depth_model, dino_model, prompt
        )

        if pcl is None:
            print(f"  [skip] frame {frame_idx} — no valid points")
            frame_idx += 1
            continue

        # CAPTRA tracking
        try:
            pose = estimate_pose_single(captra, pcl, prev_pose, device)
        except Exception as e:
            print(f"  [skip] frame {frame_idx} — {e}")
            frame_idx += 1
            continue

        prev_pose = pose

        results.append({
            "frame":       frame_idx,
            "timestamp":   frame_idx / fps,
            "rotation":    pose["rotation"],
            "translation": pose["translation"],
            "scale":       pose["scale"],
        })

        print(
            f"  frame {frame_idx:05d} | "
            f"t(s)={frame_idx/fps:.2f} | "
            f"xyz=[{pose['translation'][0]:.3f}, "
            f"{pose['translation'][1]:.3f}, "
            f"{pose['translation'][2]:.3f}]"
        )

        # live preview
        if display:
            # draw mask overlay on frame
            preview = frame_bgr.copy()
            colored = np.zeros_like(frame_bgr)
            colored[mask > 0] = (0, 165, 255)   # orange for object
            preview = cv2.addWeighted(preview, 0.7, colored, 0.3, 0)

            # draw translation as text
            t = pose["translation"]
            cv2.putText(preview,
                f"xyz: [{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}]",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(preview,
                f"scale: {pose['scale']:.3f}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(preview,
                f"frame: {frame_idx}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("CAPTRA Pose Tracking", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopped by user.")
                break

        frame_idx += 1
        processed += 1

    cap.release()
    if display:
        cv2.destroyAllWindows()

    # save
    os.makedirs(os.path.dirname(os.path.abspath(output_pkl)), exist_ok=True)
    with open(output_pkl, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved {len(results)} poses → {output_pkl}")

    return results

# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video",             default=None,   help="Path to input video file")
    p.add_argument("--frames_dir",        default=None,   help="Folder of input RGB frames")
    p.add_argument("--depth_checkpoint",  required=True)
    p.add_argument("--rot_dir",           required=True)
    p.add_argument("--coord_dir",         required=True)
    p.add_argument("--category",          type=int, default=1)
    p.add_argument("--prompt",            default="")
    p.add_argument("--output",            default="results/poses.pkl")
    p.add_argument("--max_frames",        type=int, default=None)
    p.add_argument("--frame_skip",        type=int, default=1)
    p.add_argument("--no_display",        action="store_true")
    args = p.parse_args()

    if args.video:
        run_pipeline_video(
            video_path       = args.video,
            depth_checkpoint = args.depth_checkpoint,
            rot_dir          = args.rot_dir,
            coord_dir        = args.coord_dir,
            category         = args.category,
            prompt           = args.prompt,
            output_pkl       = args.output,
            max_frames       = args.max_frames,
            frame_skip       = args.frame_skip,
            display          = not args.no_display,
        )
    elif args.frames_dir:
        run_pipeline(
            frames_dir       = args.frames_dir,
            depth_checkpoint = args.depth_checkpoint,
            rot_dir          = args.rot_dir,
            coord_dir        = args.coord_dir,
            category         = args.category,
            prompt           = args.prompt,
            output_pkl       = args.output,
            max_frames       = args.max_frames,
        )
    else:
        print("Error: provide either --video or --frames_dir")
