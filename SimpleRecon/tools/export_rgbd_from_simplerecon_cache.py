#!/usr/bin/env python3
"""
Export RGB-D from SimpleRecon depth cache.

Reads pickle files produced by SimpleRecon test.py with --cache_depths
(results_path/depths/<scan_id>/*.pickle) and exports:
  - previews/: human-viewable depth visualizations (grayscale or colormap PNG)
  - rgbd/: rgb/, depth/, intrinsics/, frames.json for downstream 3D work.

SimpleRecon cache pickle keys (from utils/generic_utils.py cache_model_outputs):
  - depth_pred_s0_b1hw, depth_pred_s1_b1hw, ... (multi-scale; s0 = highest res)
  - log_depth_pred_s*_b1hw, lowest_cost_bhw, overall_mask_bhw
  - K_full_depth_b44: 4x4 intrinsics for full-res depth
  - K_s0_b44: 4x4 intrinsics at scale 0 (model resolution)
  - frame_id: str (e.g. "     0" or "frame-000261")
  - src_ids: list of source frame id strings
  Pose (cam_T_world / world_T_cam) is NOT stored in the cache; include in
  frames.json only if we add it later or find it in pickle.
Depth values in the cache are in METERS (linear scale; cost volume uses 0.25–5 m).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

# Torch only for loading pickle (SimpleRecon cache contains torch tensors)
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# -----------------------------------------------------------------------------
# Pickle loading (SimpleRecon uses pickle.dump with torch tensors)
# -----------------------------------------------------------------------------

import pickle as _pickle


def load_pickle(path: Path) -> dict[str, Any]:
    """Load a SimpleRecon cache pickle; convert tensors to numpy if needed."""
    with open(path, "rb") as f:
        raw = _pickle.load(f)
    out = {}
    for k, v in raw.items():
        if v is None:
            out[k] = None
        elif HAS_TORCH and isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().numpy()
        elif isinstance(v, np.ndarray):
            out[k] = v
        elif isinstance(v, (list, str, int, float, bool)):
            out[k] = v
        else:
            out[k] = v
    return out


# -----------------------------------------------------------------------------
# Depth and intrinsics extraction
# -----------------------------------------------------------------------------

DEPTH_KEYS_PREFERRED = ["depth_pred_s0_b1hw", "depth_pred_s1_b1hw", "depth_pred_s2_b1hw", "depth_pred_s3_b1hw"]


def get_depth_from_pickle(data: dict) -> tuple[np.ndarray, int, int]:
    """Extract best available depth map (B,1,H,W) or (1,H,W); return (depth_HW, H, W)."""
    for key in DEPTH_KEYS_PREFERRED:
        if key in data and data[key] is not None:
            d = np.asarray(data[key], dtype=np.float64)
            while d.ndim > 2:
                d = d[0]
            # (H, W) float, meters
            return d, int(d.shape[0]), int(d.shape[1])
    raise KeyError(
        f"No depth key found. Keys present: {sorted(k for k in data if data[k] is not None)}"
    )


def get_K_from_pickle(data: dict) -> np.ndarray:
    """Get 3x3 intrinsics matrix. Prefer K_full_depth_b44, else K_s0_b44."""
    for key in ["K_full_depth_b44", "K_s0_b44"]:
        if key in data and data[key] is not None:
            K = data[key]
            if K.ndim == 3:
                K = K[0]
            if K.shape[0] == 4:
                return K[:3, :3]
            return K[:3, :3]
    raise KeyError(
        f"No intrinsics key found. Keys present: {sorted(k for k in data if data[k] is not None)}"
    )


def K_to_fxfycxcy(K: np.ndarray) -> tuple[float, float, float, float]:
    """Extract fx, fy, cx, cy from 3x3 K."""
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    return fx, fy, cx, cy


def scale_intrinsics(fx: float, fy: float, cx: float, cy: float, scale_x: float, scale_y: float) -> tuple[float, float, float, float]:
    """Scale intrinsics when resizing depth to RGB resolution."""
    return fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y


# -----------------------------------------------------------------------------
# RGB frame lookup
# -----------------------------------------------------------------------------

def frame_id_to_stem(frame_id: str, index: int) -> str:
    """Normalize frame_id to a filename stem. Prefer numeric index for stable names."""
    s = (frame_id or "").strip()
    if re.match(r"^\d+$", s):
        return f"frame_{int(s):06d}"
    if s:
        # Use as stem but sanitize
        safe = re.sub(r"[^\w\-.]", "_", s)
        return safe
    return f"frame_{index:06d}"


def find_rgb_path(rgb_dir: Path, stem: str, frame_id: str, index: int, exts: tuple[str, ...]) -> Optional[Path]:
    """Find RGB image path by stem or frame_id. exts = ('.jpg', '.png', ...)."""
    candidates = []
    # Try stem-based names
    for ext in exts:
        candidates.append(rgb_dir / f"{stem}{ext}")
    # Try frame_id as filename (with/without strip)
    fid = (frame_id or "").strip()
    for ext in exts:
        candidates.append(rgb_dir / f"{fid}{ext}")
    candidates.append(rgb_dir / f"frame_{index:06d}.jpg")
    candidates.append(rgb_dir / f"frame_{index:06d}.png")
    for p in candidates:
        if p.is_file():
            return p
    return None


def get_rgb_exts(rgb_ext: str) -> tuple[str, ...]:
    """Return tuple of extensions to try (e.g. .jpg, .jpeg, .png)."""
    ext = (rgb_ext or "").strip().lower()
    if not ext:
        return (".jpg", ".jpeg", ".png")
    if not ext.startswith("."):
        ext = "." + ext
    if ext == ".jpg":
        return (".jpg", ".jpeg", ".png")
    return (ext, ".jpg", ".jpeg", ".png")


# -----------------------------------------------------------------------------
# Resize depth and scale intrinsics
# -----------------------------------------------------------------------------

def resize_depth_nearest(depth: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize depth with nearest-neighbor (preserves values)."""
    from PIL import Image as PILImage
    h, w = depth.shape
    if (h, w) == (target_h, target_w):
        return depth
    img = PILImage.fromarray(depth.astype(np.float32))
    # PIL doesn't support float resize with NEAREST nicely; use numpy
    y = np.linspace(0, h - 1, target_h).astype(np.int32)
    x = np.linspace(0, w - 1, target_w).astype(np.int32)
    return depth[np.ix_(y, x)]


# -----------------------------------------------------------------------------
# Output writers
# -----------------------------------------------------------------------------

def write_depth_png16mm(path: Path, depth_m: np.ndarray) -> None:
    """Write depth in millimeters as 16-bit PNG; clamp to [0, 65535]."""
    depth_mm = depth_m * 1000.0
    depth_mm = np.nan_to_num(depth_mm, nan=0.0, posinf=0.0, neginf=0.0)
    depth_mm = np.clip(depth_mm, 0, 65535)
    arr = np.round(depth_mm).astype(np.uint16)
    try:
        import cv2
        cv2.imwrite(str(path), arr)
    except ImportError:
        raise RuntimeError(
            "16-bit PNG requires opencv (pip install opencv-python). "
            "Use --depth_format npy32 for float32 .npy in meters without opencv."
        )


def write_depth_npy32(path: Path, depth_m: np.ndarray) -> None:
    """Write depth in meters as float32 .npy."""
    arr = np.asarray(depth_m, dtype=np.float32)
    np.save(path, arr)


def write_preview_grayscale(path: Path, depth: np.ndarray, pct_low: float = 2.0, pct_high: float = 98.0) -> None:
    """Normalize depth by percentiles and save as 8-bit grayscale PNG."""
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        vmin, vmax = 0.0, 1.0
    else:
        vmin = np.percentile(depth[valid], pct_low)
        vmax = np.percentile(depth[valid], pct_high)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    norm = (depth - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0, 1)
    norm = np.nan_to_num(norm, nan=0.0)
    arr = (norm * 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def write_preview_colormap(path: Path, depth: np.ndarray, pct_low: float = 2.0, pct_high: float = 98.0) -> None:
    """Normalize depth and apply a simple viridis-like colormap; save as RGB PNG."""
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        vmin, vmax = 0.0, 1.0
    else:
        vmin = np.percentile(depth[valid], pct_low)
        vmax = np.percentile(depth[valid], pct_high)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    norm = (depth - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0, 1)
    norm = np.nan_to_num(norm, nan=0.0)
    # Simple colormap: R, G, B from norm
    r = (norm * 255).astype(np.uint8)
    g = (np.clip(1.2 * norm - 0.1, 0, 1) * 255).astype(np.uint8)
    b = (np.clip(1.5 * norm - 0.5, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    Image.fromarray(rgb, mode="RGB").save(path)


def write_intrinsics_json(path: Path, fx: float, fy: float, cx: float, cy: float, width: int, height: int) -> None:
    """Write intrinsics as JSON."""
    obj = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": width, "height": height}
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# -----------------------------------------------------------------------------
# Main export logic
# -----------------------------------------------------------------------------

def collect_pickle_files(cache_dir: Path) -> list[Path]:
    """Find all .pickle files under cache_dir recursively."""
    return sorted(cache_dir.rglob("*.pickle"))


def run_export(
    cache_dir: Path,
    rgb_dir: Path,
    out_dir: Path,
    rgb_ext: str = "",
    depth_format: str = "png16mm",
    colormap: bool = False,
    max_frames: Optional[int] = None,
    dry_run: bool = False,
) -> int:
    """Export cache to previews/ and rgbd/. Returns number of frames exported."""
    if not HAS_TORCH:
        print("Warning: PyTorch not installed; pickle may contain tensors. Install torch for full support.", file=sys.stderr)
    pickles = collect_pickle_files(cache_dir)
    if not pickles:
        print(f"No .pickle files under {cache_dir}", file=sys.stderr)
        return 0
    if max_frames is not None:
        pickles = pickles[: max_frames]
    exts = get_rgb_exts(rgb_ext)
    preview_dir = out_dir / "previews"
    rgbd_dir = out_dir / "rgbd"
    rgb_out = rgbd_dir / "rgb"
    depth_out = rgbd_dir / "depth"
    intrinsics_out = rgbd_dir / "intrinsics"
    if not dry_run:
        preview_dir.mkdir(parents=True, exist_ok=True)
        rgb_out.mkdir(parents=True, exist_ok=True)
        depth_out.mkdir(parents=True, exist_ok=True)
        intrinsics_out.mkdir(parents=True, exist_ok=True)

    frames_manifest = []
    exported = 0
    for index, pkl_path in enumerate(pickles):
        try:
            data = load_pickle(pkl_path)
        except Exception as e:
            print(f"Skip {pkl_path}: failed to load: {e}", file=sys.stderr)
            continue
        try:
            depth, depth_h, depth_w = get_depth_from_pickle(data)
        except KeyError as e:
            print(f"Skip {pkl_path}: {e}", file=sys.stderr)
            continue
        try:
            K = get_K_from_pickle(data)
        except KeyError as e:
            print(f"Skip {pkl_path}: {e}", file=sys.stderr)
            continue
        frame_id = data.get("frame_id")
        if frame_id is None:
            frame_id = str(index)
        stem_candidate = frame_id_to_stem(str(frame_id), index)
        rgb_path = find_rgb_path(rgb_dir, stem_candidate, str(frame_id), index, exts)
        if rgb_path is None:
            print(f"Skip {stem_candidate}: no RGB found in {rgb_dir} for stem={stem_candidate!r} frame_id={frame_id!r}", file=sys.stderr)
            continue
        try:
            img = Image.open(rgb_path).convert("RGB")
        except Exception as e:
            print(f"Skip {stem_candidate}: cannot open RGB {rgb_path}: {e}", file=sys.stderr)
            continue
        # Stable output filenames: frame_000000, frame_000001, ...
        stem = f"frame_{exported:06d}"
        rgb_h, rgb_w = img.size[1], img.size[0]
        fx, fy, cx, cy = K_to_fxfycxcy(K)
        if (depth_h, depth_w) != (rgb_h, rgb_w):
            scale_x = rgb_w / depth_w
            scale_y = rgb_h / depth_h
            depth = resize_depth_nearest(depth, rgb_h, rgb_w)
            fx, fy, cx, cy = scale_intrinsics(fx, fy, cx, cy, scale_x, scale_y)
            depth_h, depth_w = rgb_h, rgb_w
        depth_ext = ".png" if depth_format == "png16mm" else ".npy"
        depth_filename = f"{stem}_depth{depth_ext}"
        rgb_filename = f"{stem}{rgb_path.suffix}"
        intrinsic_filename = f"{stem}_intrinsics.json"
        preview_filename = f"{stem}_preview.png"
        if dry_run:
            frames_manifest.append({
                "frame_id": str(frame_id),
                "stem": stem,
                "rgb": str(rgb_out / rgb_filename),
                "depth": str(depth_out / depth_filename),
                "intrinsics": str(intrinsics_out / intrinsic_filename),
            })
            exported += 1
            continue
        # Copy or symlink RGB
        rgb_dst = rgb_out / rgb_filename
        try:
            shutil.copy2(rgb_path, rgb_dst)
        except OSError:
            try:
                rgb_dst.symlink_to(rgb_path.resolve())
            except OSError:
                shutil.copy2(rgb_path, rgb_dst)
        # Depth
        depth_path = depth_out / depth_filename
        if depth_format == "png16mm":
            write_depth_png16mm(depth_path, depth)
        else:
            write_depth_npy32(depth_path, depth)
        # Intrinsics (at RGB resolution)
        write_intrinsics_json(intrinsics_out / intrinsic_filename, fx, fy, cx, cy, rgb_w, rgb_h)
        # Preview
        if colormap:
            write_preview_colormap(preview_dir / preview_filename, depth)
        else:
            write_preview_grayscale(preview_dir / preview_filename, depth)
        pose = None
        for key in ("cam_T_world_b44", "world_T_cam_b44"):
            if key in data and data[key] is not None:
                M = data[key]
                if hasattr(M, "tolist"):
                    pose = M.tolist()
                else:
                    pose = np.asarray(M).tolist()
                break
        frames_manifest.append({
            "frame_id": str(frame_id),
            "stem": stem,
            "rgb": str(rgb_dst),
            "depth": str(depth_path),
            "intrinsics": str(intrinsics_out / intrinsic_filename),
            "intrinsics_params": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": rgb_w, "height": rgb_h},
            "pose": pose,
        })
        exported += 1
    if not dry_run and frames_manifest:
        with open(rgbd_dir / "frames.json", "w") as f:
            json.dump({"frames": frames_manifest}, f, indent=2)
    if dry_run:
        print(f"Dry run: would export {exported} frames.")
        for m in frames_manifest[:5]:
            print("  ", m.get("stem"), m.get("rgb"))
        if len(frames_manifest) > 5:
            print("  ...")
    return exported


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export RGB-D from SimpleRecon depth cache (pickles from test.py --cache_depths)."
    )
    parser.add_argument("--cache_dir", type=Path, required=True, help="Path to depth cache (root containing .pickle files)")
    parser.add_argument("--rgb_dir", type=Path, required=True, help="Path to original RGB images")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output root; creates previews/ and rgbd/ under it")
    parser.add_argument("--rgb_ext", type=str, default="", help="RGB extension: jpg|png|jpeg or empty for auto")
    parser.add_argument("--depth_format", type=str, default="png16mm", choices=["png16mm", "npy32"],
                        help="Depth format: png16mm (16-bit PNG in mm) or npy32 (float32 .npy in meters)")
    parser.add_argument("--colormap", action="store_true", help="Use colormap for preview PNGs")
    parser.add_argument("--max_frames", type=int, default=None, help="Max number of frames to export")
    parser.add_argument("--dry_run", action="store_true", help="Only list what would be exported")
    args = parser.parse_args()
    n = run_export(
        cache_dir=args.cache_dir,
        rgb_dir=args.rgb_dir,
        out_dir=args.out_dir,
        rgb_ext=args.rgb_ext,
        depth_format=args.depth_format,
        colormap=args.colormap,
        max_frames=args.max_frames,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        print(f"Exported {n} frames to {args.out_dir}")
    return 0 if n >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
