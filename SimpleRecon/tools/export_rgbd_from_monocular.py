#!/usr/bin/env python3
"""
Export RGB-D from a folder of unrelated images using monocular depth estimation.

No COLMAP or SimpleRecon required. Each image is processed independently.
Output layout matches the SimpleRecon export: previews/, rgbd/rgb, depth/,
intrinsics/, frames.json. Intrinsics are guessed from EXIF (focal length) or
a default FOV; depth scale is relative (not metric) unless you calibrate.

Requires: PyTorch, torchvision, numpy, Pillow. Optional: opencv-python for
--depth_format png16mm.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# Reuse writers from the SimpleRecon export script when run from same repo
try:
    from export_rgbd_from_simplerecon_cache import (
        write_depth_npy32,
        write_depth_png16mm,
        write_intrinsics_json,
        write_preview_colormap,
        write_preview_grayscale,
    )
except ImportError:
    # Standalone fallbacks (minimal copies)
    def write_depth_png16mm(path: Path, depth_m: np.ndarray) -> None:
        depth_mm = np.clip(depth_m * 1000.0, 0, 65535)
        arr = np.nan_to_num(depth_mm, nan=0.0).astype(np.uint16)
        try:
            import cv2
            cv2.imwrite(str(path), arr)
        except ImportError:
            raise RuntimeError("16-bit PNG requires opencv-python. Use --depth_format npy32.")

    def write_depth_npy32(path: Path, depth_m: np.ndarray) -> None:
        np.save(path, np.asarray(depth_m, dtype=np.float32))

    def write_preview_grayscale(path: Path, depth: np.ndarray, pct_low: float = 2.0, pct_high: float = 98.0) -> None:
        valid = np.isfinite(depth) & (depth > 0)
        vmin = np.percentile(depth[valid], pct_low) if np.any(valid) else 0.0
        vmax = np.percentile(depth[valid], pct_high) if np.any(valid) else 1.0
        if vmax <= vmin:
            vmax = vmin + 1e-6
        norm = np.clip((depth - vmin) / (vmax - vmin), 0, 1)
        Image.fromarray((np.nan_to_num(norm, nan=0.0) * 255).astype(np.uint8), mode="L").save(path)

    def write_preview_colormap(path: Path, depth: np.ndarray, pct_low: float = 2.0, pct_high: float = 98.0) -> None:
        valid = np.isfinite(depth) & (depth > 0)
        vmin = np.percentile(depth[valid], pct_low) if np.any(valid) else 0.0
        vmax = np.percentile(depth[valid], pct_high) if np.any(valid) else 1.0
        if vmax <= vmin:
            vmax = vmin + 1e-6
        norm = np.clip((depth - vmin) / (vmax - vmin), 0, 1)
        norm = np.nan_to_num(norm, nan=0.0)
        r = (norm * 255).astype(np.uint8)
        g = (np.clip(1.2 * norm - 0.1, 0, 1) * 255).astype(np.uint8)
        b = (np.clip(1.5 * norm - 0.5, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(np.stack([r, g, b], axis=-1), mode="RGB").save(path)

    def write_intrinsics_json(path: Path, fx: float, fy: float, cx: float, cy: float, width: int, height: int) -> None:
        with open(path, "w") as f:
            json.dump({"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": width, "height": height}, f, indent=2)


def get_focal_from_exif(img: Image.Image) -> Optional[float]:
    """Guess focal length in pixels from EXIF (35mm equivalent -> px). Returns None if not available."""
    try:
        exif = img.getexif()
        if not exif:
            return None
        # Focal length in mm
        focal_mm = exif.get(37386)  # FocalLength
        if focal_mm is None:
            return None
        focal_mm = float(focal_mm)
        # Sensor / 35mm scale: assume default if not in EXIF
        try:
            from PIL.ExifTags import TAGS
            # FocalLengthIn35mmFilm (41989) gives 35mm-equiv
            fl35 = exif.get(41989)
            if fl35 is not None:
                focal_mm = float(fl35)
        except Exception:
            pass
        w, h = img.size
        # Approximate: fx ≈ focal_px with fx = focal_mm * width_mm / sensor_width_mm.
        # Assume ~60deg horizontal FOV for typical phone => fx ≈ w / (2*tan(30°)) ≈ w * 0.87
        # Or use 35mm equiv: sensor width ~36mm => fx = focal_mm * w / 36
        fx_px = focal_mm * max(w, h) / 36.0
        return fx_px
    except Exception:
        return None


def intrinsics_from_image(img: Image.Image, default_fov_deg: float = 60.0) -> tuple[float, float, float, float]:
    """Return (fx, fy, cx, cy). Uses EXIF focal if available, else default FOV."""
    w, h = img.size
    cx = w / 2.0
    cy = h / 2.0
    fx_px = get_focal_from_exif(img)
    if fx_px is not None and fx_px > 0:
        fy = fx_px  # assume square pixels
        fx = fx_px
    else:
        # Default: horizontal FOV = default_fov_deg
        import math
        fx = w / (2.0 * math.tan(math.radians(default_fov_deg / 2.0)))
        fy = fx
    return fx, fy, cx, cy


def load_midas(model_name: str = "DPT_Large"):
    """Load MiDaS or DPT model via torch.hub."""
    import torch
    # model_name: MiDaS_small, MiDaS, DPT_Large, DPT_Hybrid, etc.
    try:
        model = torch.hub.load("intel-isl/MiDaS", model_name, trust_repo=True)
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or str(e)
        raise RuntimeError(
            f"Missing dependency while loading MiDaS model '{model_name}': {missing}\n"
            "Install missing packages, e.g.:\n"
            "  python -m pip install timm\n"
            "If you want fewer dependencies / smaller download, try:\n"
            "  --model MiDaS_small\n"
        ) from e
    model.eval()
    return model


def predict_depth_midas(model, img_rgb: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Run MiDaS on HWC uint8 RGB; return depth map (H,W) in arbitrary relative scale."""
    import torch
    from torchvision import transforms

    h_orig, w_orig = img_rgb.shape[:2]
    # Standard MiDaS preprocessing: resize to 384, ImageNet normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((384, 384), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor)
    if isinstance(pred, dict):
        pred = pred.get("depth", list(pred.values())[0])
    # Ensure NCHW for interpolate
    if pred.ndim == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
    elif pred.ndim == 3:
        # (N, H, W) -> (N, 1, H, W)
        pred = pred.unsqueeze(1)
    elif pred.ndim == 4:
        pass
    else:
        raise ValueError(f"Unexpected MiDaS output shape: {tuple(pred.shape)}")

    pred = torch.nn.functional.interpolate(
        pred,
        size=(h_orig, w_orig),
        mode="bilinear",
        align_corners=False,
    ).squeeze().cpu().numpy()
    return pred


def run_export(
    rgb_dir: Path,
    out_dir: Path,
    rgb_ext: str = "",
    depth_format: str = "png16mm",
    colormap: bool = False,
    max_frames: Optional[int] = None,
    dry_run: bool = False,
    model_name: str = "DPT_Large",
    default_fov_deg: float = 60.0,
) -> int:
    """Export RGB-D from a folder of images using monocular depth. Returns number exported."""
    import torch

    rgb_dir = Path(rgb_dir)
    out_dir = Path(out_dir)
    exts = (".jpg", ".jpeg", ".png") if not rgb_ext else (f".{rgb_ext.strip().lower().lstrip('.')}",)
    images = []
    for ext in exts:
        images.extend(rgb_dir.glob(f"*{ext}"))
    images = sorted(set(images))
    if not images:
        print(f"No images found in {rgb_dir} with extensions {exts}", file=sys.stderr)
        return 0
    if max_frames is not None:
        images = images[: max_frames]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading monocular depth model ({model_name}) on {device}...", file=sys.stderr)
    model = load_midas(model_name)
    model = model.to(device)

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
    for idx, rgb_path in enumerate(images):
        stem = f"frame_{idx:06d}"
        try:
            img = Image.open(rgb_path).convert("RGB")
        except Exception as e:
            print(f"Skip {rgb_path}: {e}", file=sys.stderr)
            continue
        w, h = img.size
        fx, fy, cx, cy = intrinsics_from_image(img, default_fov_deg=default_fov_deg)
        if dry_run:
            frames_manifest.append({
                "frame_id": rgb_path.stem,
                "stem": stem,
                "rgb": str(rgb_out / f"{stem}{rgb_path.suffix}"),
                "depth": str(depth_out / f"{stem}_depth.{'png' if depth_format == 'png16mm' else 'npy'}"),
                "intrinsics": str(intrinsics_out / f"{stem}_intrinsics.json"),
            })
            continue
        img_np = np.array(img)
        depth = predict_depth_midas(model, img_np, device=device)
        # MiDaS returns inverse-ish relative depth; convert to positive and scale so median ~ 1m for storage
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        valid = depth > 0
        if np.any(valid):
            med = np.median(depth[valid])
            if med > 1e-6:
                depth = depth / med  # scale so median = 1 (treat as ~meters for compatibility)
        depth = np.clip(depth, 0.1, 10.0)

        shutil.copy2(rgb_path, rgb_out / f"{stem}{rgb_path.suffix}")
        depth_ext = ".png" if depth_format == "png16mm" else ".npy"
        depth_path = depth_out / f"{stem}_depth{depth_ext}"
        if depth_format == "png16mm":
            write_depth_png16mm(depth_path, depth)
        else:
            write_depth_npy32(depth_path, depth)
        write_intrinsics_json(intrinsics_out / f"{stem}_intrinsics.json", fx, fy, cx, cy, w, h)
        if colormap:
            write_preview_colormap(preview_dir / f"{stem}_preview.png", depth)
        else:
            write_preview_grayscale(preview_dir / f"{stem}_preview.png", depth)
        frames_manifest.append({
            "frame_id": rgb_path.stem,
            "stem": stem,
            "rgb": str(rgb_out / f"{stem}{rgb_path.suffix}"),
            "depth": str(depth_path),
            "intrinsics": str(intrinsics_out / f"{stem}_intrinsics.json"),
            "intrinsics_params": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": w, "height": h},
            "pose": None,
            "depth_scale_note": "Monocular depth: scale is relative (median normalized to 1); not metric.",
        })

    if not dry_run and frames_manifest:
        with open(rgbd_dir / "frames.json", "w") as f:
            json.dump({"frames": frames_manifest}, f, indent=2)
    n = len(frames_manifest)
    if dry_run:
        print(f"Dry run: would export {n} frames.", file=sys.stderr)
    return n


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export RGB-D from a folder of unrelated images (monocular depth). No COLMAP or SimpleRecon."
    )
    parser.add_argument("--rgb_dir", type=Path, required=True, help="Folder of input RGB images")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output root; creates previews/ and rgbd/")
    parser.add_argument("--rgb_ext", type=str, default="", help="Filter by extension: jpg|png|jpeg or empty for all")
    parser.add_argument("--depth_format", choices=["png16mm", "npy32"], default="png16mm")
    parser.add_argument("--colormap", action="store_true")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--model", type=str, default="DPT_Large",
                        help="MiDaS model: DPT_Large, DPT_Hybrid, MiDaS, MiDaS_small")
    parser.add_argument("--default_fov_deg", type=float, default=60.0,
                        help="Default horizontal FOV in degrees when EXIF focal is missing")
    args = parser.parse_args()
    n = run_export(
        rgb_dir=args.rgb_dir,
        out_dir=args.out_dir,
        rgb_ext=args.rgb_ext,
        depth_format=args.depth_format,
        colormap=args.colormap,
        max_frames=args.max_frames,
        dry_run=args.dry_run,
        model_name=args.model,
        default_fov_deg=args.default_fov_deg,
    )
    if not args.dry_run:
        print(f"Exported {n} frames to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
