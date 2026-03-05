# RGB-D Pipeline: Photos → Depth → Export

This document describes **reproducible pipelines** that take a folder of photos and produce:

1. **Preview depth visualizations** — quick-look images for inspection.
2. **RGB-D outputs** — aligned depth maps, intrinsics, and a manifest for downstream 3D work.

**Two modes:**

| Mode | Use case | Needs |
|------|----------|--------|
| **Multi-view (COLMAP + SimpleRecon)** | Multiple images of the **same scene** with overlap | COLMAP, SimpleRecon, poses |
| **Single-image (monocular)** | **Unrelated** images; each processed independently | PyTorch + MiDaS (no COLMAP, no SimpleRecon) |

- **Multi-view**: Better depth quality and metric scale when you have overlapping views; requires COLMAP to succeed (enough matches). See sections 1–6 below.
- **Single-image**: Use `tools/export_rgbd_from_monocular.py` or `tools/run_single_image_pipeline.sh`. No poses; depth scale is relative. See **Section 0** below.

The pipeline does **not** modify the [SimpleRecon](https://github.com/nianticlabs/simplerecon) codebase; SimpleRecon is used only for the multi-view path.

---

## 0. Single-image pipeline (unrelated images)

For **individual, unrelated images** (no shared scene, no COLMAP):

1. Install: PyTorch, torchvision, numpy, Pillow. Optional: `opencv-python` for 16-bit depth PNG.\n+   - Some MiDaS/DPT backbones also require `timm` (`python -m pip install timm`).
2. Run:

   ```bash
   python tools/export_rgbd_from_monocular.py --rgb_dir /path/to/images --out_dir /path/to/export_out
   ```

   Or use the shell wrapper:

   ```bash
   IMG_DIR=/path/to/photos OUT_DIR=/path/out ./tools/run_single_image_pipeline.sh
   ```

3. Output layout is the same as the multi-view export: `out_dir/previews/`, `out_dir/rgbd/rgb/`, `depth/`, `intrinsics/`, `frames.json`.

**Options:** `--depth_format npy32`, `--colormap`, `--max_frames N`, `--dry_run`, `--model MiDaS_small` (faster), `--default_fov_deg 60`.

**Note:** Monocular depth is **relative** (not metric). Intrinsics are guessed from EXIF or a default FOV. Poses are not available.

---

## 1. Setup (multi-view pipeline)

### 1.1 Python environment and dependencies

- Create a conda env from SimpleRecon’s `simplerecon_env.yml` (see [SimpleRecon README](https://github.com/nianticlabs/simplerecon)).
- Install SimpleRecon’s dependencies and download the **hero_model** weights into `weights/` (see SimpleRecon’s “Models” section).
- For the **export script** only:
  - Required: `numpy`, `Pillow` (PIL).
  - For 16-bit depth PNG output: `opencv-python` (`pip install opencv-python`).  
    Without OpenCV, use `--depth_format npy32` (float32 `.npy` in meters).
  - PyTorch is only needed if the cache was produced with tensors in the pickle; the export script can run with just numpy + PIL for pre-converted caches, but typically you need PyTorch to load SimpleRecon’s pickle files.

### 1.2 Folder layout (reference)

```
WORK_DIR/
  images/          # Input photos (e.g. from phone/camera)
  colmap_db/       # COLMAP database and sparse model (created by pipeline)
  sparse/          # COLMAP sparse reconstruction
  dense/           # Optional dense model
  simplerecon/     # Clone or submodule of SimpleRecon
  simplerecon_results/   # SimpleRecon output (depths cache, viz, etc.)
  export_out/       # Output of export script (previews + rgbd)
```

---

## 2. COLMAP: From photos to poses and intrinsics

SimpleRecon expects **posed RGB images** and (for COLMAP data) metric-ish scale. COLMAP gives you camera poses and intrinsics; **metric scale is unknown** unless you provide scale (e.g. from a known distance). You may need to scale poses/depth later.

### 2.1 Recommended COLMAP commands

Run from a working directory that contains your `images/` folder (e.g. `WORK_DIR`).

```bash
# Set paths (edit to match your layout)
IMG_DIR=images
WORK_DIR=.
COLMAP_DB=colmap_db
SPARSE_DIR=sparse
DENSE_DIR=dense

# 1) Feature extraction
colmap feature_extractor \
  --database_path "$COLMAP_DB/database.db" \
  --image_path "$IMG_DIR"

# 2) Exhaustive matching (use sequential_matcher for ordered images)
colmap exhaustive_matcher \
  --database_path "$COLMAP_DB/database.db"

# 3) Mapper (sparse reconstruction)
mkdir -p "$SPARSE_DIR"
colmap mapper \
  --database_path "$COLMAP_DB/database.db" \
  --image_path "$IMG_DIR" \
  --output_path "$SPARSE_DIR"

# 4) Image undistorter (optional; exports undistorted images and cameras)
colmap image_undistorter \
  --image_path "$IMG_DIR" \
  --input_path "$SPARSE_DIR/0" \
  --output_path "$DENSE_DIR" \
  --output_type COLMAP
```

### 2.2 Export cameras and images for SimpleRecon

- Use COLMAP’s text or binary export in `sparse/0/` (e.g. `cameras.txt`, `images.txt`, `points3D.txt`).
- SimpleRecon’s COLMAP dataloader expects a specific layout; see SimpleRecon’s “COLMAP Dataset” section. You may need to **crop images** to a ScanNet-like FOV and **scale poses** using known real-world scale if available.
- **Scale ambiguity**: Without scale, depths and poses are up to an unknown scale factor. SimpleRecon’s depth predictions are in a metric-like range (e.g. 0.25–5 m); you may need to align COLMAP scale to that (e.g. by scaling poses or depth after export).

---

## 3. Prepare SimpleRecon dataset config and tuples

- Point SimpleRecon’s **data config** to your COLMAP output (dataset path, tuple file location, etc.). Edit the appropriate YAML under `configs/data/` (e.g. a COLMAP-specific config or duplicate and adapt an existing one).
- Ensure a **tuple / split file** exists that lists which frames to run (e.g. one line per reference frame). You may need to run SimpleRecon’s tuple generation scripts (e.g. `generate_test_tuples.py`) or create a minimal list that matches your COLMAP `images.txt` frame names.

See SimpleRecon’s README and `configs/data/` for the exact keys (`dataset_path`, `tuple_info_file_location`, etc.).

---

## 4. Run SimpleRecon inference with depth caching

From the SimpleRecon repo root:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --name HERO_MODEL \
  --output_base_path /path/to/simplerecon_results \
  --config_file configs/models/hero_model.yaml \
  --load_weights_from_checkpoint weights/hero_model.ckpt \
  --data_config configs/data/your_colmap_config.yaml \
  --num_workers 8 \
  --batch_size 2 \
  --cache_depths \
  --dump_depth_visualization
```

- **`--cache_depths`**: Saves per-frame depth (and intrinsics, frame_id, etc.) as **pickle files** under  
  `output_base_path/name/dataset/frame_tuple_type/depths/<scan_id>/*.pickle`.  
  This is what the export script reads.
- **`--dump_depth_visualization`**: Writes quick depth visualizations under  
  `output_base_path/.../viz/quick_viz/`.  
  This is for a fast visual check only; the **export script** produces the canonical **previews** and **rgbd** outputs with aligned depth and intrinsics.

---

## 5. Export RGB-D: previews and machine-readable outputs

Use the export script to turn the depth cache into **previews** and **rgbd** (rgb + depth + intrinsics + manifest).

### 5.1 Command

```bash
python tools/export_rgbd_from_simplerecon_cache.py \
  --cache_dir /path/to/simplerecon_results/HERO_MODEL/your_dataset/your_tuple_type/depths \
  --rgb_dir /path/to/your/original/images \
  --out_dir /path/to/export_out
```

Optional arguments:

- `--rgb_ext jpg` or `png` — restrict RGB extension (default: auto-detect .jpg, .jpeg, .png).
- `--depth_format png16mm` (default) — depth as 16-bit PNG in **millimeters** (requires `opencv-python`).  
  Use `--depth_format npy32` for float32 `.npy` in **meters** (no OpenCV needed).
- `--colormap` — use a colorized depth preview instead of grayscale.
- `--max_frames N` — limit number of frames.
- `--dry_run` — only list what would be exported (no files written).

### 5.2 Output layout

```
export_out/
  previews/           # Human-viewable depth images
    frame_000000_preview.png
    frame_000001_preview.png
    ...
  rgbd/
    rgb/              # RGB frames (copied or symlinked)
      frame_000000.jpg
      frame_000001.png
      ...
    depth/            # Depth maps (png16mm or npy32)
      frame_000000_depth.png   # 16-bit mm, or
      frame_000000_depth.npy  # float32 meters
      ...
    intrinsics/       # Per-frame intrinsics
      frame_000000_intrinsics.json
      ...
    frames.json       # Manifest: frame_id, paths, intrinsics, optional pose
```

### 5.3 Manifest `frames.json`

Structure:

```json
{
  "frames": [
    {
      "frame_id": "original_id_from_cache",
      "stem": "frame_000000",
      "rgb": "path/to/rgbd/rgb/frame_000000.jpg",
      "depth": "path/to/rgbd/depth/frame_000000_depth.png",
      "intrinsics": "path/to/rgbd/intrinsics/frame_000000_intrinsics.json",
      "intrinsics_params": { "fx", "fy", "cx", "cy", "width", "height" },
      "pose": null
    }
  ]
}
```

`pose` is only set if present in the cache (SimpleRecon’s default cache does not store pose).

### 5.4 Intrinsics JSON

Each `*_intrinsics.json` contains:

- `fx`, `fy`, `cx`, `cy` — camera intrinsics.
- `width`, `height` — resolution of the **RGB (and aligned depth)** image.

If the cached depth had a different resolution, the export script **resizes depth to RGB** with nearest-neighbor and **scales intrinsics** accordingly (see script docstring).

### 5.5 Depth units and format

- **Cache**: SimpleRecon depth is in **meters** (linear scale; cost volume ~0.25–5 m).
- **png16mm**: Export converts meters → millimeters, clamps to [0, 65535], saves as uint16 PNG.
- **npy32**: Float32 array in **meters**.

---

## 6. Notes and troubleshooting

### 6.1 Pose / intrinsics mismatch

- Poses come from COLMAP (or your dataset), not from the depth cache. If you need poses in `frames.json`, you must either extend SimpleRecon’s cache to write them or merge them in a post-step from COLMAP’s `images.txt` / `cameras.txt`.

### 6.2 Resizing and intrinsics scaling

- When depth resolution ≠ RGB resolution, the script:
  - Resizes depth to RGB size with **nearest-neighbor**.
  - Scales intrinsics: `scale_x = rgb_w / depth_w`, `scale_y = rgb_h / depth_h`, then  
    `fx *= scale_x`, `fy *= scale_y`, `cx *= scale_x`, `cy *= scale_y`.

### 6.3 Depth scale ambiguity

- COLMAP’s scale is arbitrary unless you fix it (e.g. known baseline or distance). SimpleRecon predicts metric-like depth; combining COLMAP poses with SimpleRecon depth may require a global scale factor for consistent 3D.

### 6.4 Verifying alignment (depth vs RGB)

- Overlay depth edges on RGB: e.g. compute depth gradients or contours, then alpha-blend with the RGB image. Alternatively, render the depth as a colored overlay (using the same intrinsics) and check that object boundaries line up.
- Quick check: open `previews/frame_000000_preview.png` next to `rgbd/rgb/frame_000000.*` and compare edges.

### 6.5 Missing pickle keys

- If the script reports missing keys, it will print the keys present in the pickle. Expected keys include `depth_pred_s0_b1hw`, `K_full_depth_b44` or `K_s0_b44`, and `frame_id`. Ensure you ran SimpleRecon with `--cache_depths` and the same code version that writes these keys.

### 6.6 RGB not found for a frame

- The script matches cache `frame_id` to files in `--rgb_dir` by trying: stem from `frame_id`, raw `frame_id`, and `frame_000000`, `frame_000001`, … by index. If your image names differ, place a copy/symlink with one of these naming conventions or extend the script’s `find_rgb_path` logic.

---

## 7. One-command skeleton (optional)

See `tools/run_full_pipeline.sh` for a parameterized skeleton that runs COLMAP → SimpleRecon → export. Edit the variables at the top (e.g. `IMG_DIR`, `WORK_DIR`, `OUT_DIR`) to match your paths.
