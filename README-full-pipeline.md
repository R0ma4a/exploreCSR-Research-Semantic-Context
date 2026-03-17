## BrownCSR Full Pipeline Test

This document explains how to run an end-to-end test of the current BrownCSR research pipeline using the provided helper script.

The script connects:

- **DepthAnything** (`depth_anything.DepthAnything`) for RGB → depth
- **DINO** (`dino.dino`) for segmentation (steps 1–4 + upsampling)
- **CAPTRA** (`CAPTRA` in `RGed-research/captra.py`) for object-centered pose/reference estimation
- **CAPTRA visualization** (`captra_viz.py`) for optional debugging plots

The goal is to go from a single RGB input image to:

- A DINO-derived segmentation map
- A CAPTRA pose vector and reference frame
- Optional visualizations of mask, depth, and 3D geometry

---

## Requirements

- Python environment with the BrownCSR repo checked out.
- Dependencies already used in the repo, including at least:
  - **torch**
  - **timm**
  - **numpy**
  - **opencv-python**
  - **scikit-learn**
  - **matplotlib**
  - Any libraries required by `depth_anything_v2` and the CAPTRA submodule.
- A DepthAnything checkpoint file (e.g. `depth_anything_v2_vitb.pth`).

The script assumes that `RGed-research` is present in the BrownCSR root directory (which matches the existing layout) and dynamically adds it to `sys.path`, so you do **not** need to install it as a package.

---

## Script locations

- **Full pipeline test**:
  - `run_full_pipeline.py` (in the **BrownCSR root directory**)
  - Entry point for: RGB → DepthAnything → DINO → CAPTRA (+ optional visualization)

- **DINO segmentation debugging helper**:
  - `debug_dino_segmentation.py` (in the **BrownCSR root directory**)
  - Entry point for quickly visualizing DINO segmentation labels as overlays, so you can select a good `--target-label` for CAPTRA.

You can call both scripts directly with `python SCRIPT_NAME.py ...` from the BrownCSR root.

---

## Basic usage: full pipeline

### Minimal example (with default checkpoint path)

From the BrownCSR root:

```bash
python run_full_pipeline.py --image PATH/TO/IMAGE.jpeg --show
```

This will:

- Load the RGB image from `PATH/TO/IMAGE.jpeg`.
- Use the default DepthAnything checkpoint path (matching `RGed-research/main.py`).
- Predict a normalized depth map for the image.
- Run DINO (steps 1–4) to obtain a patch-level segmentation and then upsample it to image size.
- Run CAPTRA on the RGB, depth, and upsampled segmentation to estimate:
  - Translation (3D)
  - Rotation (Euler XYZ)
  - Scale
  - A pose vector `[tx, ty, tz, rx, ry, rz, s]`
- Print a concise pose summary and basic diagnostics.
- Display a few matplotlib figures (`--show`):
  - Mask overlay on RGB
  - Masked depth visualization
  - Object point cloud
  - Reference frame axes overlaid on the point cloud

### Arguments

- **`--image`** (required): Path to the input RGB image.
- **`--weights`** (optional):
  - Path to the DepthAnything checkpoint `.pth` file.
  - Defaults to the same path that `RGed-research/main.py` uses.
- **`--fx`, `--fy`** (optional):
  - Camera focal lengths in pixels.
  - Defaults: `fx = 500.0`, `fy = 500.0`.
- **`--cx`, `--cy`** (optional):
  - Camera principal point coordinates in pixels.
  - Defaults: `cx = image_width / 2`, `cy = image_height / 2`.
- **`--depth-scale`** (optional):
  - Scale factor applied to depth values before they reach CAPTRA.
  - Because DepthAnything outputs normalized depth, `1.0` is typically fine.
- **`--k`** (optional):
  - Number of clusters for the DINO KMeans segmentation.
  - Default: `5`.
- **`--target-label`** (optional):
  - Integer segmentation label to treat as the object of interest.
  - If omitted, CAPTRA auto-selects the most frequent non-zero label.
- **`--show`** (flag):
  - If set, enable visualizations using `captra_viz.py`.

---

## Example commands: full pipeline

### 1. Quick sanity check (single RGB + default settings)

```bash
cd path/to/BrownCSR

python run_full_pipeline.py \
  --image RGed-research/imgs/obj00004/img00003.jpeg \
  --show
```

What you should see:

- Console output summarizing:
  - Segmentation labels present.
  - CAPTRA pose (translation, rotation Euler angles in degrees, scale).
  - Diagnostics about mask size and number of valid depth points.
- A few matplotlib windows showing:
  - Mask overlay on the RGB image.
  - Masked, smoothed depth.
  - Approximate object point cloud.
  - Principal axes / reference frame attached to the object.

### 2. Custom checkpoint and intrinsics

```bash
python run_full_pipeline.py \
  --image /path/to/your_image.png \
  --weights /path/to/depth_anything_v2_vitb.pth \
  --fx 520 --fy 520 --cx 320 --cy 240 \
  --k 4 \
  --target-label 1 \
  --show
```

Use this form if your camera intrinsics are known and you prefer to align CAPTRA with the true pinhole model rather than generic defaults.

---

## DINO segmentation debugging helper

Before running CAPTRA on a new dataset, it is often useful to inspect which DINO segmentation label corresponds to your object of interest.

Use the helper script:

```bash
python debug_dino_segmentation.py \
  --image PATH/TO/IMAGE.jpeg \
  --k 5
```

This will:

- Run the same DINO pipeline used in the full script (steps 1–4 + upsampling).
- Print the unique labels present in the segmentation and their pixel counts.
- Open a series of matplotlib windows, each showing the RGB image overlaid with a single label’s mask.

You can then:

- Visually identify which label (e.g. 1, 2, 3, …) best matches your object.
- Use that label index as the `--target-label` in `run_full_pipeline.py`.

If you only want printed stats (no plotting), use:

```bash
python debug_dino_segmentation.py \
  --image PATH/TO/IMAGE.jpeg \
  --k 5 \
  --no-show
```

---

## Interpreting full-pipeline outputs

The full-pipeline script prints a **pose summary** obtained from CAPTRA:

- **Translation**: 3D vector capturing centroid shift between reference frames. For a single frame (no previous state), it will be zero.
- **Rotation (Euler XYZ, degrees)**: Orientation of the object frame relative to a previous reference. For a single frame, this is identity (zeros).
- **Scale**: Ratio of object size between frames. For a single frame, this is `1.0`.
- **Pose vector**: Concatenation `[tx, ty, tz, rx, ry, rz, s]`, suitable for feeding into the later MLP as the **x-axis** coordinates (pose change).

Diagnostic values include:

- Number of pixels in the object mask.
- Number of valid depth points (after masking and filtering).
- Boolean validity flag and status message.

These let you quickly spot issues such as:

- Empty or tiny masks.
- Missing/invalid depth.
- Degenerate PCA cases where the reference frame cannot be estimated reliably.

---

## Extending the pipeline

Once you are comfortable that the end-to-end script works, you can:

- Integrate the **pose vector** output into your MLP training script, using it as the x-axis representation of pose change.
- Connect feature embeddings (from DINO) to form the y-axis via cosine similarity.
- Add small loops over sequences of frames to test temporal consistency by:
  - Passing the `reference_state` from frame \(t\) into frame \(t+1\).
  - Recording the resulting pose vectors across the sequence.

The `run_full_pipeline.py` script is intentionally lightweight and explicit so that you can copy or adapt its logic inside notebooks, training scripts, or experiment drivers without needing to restructure the rest of the codebase.

