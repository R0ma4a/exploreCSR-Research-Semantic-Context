# exploreCSR — Semantic Feature & Pose Correlation

Research pipeline that finds correlation between **DINO semantic features** and **6-DoF object motion** (translation + rotation) across video frames.

```
RGB video / images
  │
  ├─ DepthAnything V2 ──────────────────── depth map (normalized 0–1)
  │
  ├─ DINO (DINOSegmenter) ──────────────── object mask + patch features
  │                                         (object_mean, cls_token)
  │
  ├─ Neural CAPTRA ─────────────────────── absolute pose per frame
  │   (PointNet2 + NOCS, category-level)   translation (3,), rotation (3×3), scale
  │   + centroid/scale fallback            geometric fallback when network freezes
  │
  └─ FeaturePoseTracker ────────────────── Δfeature ↔ Δpose correlation
                                            ICP rotation, plots, CSV export
```

## System Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | ≥ 3.9 | |
| CUDA toolkit | ≥ 11.x | Required to build PointNet2 CUDA extensions |
| C++ compiler | MSVC 2019+ (Windows) / GCC 9+ (Linux) | Required to build PointNet2 CUDA extensions |
| GPU | CUDA-capable | CPU fallback exists but is very slow |

On **Windows**: install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with the "Desktop development with C++" workload. Make sure `cl.exe` is on your PATH (use a VS Developer Command Prompt or run `vcvarsall.bat x64`).

On **Linux**: `sudo apt install build-essential` (or equivalent) plus a matching CUDA toolkit.

## Setup

### 1. Install Python dependencies

```bash
pip install torch torchvision timm opencv-python numpy matplotlib scipy scikit-learn
```

| Package | Used for |
|---------|----------|
| `torch` / `torchvision` | DepthAnything, DINO, CAPTRA networks |
| `timm` | DepthAnything ViT backbone |
| `opencv-python` | Image I/O, preview window |
| `numpy` / `matplotlib` | Numerics, plotting |
| `scipy` | ICP — KDTree nearest-neighbor matching |
| `scikit-learn` | DINOSegmenter PCA / KMeans clustering |

### 2. Build the PointNet2 CUDA extensions

CAPTRA requires custom CUDA ops (ball query, FPS, grouping, interpolation).  
Build once before first use:

```bash
cd exploreCSR/pose/captra_network/network/models/pointnet_lib
python setup.py install
```

This compiles `pointnet2_cuda` against your installed CUDA toolkit and C++ compiler.  
If the build fails, confirm that:
- `nvcc --version` prints your CUDA version
- On Windows, `cl.exe` is available (run from a VS Developer Command Prompt)
- Your PyTorch CUDA version matches your installed CUDA toolkit (`torch.version.cuda`)

### 3. Install the package (editable)

```bash
pip install -e .
```

### 4. Download checkpoints

Place all checkpoint files under a `weights/` directory at the project root:

```
weights/
├── depth_anything_v2_vitb.pth          ← DepthAnything V2 backbone
└── captra/
    └── runs/
        ├── 1_bottle_rot/ckpt/model_0000.pt
        ├── 1_bottle_coord/ckpt/model_0000.pt
        ├── 2_bowl_rot/ckpt/model_0000.pt
        ├── 2_bowl_coord/ckpt/model_0000.pt
        ├── 3_camera_rot/ckpt/model_0000.pt
        ├── 3_camera_coord/ckpt/model_0000.pt
        ├── 4_can_rot/ckpt/model_0000.pt
        ├── 4_can_coord/ckpt/model_0000.pt
        ├── 5_laptop_rot/ckpt/model_0000.pt
        ├── 5_laptop_coord/ckpt/model_0000.pt
        ├── 6_mug_rot/ckpt/model_0000.pt
        └── 6_mug_coord/ckpt/model_0000.pt
```

- **DepthAnything V2**: download `depth_anything_v2_vitb.pth` from the [DepthAnything V2 releases](https://github.com/DepthAnything/Depth-Anything-V2)
- **CAPTRA weights**: one directory per network, each containing `ckpt/model_0000.pt`

| Category | Index | rot dir | coord dir |
|----------|-------|---------|-----------|
| bottle | 1 | `1_bottle_rot` | `1_bottle_coord` |
| bowl | 2 | `2_bowl_rot` | `2_bowl_coord` |
| camera | 3 | `3_camera_rot` | `3_camera_coord` |
| can | 4 | `4_can_rot` | `4_can_coord` |
| laptop | 5 | `5_laptop_rot` | `5_laptop_coord` |
| mug | 6 | `6_mug_rot` | `6_mug_coord` |

## Running

### Video

```bash
python -m exploreCSR.scripts.run_tracked \
    --video clip.mp4 \
    --weights weights/depth_anything_v2_vitb.pth \
    --prompt "mug" \
    --captra-weights-dir weights/captra/runs \
    --category 6 \
    --fps 4.0 \
    --coupling-alpha 5.4053 \
    --preview \
    --title "Mug 4fps" \
    --save-plot results/mug_4fps.png \
    --save-csv  results/mug_4fps.csv
```

### Image sequence

```bash
python -m exploreCSR.scripts.run_tracked \
    --glob "frames/*.jpg" \
    --weights weights/depth_anything_v2_vitb.pth \
    --prompt "mug" \
    --captra-weights-dir weights/captra/runs \
    --category 6 \
    --coupling-alpha 5.4053 \
    --save-plot results/mug_seq.png
```

You can also pass `--rot-dir` and `--coord-dir` directly if you want to point at a specific checkpoint location instead of using the `weights/` convention.

### Calibrate ICP coupling coefficient (translation-only baseline)

Run on a clip where the object **only translates** (no rotation), then pass the result to rotation runs:

```bash
# Step 1: fit α on a translation-only baseline
python -m exploreCSR.scripts.run_tracked \
    --video baseline_zoom.mp4 \
    --weights weights/depth_anything_v2_vitb.pth \
    --captra-weights-dir weights/captra/runs \
    --category 6 \
    --fit-coupling

# Step 2: use the printed α value on rotation runs
python -m exploreCSR.scripts.run_tracked \
    --video mug_rotation.mp4 \
    --weights weights/depth_anything_v2_vitb.pth \
    --captra-weights-dir weights/captra/runs \
    --category 6 \
    --coupling-alpha 5.4053
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | — | Input video file |
| `--images` | — | Explicit image list (in order) |
| `--glob` | — | Glob pattern, e.g. `"frames/*.jpg"` |
| `--fps` | `2.0` | Target frames-per-second sampled from video |
| `--weights` | required | DepthAnything V2 checkpoint path |
| `--prompt` | required | Text description of the object to segment |
| `--captra-weights-dir` | — | Base `runs/` directory; auto-derives rot/coord dirs from `--category` |
| `--rot-dir` | — | CAPTRA rotation net checkpoint dir (overrides `--captra-weights-dir`) |
| `--coord-dir` | — | CAPTRA coordinate net checkpoint dir (overrides `--captra-weights-dir`) |
| `--category` | `6` | NOCS category index (1–6, see table above) |
| `--feature-key` | `object_mean` | `object_mean` (masked patch avg) or `cls_token` |
| `--coupling-alpha` | `0.0` | ICP coupling correction (°/depth-unit). Use `5.4053` (fitted) or calibrate with `--fit-coupling` |
| `--fit-coupling` | off | Fit and print the translation-coupling α; run on a translation-only baseline |
| `--preview` | off | Show live cv2 preview (mask overlay + pose text, max 640px) |
| `--save-plot` | — | Save final 3×3 plot to PNG |
| `--save-csv` | — | Export per-frame data to CSV |
| `--no-plot` | off | Skip plotting, print summary only |
| `--title` | — | Custom plot title |
| `--ylim-feat` | auto | Fixed y-axis limits for Δfeature panel |
| `--ylim-trans` | auto | Fixed y-axis limits for Δtranslation panel |
| `--ylim-rot` | auto | Fixed y-axis limits for ICP rotation panel |
| `--xlim-trans` | auto | Fixed x-axis for translation scatter |
| `--xlim-rot` | auto | Fixed x-axis for rotation scatter |
| `--track-world-points` | off | Enable optical-flow correspondence-based feature analysis |
| `--max-track-points` | `300` | Max tracked points initialized from frame-0 mask |
| `--track-min-valid-ratio` | `0.7` | Min fraction of frames a track must survive |
| `--track-reference` | `prev` | Reference frame for feature deltas: `prev` or `first` |
| `--tracked-output-dir` | auto | Output directory for tracked-point artifacts |

## Output

**Plot** — 3×3 figure:

- **Row 0**: Absolute translation trajectory — X (`tx`), Y (`ty`), Z (`tz`) axes separately
- **Row 1**: ICP rotation accumulation (α-corrected, signed) · Δfeature magnitude · Δtranslation magnitude (with rolling mean overlay)
- **Row 2**: Scale over time (geometric object scale from CAPTRA) · scatter(Δtranslation vs Δfeature) · scatter(|ΔICP rotation| vs Δfeature)

Each scatter shows Pearson r and a regression line. Frame index is color-encoded.

**Summary stats** printed to console include:

- Per-signal mean / std / max
- `correlation_feature_translation` — Pearson r(Δfeature, Δtranslation)
- `correlation_feature_rotation` — Pearson r(Δfeature, |ΔICP rotation|)
- `correlation_feature_total_motion` — Pearson r(Δfeature, normalized Δtrans + |Δrot|)
- Mean cosine similarity between consecutive feature vectors
- Scale μ (mean object scale over sequence)

**CSV** — one row per frame:
`frame_idx, tx, ty, tz, scale, icp_rotation_deg, delta_feature_mag, delta_translation_mag, cosine_similarity`

## Package structure

```
exploreCSR/
├── pipeline.py              # run_sequence_tracked, run_video_tracked
├── config.py                # CAPTRAConfig, CameraConfig, PipelineConfig
│
├── depth/
│   └── depth_anything.py    # DepthAnything V2 wrapper
│
├── segmentation/
│   └── dino_segmenter.py    # DINO attention segmentation + feature extraction
│
├── pose/
│   ├── captra.py            # Neural CAPTRA (PointNet2 + NOCS) — main pose estimator
│   │                        # includes centroid fallback + scale fallback
│   ├── captra_network/      # Network code (copied from CAPTRA repo, self-contained)
│   │   └── network/models/pointnet_lib/setup.py  # CUDA extension build
│   └── old/                 # Archived geometric (PCA-based) pipeline
│
├── combination/
│   ├── tracker.py           # FeaturePoseTracker — stores frames, ICP, computes deltas, plots
│   └── features.py          # extract_object_features helper
│
├── tracking/                # Optical-flow point correspondence analysis
│
├── visualization/
│   └── viz.py               # mask overlay, depth viz, point cloud, pose summary
│
└── scripts/
    ├── run_tracked.py       # Unified CLI entry point (video or image sequence)
    ├── run_video.py         # Standalone video-only script
    └── run_sequence.py      # Standalone image-sequence script
```

## How it works

1. **DepthAnything V2** converts each RGB frame to a normalized depth map `[0, 1]`, then scaled to approximate metric depth `(1 - d) * 3.0 + 0.3`.

2. **DINOSegmenter** uses DINO self-attention to segment the prompted object and extracts `object_mean` (L2-normalized average of masked patch tokens).

3. **CAPTRA** back-projects the masked depth into a point cloud (4096 pts via farthest-point sampling), then runs two trained PointNet2 networks — a rotation net and a coordinate net — to produce an **absolute** `(R, t, scale)` per frame using NOCS category-level priors. CAPTRA uses the previous frame's pose as initialization context.

   - **Centroid fallback**: if the point cloud centroid moves > 0.01 m but CAPTRA translation responds < 25% of that for 3+ consecutive frames, the geometric centroid replaces the frozen translation output.
   - **Scale fallback**: if the mean point distance from centroid changes > 0.005 but CAPTRA scale responds < 25% for 3+ frames, the geometric scale replaces the frozen scale output.
   - Both fallbacks preserve the rotation and the internal `_prev_pose` state so CAPTRA can recover on the next frame.

4. **ICP rotation** (frame-to-frame) — unit sphere projection first removes radial (zoom) motion so only angular rotation remains. ICP is then run on the projected sphere surface to estimate signed rotation (Y-axis sign). Rotations > 45° are treated as ICP failures (returned as NaN). A coupling correction `α × Σtranslation` is subtracted from accumulated ICP rotation to remove the perspective-induced pseudo-rotation from forward/backward motion.

5. **FeaturePoseTracker** accumulates absolute poses and features, then computes frame-to-frame deltas and Pearson correlations between feature change and motion magnitude.

### Preview window

The live preview (`--preview`) shows a red mask overlay + translation text per frame. The window is automatically resized to fit on screen (longest edge ≤ 640 px).
