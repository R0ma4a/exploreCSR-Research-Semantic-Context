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
  │
  └─ FeaturePoseTracker ────────────────── Δfeature ↔ Δpose correlation
                                            plots, CSV export, summary stats
```

## Setup

```bash
# 1. Install the package (editable)
pip install -e .

# 2. Install dependencies
pip install torch torchvision timm opencv-python numpy matplotlib
```

Requires Python ≥ 3.9 and a DepthAnything V2 checkpoint (`depth_anything_v2_vitb.pth`).

## Checkpoints

CAPTRA pretrained weights live in `RGed-research/captra/runs/` — one directory per network:

| Category | Index | rot dir                          | coord dir                           |
|----------|-------|----------------------------------|-------------------------------------|
| bottle   | 1     | `runs/1_bottle_rot`              | `runs/1_bottle_coord`               |
| bowl     | 2     | `runs/2_bowl_rot`                | `runs/2_bowl_coord`                 |
| camera   | 3     | `runs/3_camera_rot`              | `runs/3_camera_coord`               |
| can      | 4     | `runs/4_can_rot`                 | `runs/4_can_coord`                  |
| laptop   | 5     | `runs/5_laptop_rot`              | `runs/5_laptop_coord`               |
| mug      | 6     | `runs/6_mug_rot`                 | `runs/6_mug_coord`                  |

## Running

### Video

```bash
python -m exploreCSR.scripts.run_tracked \
    --video clip.mp4 \
    --weights depth_anything_v2_vitb.pth \
    --prompt "mug" \
    --rot-dir   RGed-research/captra/runs/6_mug_rot \
    --coord-dir RGed-research/captra/runs/6_mug_coord \
    --category 6 \
    --fps 4.0 \
    --preview \
    --title "Mug 4fps" \
    --save-plot results/mug_4fps.png \
    --save-csv  results/mug_4fps.csv
```

### Image sequence

```bash
python -m exploreCSR.scripts.run_tracked \
    --glob "frames/*.jpg" \
    --weights depth_anything_v2_vitb.pth \
    --prompt "mug" \
    --rot-dir   RGed-research/captra/runs/6_mug_rot \
    --coord-dir RGed-research/captra/runs/6_mug_coord \
    --category 6 \
    --preview \
    --save-plot results/mug_seq.png
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--fps` | `2.0` | Frames per second sampled from video |
| `--preview` | off | Show live cv2 window: mask overlay + pose text per frame |
| `--category` | `6` | NOCS category index (1–6, see table above) |
| `--feature-key` | `object_mean` | `object_mean` (masked patch avg) or `cls_token` |
| `--save-plot` | — | Save final plot to PNG instead of showing |
| `--save-csv` | — | Export per-frame data to CSV |
| `--no-plot` | off | Skip plotting, print summary only |
| `--title` | — | Custom title for the plot |

## Output

**Plot** — 3×3 figure:

- Row 1: Absolute translation trajectory (X, Y, Z axes separately)
- Row 2: Rotation angle from frame 0 · Δfeature · Δtranslation (with rolling mean)
- Row 3: Δrotation · scatter(Δtranslation vs Δfeature) · scatter(Δrotation vs Δfeature)

Each scatter shows Pearson r and a regression line. Frame index is color-encoded.

**Summary stats** printed to console include:

- Per-signal mean / std / max
- `correlation_feature_translation` — Pearson r(Δfeature, Δtranslation)
- `correlation_feature_rotation` — Pearson r(Δfeature, Δrotation)
- `correlation_feature_total_motion` — Pearson r(Δfeature, normalized Δtrans + Δrot)
- Mean cosine similarity between consecutive feature vectors

**CSV** — one row per frame: `tx, ty, tz, rotation_angle_from_ref_deg, scale, delta_feature_mag, delta_translation_mag, delta_rotation_deg, cosine_similarity`

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
│   ├── captra_network/      # Network code (copied from CAPTRA repo, self-contained)
│   └── old/                 # Archived geometric (PCA-based) pipeline
│
├── combination/
│   ├── tracker.py           # FeaturePoseTracker — stores frames, computes deltas, plots
│   └── features.py          # extract_object_features helper
│
├── visualization/
│   └── viz.py               # mask overlay, depth viz, point cloud, pose summary
│
└── scripts/
    └── run_tracked.py       # CLI entry point
```

## How it works

1. **DepthAnything V2** converts each RGB frame to a normalized depth map `[0, 1]`, then scaled to approximate metric depth `(1 - d) * 3.0 + 0.3`.
2. **DINOSegmenter** uses DINO self-attention to segment the prompted object and extracts `object_mean` (L2-normalized average of masked patch tokens).
3. **Neural CAPTRA** back-projects the masked depth into a point cloud (4096 pts via farthest-point sampling), then runs two trained PointNet2 networks — a rotation net and a coordinate net — to produce an **absolute** `(R, t, scale)` per frame using NOCS category-level priors.
4. **FeaturePoseTracker** accumulates absolute poses and features, then computes frame-to-frame deltas and Pearson correlations between feature change and motion magnitude.
