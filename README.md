# exploreCSR Research — Semantic Context & Object Pose

**Research question:** Can the scalability of an object still encompass semantic connotation?
Does semantic meaning encoded in deep features remain stable as an object undergoes translation, rotation, and scale change across frames?

This repository investigates whether DINO visual features and 6-DoF pose change (from neural CAPTRA) are correlated — and whether semantic identity is preserved across viewpoint and scale variation.

---

## Pipeline

```
RGB video / images
  │
  ├─ DepthAnything V2 ──── per-frame metric depth map
  │
  ├─ DINO ──────────────── object mask (self-attention)
  │                         patch feature vectors (object_mean / cls_token)
  │
  ├─ Neural CAPTRA ─────── absolute 6-DoF pose per frame
  │   PointNet2 + NOCS     translation (3,), rotation (3×3), scale
  │   category-level       + centroid & scale geometric fallbacks
  │
  └─ FeaturePoseTracker ── Δfeature ↔ Δpose correlation
                            ICP frame-to-frame rotation, Pearson r, plots, CSV
```

The tracker produces a 3×3 figure with translation trajectories, ICP rotation accumulation, scale over time, and scatter plots of feature change vs pose change — each with a Pearson correlation coefficient and regression line.

---

## Repository Structure

```
exploreCSR-Research-Semantic-Context/
│
├── exploreCSR/                  ← Main research pipeline (Python package)
│   ├── pipeline.py              #   run_sequence_tracked, run_video_tracked
│   ├── depth/                   #   DepthAnything V2 wrapper
│   ├── segmentation/            #   DINO segmenter + feature extractor
│   ├── pose/                    #   Neural CAPTRA (PointNet2 + NOCS)
│   ├── combination/             #   FeaturePoseTracker — deltas, ICP, plots
│   ├── tracking/                #   Optical-flow point correspondence analysis
│   ├── visualization/           #   Mask overlay, depth, point cloud, pose summary
│   └── scripts/                 #   CLI entry points
│       ├── run_tracked.py       #     Unified: video or image sequence
│       ├── run_video.py         #     Video only
│       └── run_sequence.py      #     Image sequence only
│
├── weights/                     ← Pretrained checkpoint files (not in git)
│   ├── depth_anything_v2_vitb.pth
│   └── captra/runs/             #   12 CAPTRA dirs (6 categories × rot + coord)
│
├── results/                     ← Output plots from experiments
│   ├── mug_translation_baseline.png
│   ├── mug_rotation_baseline.png
│   ├── mug_scale_baseline.png
│   └── mug_complex_{1,2,3}_results.png
│
├── docs/                        ← Research proposal (LaTeX)
│   └── main.tex
│
├── RGed-research/               ← Original CAPTRA weights + legacy scripts
│   └── captra/runs/             #   Pretrained checkpoints (source)
│
├── Pointnet2_PyTorch/           ← PointNet2 ops (git submodule)
├── Depth-Anything-V2/           ← DepthAnything V2 source (reference)
└── pyproject.toml               ← Package build config
```

---

## Quick Start

### 1. System requirements

| Requirement | Notes |
|-------------|-------|
| Python ≥ 3.9 | |
| CUDA toolkit ≥ 11.x | Required for PointNet2 CUDA extensions |
| C++ compiler | MSVC 2019+ (Windows) / GCC 9+ (Linux) |
| CUDA-capable GPU | CPU fallback is very slow |

### 2. Install dependencies

```bash
pip install torch torchvision timm opencv-python numpy matplotlib scipy scikit-learn
pip install -e .
```

### 3. Build PointNet2 CUDA extensions

```bash
cd exploreCSR/pose/captra_network/network/models/pointnet_lib
python setup.py install
```

### 4. Place checkpoints in `weights/`

```
weights/
├── depth_anything_v2_vitb.pth
└── captra/runs/
    ├── 6_mug_rot/ckpt/model_0000.pt
    ├── 6_mug_coord/ckpt/model_0000.pt
    └── ...   (see exploreCSR/README.md for full table)
```

### 5. Run

```bash
# Video
python -m exploreCSR.scripts.run_tracked \
    --video clip.mp4 \
    --weights weights/depth_anything_v2_vitb.pth \
    --prompt "mug" \
    --captra-weights-dir weights/captra/runs \
    --category 6 \
    --fps 4.0 \
    --coupling-alpha 5.4053 \
    --save-plot results/my_run.png

# Image sequence
python -m exploreCSR.scripts.run_tracked \
    --glob "frames/*.jpg" \
    --weights weights/depth_anything_v2_vitb.pth \
    --prompt "mug" \
    --captra-weights-dir weights/captra/runs \
    --category 6 \
    --save-plot results/my_run.png
```

See [exploreCSR/README.md](exploreCSR/README.md) for the full flag reference and detailed setup instructions.

---

## Research Background

This project is part of the [Brown University exploreCSR](https://explorecsr.cs.brown.edu/) program.

**Core hypothesis:** Semantic feature vectors extracted by DINO from masked object regions will change in proportion to the magnitude of 6-DoF pose change (translation, rotation, scale). Preserved correlation across scale changes would indicate that semantic connotation survives scalability.

**Methodology:**
1. Segment the object of interest in each frame using DINO self-attention.
2. Recover per-frame absolute pose `(R, t, scale)` via neural CAPTRA (category-level PointNet2 networks trained on NOCS).
3. Extract DINO patch features from the masked region.
4. Compute frame-to-frame deltas: Δfeature magnitude, Δtranslation, Δscale, ΔICP rotation.
5. Measure Pearson correlation between feature change and each pose axis.

**Validation:** A successful outcome occurs when scale-variant observations preserve high feature similarity (cosine) while still tracking distinct pose coordinates — confirming that semantic connotation is maintained across scalable representations.

---

## NOCS Category Index

| Index | Category |
|-------|----------|
| 1 | bottle |
| 2 | bowl |
| 3 | camera |
| 4 | can |
| 5 | laptop |
| 6 | mug |
