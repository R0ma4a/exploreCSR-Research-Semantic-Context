# exploreCSR — Object-Centered Pose Estimation Pipeline

```
RGB Images ──┬── Segmentation Module (DINO) ──── Object Mask ──┐
             │                                                  ├── MLP ── Visual Feedback
             └── RGB-D Processor (DepthAnything) → CAPTRA ─────┘          Coordinate Features
```

## Package Structure

```
exploreCSR/
├── __init__.py                          # Package root, version
├── config.py                            # Dataclass configs (CameraConfig, PipelineConfig, etc.)
│
├── depth/                               # Monocular depth estimation
│   ├── __init__.py
│   └── depth_anything.py                # DepthAnything V2 wrapper
│
├── segmentation/                        # Object segmentation
│   ├── __init__.py
│   └── dino_segmenter.py               # DINO attention-based segmentation
│
├── pose/                                # Pose estimation
│   ├── __init__.py
│   ├── captra.py                        # CAPTRA core (reference frame + pose change)
│   ├── validate.py                      # Synthetic validation checks
│   └── surface.py                       # Surface-relative delta poses + calibration
│
├── visualization/                       # Plotting and overlays
│   ├── __init__.py
│   └── viz.py                           # Mask overlay, depth, point cloud, reference frame
│
├── pipeline.py                          # Unified runner (single / sequence / video)
│
└── scripts/                             # CLI entry points
    ├── run_single.py                    # Single image
    ├── run_sequence.py                  # Multi-image temporal sequence
    ├── run_video.py                     # Video file processing
    └── debug_segmentation.py            # Debug DINO mask quality
```

## Usage

### As a library

```python
from exploreCSR.config import CameraConfig
from exploreCSR.pipeline import run_single, run_sequence, run_video

# Single image
result = run_single(
    image_path="photo.jpg",
    weights_path="depth_anything_v2_vitb.pth",
    prompt="bag",
)
print(result["translation"], result["scale"])

# Image sequence
results = run_sequence(
    image_paths=["frame01.jpg", "frame02.jpg", "frame03.jpg"],
    weights_path="depth_anything_v2_vitb.pth",
    prompt="person",
)

# Video
results = run_video(
    video_path="clip.mp4",
    weights_path="depth_anything_v2_vitb.pth",
    prompt="bag",
    out_csv="poses.csv",
    target_fps=2.0,
)
```

### From the command line

```bash
# Single image with visualization
python -m exploreCSR.scripts.run_single \
    --image photo.jpg \
    --weights depth_anything_v2_vitb.pth \
    --prompt "salt and pepper shaker" \
    --show

# Multi-image sequence
python -m exploreCSR.scripts.run_sequence \
    --glob "frames/*.jpg" \
    --weights depth_anything_v2_vitb.pth \
    --prompt "bag"

# Video processing
python -m exploreCSR.scripts.run_video \
    --video clip.mp4 \
    --weights depth_anything_v2_vitb.pth \
    --prompt "person" \
    --fps 2.0 \
    --out-csv results.csv

# Debug segmentation quality
python -m exploreCSR.scripts.debug_segmentation \
    --image photo.jpg \
    --weights depth_anything_v2_vitb.pth
```

### Validation

```python
from exploreCSR.pose.validate import run_all_synthetic_checks

results = run_all_synthetic_checks()
for name, metrics in results.items():
    print(f"[{name}]", metrics)
```

## Dependencies

- PyTorch, timm, depth-anything-v2
- OpenCV (`cv2`), NumPy, scikit-learn
- matplotlib (optional, for visualization only)
