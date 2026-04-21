# weights/

Place pretrained checkpoint files here.

## Directory layout

```
weights/
├── depth_anything_v2_vitb.pth          ← download from DepthAnything V2 releases
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

## Usage

Pass `--captra-weights-dir weights/captra/runs` to any script and it will
auto-resolve the correct rot/coord directories from `--category`.
