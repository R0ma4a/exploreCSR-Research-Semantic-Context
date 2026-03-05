#!/usr/bin/env bash
#
# Single-image (monocular) RGB-D pipeline for unrelated images.
# No COLMAP, no SimpleRecon. Each image is processed independently with
# monocular depth estimation (MiDaS/DPT). Output layout matches the
# multi-view export: previews/, rgbd/rgb, depth/, intrinsics/, frames.json.
#
# Usage: ./tools/run_single_image_pipeline.sh
# Or:    IMG_DIR=/path/to/photos OUT_DIR=/path/out ./tools/run_single_image_pipeline.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPORT_SCRIPT="${SCRIPT_DIR}/export_rgbd_from_monocular.py"

# ---------- Edit or override with env vars ----------
IMG_DIR="C:\Users\roman\Git Projects\BrownCSR\SimpleRecon\Work_dir\Input_IMGs\Related_IMGs_4"
OUT_DIR="C:\Users\roman\Git Projects\BrownCSR\SimpleRecon\Output_dir"
DEPTH_FORMAT="${DEPTH_FORMAT:-png16mm}"   # png16mm or npy32
MODEL="${MODEL:-MiDaS_small}"             # DPT_Large, DPT_Hybrid, MiDaS, MiDaS_small

mkdir -p "$OUT_DIR"

echo "=== Monocular RGB-D export (unrelated images) ==="
echo "  Input:  $IMG_DIR"
echo "  Output: $OUT_DIR"
echo "  Model:  $MODEL"
echo ""

python "$EXPORT_SCRIPT" \
  --rgb_dir "$IMG_DIR" \
  --out_dir "$OUT_DIR" \
  --depth_format "$DEPTH_FORMAT" \
  --model "$MODEL"

echo "Done. Output: $OUT_DIR/previews and $OUT_DIR/rgbd"
