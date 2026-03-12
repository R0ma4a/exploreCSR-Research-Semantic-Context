#!/usr/bin/env bash
#
# Full pipeline: COLMAP → SimpleRecon (with --cache_depths) → RGB-D export.
# Edit the variables below to match your paths. Requires COLMAP and a
# SimpleRecon environment (see README_RGBD_PIPELINE.md).
#
# Usage: ./tools/run_full_pipeline.sh
# Or:    bash tools/run_full_pipeline.sh
#

set -e

# ---------- Check dependencies ----------
if ! command -v colmap &>/dev/null; then
  echo "Error: COLMAP is not installed or not on your PATH."
  echo ""
  echo "Install COLMAP:"
  echo "  • Windows: Download from https://github.com/colmap/colmap/releases and add the bin/ folder to PATH."
  echo "  • macOS:   brew install colmap"
  echo "  • Linux:   apt install colmap   (Ubuntu/Debian)  or  dnf install colmap   (Fedora)"
  echo ""
  echo "Or build from source: https://colmap.github.io/install.html"
  exit 1
fi

# ---------- Edit these paths ----------
# Resolve paths relative to this script so you can run it from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

WORK_DIR="$REPO_ROOT/Work_dir"                                  # Working directory (contains IMG_DIR)
IMG_DIR="$WORK_DIR/Input_IMGs/Related_IMGs_3"                    # Folder of input photos
OUT_DIR="$REPO_ROOT/Output_dir"                                 # Final export output root (sibling of Work_dir)
COLMAP_DB="${COLMAP_DB:-colmap_db}"
SPARSE_DIR="${SPARSE_DIR:-sparse}"
DENSE_DIR="${DENSE_DIR:-dense}"
# Matcher: "exhaustive" (all pairs) or "sequential" (adjacent pairs). Use sequential for
# ordered photos (e.g. one scene, video frames); try it if mapper fails with "No good initial pair".
COLMAP_MATCHER="${COLMAP_MATCHER:-exhaustive}"

# SimpleRecon: paths relative to WORK_DIR or absolute
SIMPLERECON_ROOT="${SIMPLERECON_ROOT:-../simplerecon}"          # Clone or submodule of SimpleRecon
SIMPLERECON_RESULTS="${SIMPLERECON_RESULTS:-simplerecon_results}"
SIMPLERECON_DATA_CONFIG="${SIMPLERECON_DATA_CONFIG:-configs/data/vdr_dense.yaml}"  # Your COLMAP/data config
SIMPLERECON_WEIGHTS="${SIMPLERECON_WEIGHTS:-weights/hero_model.ckpt}"

# Export script (same repo as this script)
EXPORT_SCRIPT="${SCRIPT_DIR}/export_rgbd_from_simplerecon_cache.py"

# ---------- Step 1: COLMAP ----------
echo "=== Step 1: COLMAP ==="
cd "$WORK_DIR"
mkdir -p "$COLMAP_DB" "$SPARSE_DIR" "$DENSE_DIR"

# Start from a clean COLMAP database / sparse model each run so old images
# from previous experiments do not linger in database.db.
if [ -f "$COLMAP_DB/database.db" ]; then
  rm -f "$COLMAP_DB/database.db"
fi
if [ -d "$SPARSE_DIR" ]; then
  rm -rf "$SPARSE_DIR"
  mkdir -p "$SPARSE_DIR"
fi

if [ ! -d "$IMG_DIR" ]; then
  echo "Error: IMG_DIR does not exist: $IMG_DIR"
  echo "Make sure your input photos are under Work_dir/Input_IMGs/Related_IMGs_4 or update IMG_DIR in run_full_pipeline.sh."
  exit 1
fi

colmap feature_extractor \
  --database_path "$COLMAP_DB/database.db" \
  --image_path "$IMG_DIR"

colmap "$COLMAP_MATCHER"_matcher \
  --database_path "$COLMAP_DB/database.db"

colmap mapper \
  --database_path "$COLMAP_DB/database.db" \
  --image_path "$IMG_DIR" \
  --output_path "$SPARSE_DIR"

if [ ! -d "$SPARSE_DIR/0" ] || [ ! -f "$SPARSE_DIR/0/cameras.txt" ]; then
  echo ""
  echo "COLMAP failed: no sparse model was created (no good initial image pair)."
  echo ""
  echo "Try:"
  echo "  1. Use sequential matching (ordered photos from one scene):"
  echo "     COLMAP_MATCHER=sequential ./run_full_pipeline.sh"
  echo "  2. Use images from the same scene with plenty of overlap."
  echo "  3. Avoid mixing portrait and landscape; use similar resolution."
  echo "  4. Add more images or ensure consistent lighting and focus."
  exit 1
fi

# Optional: undistort for dense pipeline
# colmap image_undistorter \
#   --image_path "$IMG_DIR" \
#   --input_path "$SPARSE_DIR/0" \
#   --output_path "$DENSE_DIR" \
#   --output_type COLMAP

# ---------- Step 2: SimpleRecon (with cache_depths) ----------
echo "=== Step 2: SimpleRecon ==="
SR_ROOT="$(cd "$SIMPLERECON_ROOT" && pwd)"
WEIGHTS_PATH="$SR_ROOT/$SIMPLERECON_WEIGHTS"
if [ ! -f "$WEIGHTS_PATH" ]; then
  echo "Error: SimpleRecon weights not found at: $WEIGHTS_PATH"
  echo "Download hero_model.ckpt from the SimpleRecon README (Google Drive link) and place it in: $SR_ROOT/weights/"
  exit 1
fi
cd "$SR_ROOT"

# You may need to set dataset_path and tuple info in SIMPLERECON_DATA_CONFIG
# to point to your COLMAP output (e.g. sparse/0 and images).
python test.py --name HERO_MODEL \
  --output_base_path "$WORK_DIR/$SIMPLERECON_RESULTS" \
  --config_file configs/models/hero_model.yaml \
  --load_weights_from_checkpoint "$WEIGHTS_PATH" \
  --data_config "$SIMPLERECON_DATA_CONFIG" \
  --num_workers 8 \
  --batch_size 2 \
  --cache_depths \
  --dump_depth_visualization

# ---------- Step 3: Export RGB-D ----------
echo "=== Step 3: Export RGB-D ==="
# Adjust DEPTH_CACHE to match your SimpleRecon output layout:
# output_base_path/name/dataset/frame_tuple_type/depths
DEPTH_CACHE="${WORK_DIR}/${SIMPLERECON_RESULTS}/HERO_MODEL/$(basename "$SIMPLERECON_DATA_CONFIG" .yaml)/default/depths"
if [ ! -d "$DEPTH_CACHE" ]; then
  # Try without "default" or with dense
  DEPTH_CACHE="${WORK_DIR}/${SIMPLERECON_RESULTS}/HERO_MODEL/depths"
fi
if [ ! -d "$DEPTH_CACHE" ]; then
  echo "Warning: Depth cache not found at $DEPTH_CACHE. Find the correct path under $WORK_DIR/$SIMPLERECON_RESULTS and set DEPTH_CACHE."
  exit 1
fi

RGB_DIR="$(cd "$WORK_DIR" && cd "$IMG_DIR" && pwd)"
mkdir -p "$OUT_DIR"

python "$EXPORT_SCRIPT" \
  --cache_dir "$DEPTH_CACHE" \
  --rgb_dir "$RGB_DIR" \
  --out_dir "$OUT_DIR" \
  --depth_format png16mm

echo "Done. Output: $OUT_DIR/previews and $OUT_DIR/rgbd"
