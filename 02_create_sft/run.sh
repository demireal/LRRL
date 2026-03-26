#!/usr/bin/env bash
# =============================================================================
# 02_create_sft/run.sh -- Create SFT datasets for naivetext.
#
# Assumes 01_serialize has already been run.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"

echo "=== Creating SFT datasets from naivetext serializations ==="
python "$SCRIPT_DIR/create_sft.py" \
    --input_dir "$DATA_DIR/serialized/naivetext" \
    --output_dir "$DATA_DIR/sft/naivetext"

echo ""
echo "=== Done ==="
