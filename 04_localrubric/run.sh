#!/usr/bin/env bash
# =============================================================================
# 04_localrubric/run.sh -- Generate local rubric representations.
#
# Assumes 02_create_sft (naivetext) has been run.
# Edit --tasks to select which tasks to generate traces for.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"

# By default, generate for all 15 tasks. Override with e.g.:
#   TASKS="guo_icu new_acutemi" bash 04_localrubric/run.sh
TASKS="${TASKS:-guo_icu guo_los guo_readmission lab_thrombocytopenia lab_hyperkalemia lab_hypoglycemia lab_hyponatremia lab_anemia new_hypertension new_hyperlipidemia new_pancan new_celiac new_lupus new_acutemi chexpert}"

echo "=== Generating local rubric representations (train + val + test) ==="
python "$SCRIPT_DIR/generate_local_rubric.py" \
    --sft_dir "$DATA_DIR/sft/naivetext" \
    --output_dir "$DATA_DIR/sft/local-rubric" \
    --tasks $TASKS \
    --splits train val test

echo ""
echo "=== Done ==="
