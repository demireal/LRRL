#!/usr/bin/env bash
# =============================================================================
# 03_globalrubric/run_auto.sh -- Auto rubric pipeline: generate parsers,
# apply them, create SFT, run featurizers.
#
# This script orchestrates the "auto" rubric path:
#   1. Generate deterministic parser scripts via GPT-5.2 (create_rubric_auto.py)
#   2. Run the generated parsers on all patients (fill_rubric)
#   3. Create SFT datasets from rubricified data (create_globalrubric_sft.py)
#
# Prerequisite: 03_globalrubric/run.sh must have been run first (to create
# cohorts and rubrics via build_cohort.py + create_rubric.py).
#
# Configuration:
#   TASKS          -- override task list (default: all 15)
#   AZURE_CONFIG   -- path to GPT-5.2 Azure config (default: config/azure_config_gpt52.json)
#   VARIANT        -- "auto" or "auto_plus" (default: auto)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/.."
DATA_DIR="${PROJECT_DIR}/data"
RUBRIC_DIR="${DATA_DIR}/rubric"

# --- Configuration ---
VARIANT="${VARIANT:-auto}"
AZURE_CONFIG="${AZURE_CONFIG:-${PROJECT_DIR}/config/azure_config_gpt52.json}"

if [ "$VARIANT" = "auto_plus" ]; then
    PARSERS_DIR="${SCRIPT_DIR}/auto_parsers_plus"
    RUBRICIFIED_DIR="${DATA_DIR}/llmrubric-auto-plus/rubricified"
    SFT_DIR="${DATA_DIR}/sft/llmrubric-auto-plus"
    GENERATOR_SCRIPT="create_rubric_auto_plus.py"
else
    PARSERS_DIR="${SCRIPT_DIR}/auto_parsers"
    RUBRICIFIED_DIR="${DATA_DIR}/llmrubric-auto/rubricified"
    SFT_DIR="${DATA_DIR}/sft/llmrubric-auto"
    GENERATOR_SCRIPT="create_rubric_auto.py"
fi

# --- Tasks ---
DEFAULT_TASKS="guo_icu guo_los guo_readmission lab_thrombocytopenia lab_hyperkalemia lab_hypoglycemia lab_hyponatremia lab_anemia new_hypertension new_hyperlipidemia new_pancan new_celiac new_lupus new_acutemi chexpert"
TASKS=(${TASKS:-$DEFAULT_TASKS})

echo "=== Auto Rubric Pipeline (variant=${VARIANT}) ==="
echo "Tasks: ${TASKS[*]} (${#TASKS[@]} total)"
echo "Parsers dir: ${PARSERS_DIR}"
echo ""

# =============================================================================
# Step 1: Generate parser scripts via GPT-5.2
# =============================================================================
echo "=== Step 1: Generate parser scripts via GPT-5.2 ==="
GENERATOR_ARGS=(
    --cohort_dir "$RUBRIC_DIR"
    --rubric_dir "$RUBRIC_DIR"
    --output_dir "$PARSERS_DIR"
    --azure_config "$AZURE_CONFIG"
    --tasks "${TASKS[@]}"
)

if [ "$VARIANT" = "auto_plus" ]; then
    GENERATOR_ARGS+=(--rubricified_dir "$RUBRIC_DIR/rubricified")
fi

python "$SCRIPT_DIR/$GENERATOR_SCRIPT" "${GENERATOR_ARGS[@]}"

# =============================================================================
# Step 2: Apply auto-parsers (fill_rubric) to all patients
# =============================================================================
echo ""
echo "=== Step 2: Apply auto-parsers to all patients ==="
for task in "${TASKS[@]}"; do
    if [ -f "$RUBRICIFIED_DIR/$task/train.json" ] && \
       [ -f "$RUBRICIFIED_DIR/$task/val.json" ] && \
       [ -f "$RUBRICIFIED_DIR/$task/test.json" ]; then
        echo "  Skipping $task (rubricified already exists)"
    else
        echo "  $task ..."
        python3 "$PARSERS_DIR/${task}_parser.py" \
            --input_dir "$DATA_DIR/serialized/naivetext" \
            --output_dir "$RUBRICIFIED_DIR" \
            --task "$task" \
            --splits train val test
    fi
done

# =============================================================================
# Step 3: Create SFT datasets from rubricified data
# =============================================================================
echo ""
echo "=== Step 3: Create SFT datasets ==="
python "$SCRIPT_DIR/create_globalrubric_sft.py" \
    --input_dir "$RUBRICIFIED_DIR" \
    --output_dir "$SFT_DIR" \
    --tasks "${TASKS[@]}"

echo ""
echo "=== Done. Auto rubric pipeline complete (variant=${VARIANT}) ==="
echo "Parsers:     ${PARSERS_DIR}"
echo "Rubricified: ${RUBRICIFIED_DIR}"
echo "SFT:         ${SFT_DIR}"
