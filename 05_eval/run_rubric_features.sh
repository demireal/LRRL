#!/usr/bin/env bash
# =============================================================================
# 05_eval/run_rubric_features.sh -- Rubric feature pipeline: featurizers + eval.
#
# 1. Run featurizers (03_globalrubric/feature_extractors/) to produce .npz
# 2. Evaluate rubric features with LR + XGBoost (eval_rubric_features.py)
#
# Prerequisite: auto rubric pipeline must have been run first
# (03_globalrubric/run_auto.sh or the full pipeline).
#
# Configuration (env):
#   TASKS          -- override task list (default: all 15)
#   VARIANT        -- "auto" or "auto_plus" (default: auto)
#   AZURE_CONFIG   -- path to GPT-5.2 Azure config for generating featurizers
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/.."
DATA_DIR="${PROJECT_DIR}/data"
RUBRIC_DIR="${DATA_DIR}/rubric"
GLOBALRUBRIC_DIR="${PROJECT_DIR}/03_globalrubric"

# --- Configuration ---
VARIANT="${VARIANT:-auto}"
AZURE_CONFIG="${AZURE_CONFIG:-${PROJECT_DIR}/config/azure_config_gpt52.json}"

if [ "$VARIANT" = "auto_plus" ]; then
    PARSERS_DIR="${GLOBALRUBRIC_DIR}/auto_parsers_plus"
    SFT_DIR="${DATA_DIR}/sft/llmrubric-auto-plus"
    FEATURES_DIR="${DATA_DIR}/rubric_features_auto_plus"
    FEATURIZERS_DIR="${GLOBALRUBRIC_DIR}/feature_extractors_plus"
    RESULTS_DIR="${DATA_DIR}/results/rubric_features_auto_plus"
    METRICS_DIR="${DATA_DIR}/results/metrics/rubric_features_auto_plus_full"
else
    PARSERS_DIR="${GLOBALRUBRIC_DIR}/auto_parsers"
    SFT_DIR="${DATA_DIR}/sft/llmrubric-auto"
    FEATURES_DIR="${DATA_DIR}/rubric_features_auto"
    FEATURIZERS_DIR="${GLOBALRUBRIC_DIR}/feature_extractors"
    RESULTS_DIR="${DATA_DIR}/results/rubric_features_auto"
    METRICS_DIR="${DATA_DIR}/results/metrics/rubric_features_auto_full"
fi

# --- Tasks ---
DEFAULT_TASKS="guo_icu guo_los guo_readmission lab_thrombocytopenia lab_hyperkalemia lab_hypoglycemia lab_hyponatremia lab_anemia new_hypertension new_hyperlipidemia new_pancan new_celiac new_lupus new_acutemi chexpert"
TASKS=(${TASKS:-$DEFAULT_TASKS})

echo "=== Rubric Feature Pipeline (variant=${VARIANT}) ==="
echo "Tasks: ${TASKS[*]} (${#TASKS[@]} total)"
echo ""

# =============================================================================
# Step 0 (optional): Generate featurizer scripts if they don't exist
# =============================================================================
NEED_FEATURIZERS=0
for task in "${TASKS[@]}"; do
    [ -f "$FEATURIZERS_DIR/${task}_featurizer.py" ] || { NEED_FEATURIZERS=1; break; }
done

if [ "$NEED_FEATURIZERS" -eq 1 ]; then
    echo "=== Step 0: Generate featurizer scripts via GPT-5.2 ==="
    python "$GLOBALRUBRIC_DIR/create_feature_extractor.py" \
        --parsers_dir "$PARSERS_DIR" \
        --cohort_dir "$RUBRIC_DIR" \
        --sft_dir "$SFT_DIR" \
        --output_dir "$FEATURIZERS_DIR" \
        --azure_config "$AZURE_CONFIG" \
        --tasks "${TASKS[@]}"
    echo ""
fi

# =============================================================================
# Step 1: Run featurizers for all tasks
# =============================================================================
echo "=== Step 1: Run featurizers for all tasks ==="
for task in "${TASKS[@]}"; do
    if [ -f "$FEATURES_DIR/$task/train.npz" ] && \
       [ -f "$FEATURES_DIR/$task/val.npz" ] && \
       [ -f "$FEATURES_DIR/$task/test.npz" ]; then
        echo "  Skipping $task (features already exist)"
    else
        echo "  $task ..."
        python3 "$FEATURIZERS_DIR/${task}_featurizer.py" \
            --input_dir "$SFT_DIR" \
            --output_dir "$FEATURES_DIR" \
            --task "$task" \
            --splits train val test
    fi
done

# =============================================================================
# Step 2: Evaluate rubric features with LR + XGBoost
# =============================================================================
echo ""
echo "=== Step 2: Evaluate rubric features (LR + XGBoost) ==="
python "$SCRIPT_DIR/eval_rubric_features.py" \
    --features_dir "$FEATURES_DIR" \
    --output_dir "$RESULTS_DIR" \
    --metrics_dir "$METRICS_DIR" \
    --cohort_dir "$RUBRIC_DIR" \
    --tasks "${TASKS[@]}"

echo ""
echo "=== Done. Rubric feature pipeline complete (variant=${VARIANT}) ==="
echo "Features:    ${FEATURES_DIR}"
echo "Results:     ${RESULTS_DIR}"
echo "Metrics:     ${METRICS_DIR}"
