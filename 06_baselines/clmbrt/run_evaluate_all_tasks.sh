#!/bin/bash
# Evaluate CLMBR-T model for all 15 tasks using data from data/sft/naivetext/{task_name}.json

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default paths (EHRSHOT_ASSETS/features, same location as run_generate_clmbr_features.sh output)
BENCHMARK_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
PATH_TO_CLMBR_FEATURES="${PATH_TO_CLMBR_FEATURES:-${BENCHMARK_ROOT}/EHRSHOT_ASSETS/features/clmbr_t_features.pkl}"
PATH_TO_DATA_DIR="${PATH_TO_DATA_DIR:-${PROJECT_ROOT}}"
PATH_TO_OUTPUT_DIR="${PATH_TO_OUTPUT_DIR:-${PROJECT_ROOT}/data/results/clmbrt_n40}"
COHORT_DIR="${COHORT_DIR:-${PROJECT_ROOT}/data/rubric}"
EMBEDDINGS_DIR="${EMBEDDINGS_DIR:-}"
NUM_THREADS="${NUM_THREADS:-1}"
# Default n=40. Set N_TRAIN=full or N_TRAIN=all for full train (no cohort filter)
N_TRAIN="${N_TRAIN:-40}"
USE_COHORT=true
if [[ "${N_TRAIN}" == "full" || "${N_TRAIN}" == "all" ]]; then
    USE_COHORT=false
    N_TRAIN=""
fi

# All 15 tasks
TASKS=(
    "guo_icu"
    "guo_los"
    "guo_readmission"
    "lab_thrombocytopenia"
    "lab_hyperkalemia"
    "lab_hypoglycemia"
    "lab_hyponatremia"
    "lab_anemia"
    "new_hypertension"
    "new_hyperlipidemia"
    "new_pancan"
    "new_celiac"
    "new_lupus"
    "new_acutemi"
    "chexpert"
)

echo "=========================================="
echo "CLMBR-T Evaluation for All 15 Tasks"
echo "=========================================="
echo "CLMBR features: ${PATH_TO_CLMBR_FEATURES}"
echo "Data directory: ${PATH_TO_DATA_DIR}"
echo "Output directory: ${PATH_TO_OUTPUT_DIR}"
echo "Cohort directory: ${COHORT_DIR}"
echo "Number of threads: ${NUM_THREADS}"
[ -n "${EMBEDDINGS_DIR}" ] && echo "Labels from embeddings: ${EMBEDDINGS_DIR}"
if [ "$USE_COHORT" = true ] && [ -n "${N_TRAIN}" ]; then
    echo "Training: n=${N_TRAIN} (from ${COHORT_DIR}/{task}/cohort.json)"
else
    echo "Training: all (full train set)"
fi
echo "=========================================="

# Create output directory
mkdir -p "${PATH_TO_OUTPUT_DIR}"

# Use EHRSHOT_ENV python if available; override with PYTHON_BIN env var
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
    for p in "${HOME}/miniconda3/envs/EHRSHOT_ENV/bin/python" "${HOME}/anaconda3/envs/EHRSHOT_ENV/bin/python"; do
        if [ -x "$p" ]; then PYTHON_BIN="$p"; break; fi
    done
fi
[ -z "$PYTHON_BIN" ] && PYTHON_BIN="python"

# Run evaluation for each task
for task in "${TASKS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Evaluating task: ${task}"
    echo "=========================================="
    
    EXTRA_ARGS=()
    [ "$USE_COHORT" = true ] && [ -n "${N_TRAIN}" ] && EXTRA_ARGS+=(--n_train "${N_TRAIN}" --cohort_dir "${COHORT_DIR}")
    [ -n "${EMBEDDINGS_DIR}" ] && EXTRA_ARGS+=(--embeddings_dir "${EMBEDDINGS_DIR}")
    "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_clmbr_all_tasks.py" \
        --task_name "${task}" \
        --path_to_clmbr_features "${PATH_TO_CLMBR_FEATURES}" \
        --path_to_data_dir "${PATH_TO_DATA_DIR}" \
        --path_to_output_dir "${PATH_TO_OUTPUT_DIR}" \
        "${EXTRA_ARGS[@]}" \
        --num_threads "${NUM_THREADS}"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully evaluated ${task}"
    else
        echo "✗ Failed to evaluate ${task}"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "All tasks evaluated successfully!"
echo "=========================================="
echo "Results saved to: ${PATH_TO_OUTPUT_DIR}"

# Step 2: Compute summary metrics (same as run_eval_global-rubric_embeddings_all15.sh)
# Writes to data/results/metrics/<output_dir_basename>/ (e.g. metrics/clmbrt_n40/)
echo ""
echo "=========================================="
echo "Computing summary metrics"
echo "=========================================="
RESULTS_BASENAME="$(basename "${PATH_TO_OUTPUT_DIR}")"
METRICS_OUT="${PROJECT_ROOT}/data/results/metrics/${RESULTS_BASENAME}"
preds=$(find "${PATH_TO_OUTPUT_DIR}" -name "predictions.csv" 2>/dev/null | tr '\n' ' ')
if [ -n "$preds" ]; then
    "${PYTHON_BIN}" "${PROJECT_ROOT}/05_eval/compute_metrics.py" \
        --predictions $preds \
        --output_dir "${METRICS_OUT}"
    echo "  Metrics written to: ${METRICS_OUT}"
else
    echo "  [warn] No predictions.csv found in ${PATH_TO_OUTPUT_DIR}"
fi

echo ""
echo "Done. Per-task: ${PATH_TO_OUTPUT_DIR}"
echo "Summary metrics: ${METRICS_OUT}"
