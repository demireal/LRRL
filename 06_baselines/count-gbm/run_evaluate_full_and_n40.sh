#!/bin/bash
# Run count-GBM evaluation: n=full first, then n=40.
# Uses same train/val/test splits as CLMBR (data/sft/naivetext/{split}/{task}.json).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCHMARK_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

PATH_TO_COUNT_FEATURES="${PATH_TO_COUNT_FEATURES:-${BENCHMARK_ROOT}/EHRSHOT_ASSETS/features/count_features_naivetext.pkl}"
PATH_TO_DATA_DIR="${PATH_TO_DATA_DIR:-${PROJECT_ROOT}}"
COHORT_DIR="${COHORT_DIR:-${PROJECT_ROOT}/data/rubric}"

PYTHON_BIN="${PYTHON_BIN:-}"
for p in "${HOME}/miniconda3/envs/EHRSHOT_ENV/bin/python" "${HOME}/anaconda3/envs/EHRSHOT_ENV/bin/python"; do
    if [ -x "$p" ]; then PYTHON_BIN="$p"; break; fi
done
[ -z "$PYTHON_BIN" ] && PYTHON_BIN="python"

TASKS=(
    guo_icu guo_los guo_readmission
    lab_thrombocytopenia lab_hyperkalemia lab_hypoglycemia lab_hyponatremia lab_anemia
    new_hypertension new_hyperlipidemia new_pancan new_celiac new_lupus new_acutemi
    chexpert
)

run_eval() {
    local output_dir=$1
    local n_train=$2
    local extra_args=()
    if [ -n "$n_train" ] && [ "$n_train" != "full" ]; then
        extra_args=(--n_train "$n_train" --cohort_dir "$COHORT_DIR")
    fi
    for task in "${TASKS[@]}"; do
        echo "=== $task ==="
        "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_count_all_tasks.py" \
            --task_name "$task" \
            --path_to_count_features "$PATH_TO_COUNT_FEATURES" \
            --path_to_data_dir "$PATH_TO_DATA_DIR" \
            --path_to_output_dir "$output_dir" \
            "${extra_args[@]}" \
            --num_threads 1 || exit 1
    done
}

echo "=========================================="
echo "Count-GBM Eval: n=full, then n=40"
echo "=========================================="
echo "Features: $PATH_TO_COUNT_FEATURES"
echo ""

echo "========== Run 1: n=full =========="
run_eval "${PROJECT_ROOT}/data/results/countgbm_full" ""

echo ""
echo "========== Run 2: n=40 =========="
run_eval "${PROJECT_ROOT}/data/results/countgbm_n40" "40"

echo ""
echo "========== Compute summary metrics =========="
for base in countgbm_full countgbm_n40; do
    out="${PROJECT_ROOT}/data/results/$base"
    meta="${PROJECT_ROOT}/data/results/metrics/$base"
    preds=$(find "$out" -name "predictions.csv" 2>/dev/null | tr '\n' ' ')
    if [ -n "$preds" ]; then
        "${PYTHON_BIN}" "${PROJECT_ROOT}/05_eval/compute_metrics.py" --predictions $preds --output_dir "$meta"
        echo "  $base -> $meta"
    fi
done

echo ""
echo "Done. Results: data/results/countgbm_full, data/results/countgbm_n40"
