#!/bin/bash
# Run CLMBR-T evaluation: n=full first, then n=40.
# Prints (task, split): n=samples pos=positives for each run.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Paths
BENCHMARK_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
PATH_TO_CLMBR_FEATURES="${PATH_TO_CLMBR_FEATURES:-${BENCHMARK_ROOT}/EHRSHOT_ASSETS/features/clmbr_t_features.pkl}"
PATH_TO_DATA_DIR="${PATH_TO_DATA_DIR:-${PROJECT_ROOT}}"
COHORT_DIR="${COHORT_DIR:-${PROJECT_ROOT}/data/rubric}"

# Python
PYTHON_BIN="${PYTHON_BIN:-}"
for p in "${HOME}/miniconda3/envs/EHRSHOT_ENV/bin/python" "${HOME}/anaconda3/envs/EHRSHOT_ENV/bin/python"; do
    if [ -x "$p" ]; then PYTHON_BIN="$p"; break; fi
done
[ -z "$PYTHON_BIN" ] && PYTHON_BIN="python"

echo "=========================================="
echo "CLMBR-T Eval: n=full, then n=40"
echo "=========================================="
echo "Features: ${PATH_TO_CLMBR_FEATURES}"
echo ""

# Run 1: n=full (no cohort filter)
echo "========== Run 1: n=full (all train) =========="
PATH_TO_OUTPUT_DIR="${PROJECT_ROOT}/data/results/clmbrt_full" \
N_TRAIN=full \
bash "${SCRIPT_DIR}/run_evaluate_all_tasks.sh"

echo ""
echo "========== Run 2: n=40 (cohort) =========="
PATH_TO_OUTPUT_DIR="${PROJECT_ROOT}/data/results/clmbrt_n40" \
N_TRAIN=40 \
bash "${SCRIPT_DIR}/run_evaluate_all_tasks.sh"

echo ""
echo "=========================================="
echo "Done. Results:"
echo "  n=full: ${PROJECT_ROOT}/data/results/clmbrt_full"
echo "  n=40:   ${PROJECT_ROOT}/data/results/clmbrt_n40"
echo "=========================================="
