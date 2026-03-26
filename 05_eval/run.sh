#!/usr/bin/env bash
# =============================================================================
# 05_eval/run.sh -- Run embedding-based evaluation.
#
# Generates embeddings and evaluates via logistic regression.
#
# To run directly:
#   bash 05_eval/run_embeddings.sh
#
# Configuration (env): GPUS, TASKS, SEED (passed through to the subscripts).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"
RESULTS_DIR="${DATA_DIR}/results"

echo "=== Running embedding evaluation ==="
echo ""

bash "$SCRIPT_DIR/run_embeddings.sh"
echo ""

# Aggregate metrics across all result dirs (idempotent)
echo "=== Computing aggregate metrics (all result dirs) ==="
for dir in "$RESULTS_DIR"/*/; do
    [ -d "$dir" ] || continue
    name=$(basename "$dir")
    [ "$name" = "metrics" ] && continue
    preds=$(find "$dir" -name "predictions.csv" 2>/dev/null | tr '\n' ' ')
    if [ -n "$preds" ]; then
        echo "  [metrics] $name"
        # shellcheck disable=SC2086
        python "$SCRIPT_DIR/compute_metrics.py" \
            --predictions $preds \
            --output_dir "$RESULTS_DIR/metrics/$name"
    fi
done

echo ""
echo "=== All evaluation complete. Results in $RESULTS_DIR ==="
echo "=== Logs in $DATA_DIR/logs/eval ==="
