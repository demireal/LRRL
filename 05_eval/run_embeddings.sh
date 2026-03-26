#!/usr/bin/env bash
# =============================================================================
# 05_eval/run_embeddings.sh -- Embedding-based evaluation.
#
# 1. Generate embeddings (GPU) for each representation; skip if already present.
# 2. Evaluate embeddings via logistic regression (CPU): full and n=40.
# 3. Compute metrics for embedding result dirs.
#
# Edit REPRESENTATIONS below to choose which representations to run.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"
RESULTS_DIR="${DATA_DIR}/results"
LOG_DIR="${DATA_DIR}/logs/eval"
mkdir -p "$LOG_DIR"

# --- Edit this: which representations to generate and evaluate ---
REPRESENTATIONS=(naivetext global-rubric local-rubric)

# --- Tasks and GPU config (override via env if needed) ---
DEFAULT_TASKS="guo_icu guo_los guo_readmission lab_thrombocytopenia lab_hyperkalemia lab_hypoglycemia lab_hyponatremia lab_anemia new_hypertension new_hyperlipidemia new_pancan new_celiac new_lupus new_acutemi chexpert"
TASKS=(${TASKS:-$DEFAULT_TASKS})
GPUS=(${GPUS:-0 1 2 3})
NUM_GPUS=${#GPUS[@]}
export PYTHONHASHSEED="${SEED:-42}"

echo "=== Embedding evaluation ==="
echo "Representations: ${REPRESENTATIONS[*]}"
echo "Tasks: ${#TASKS[@]}"
echo ""

# -----------------------------------------------------------------------------
# Dispatcher
# -----------------------------------------------------------------------------
dispatch_jobs() {
    local -n _jobs=$1
    local total=${#_jobs[@]}
    [ "$total" -eq 0 ] && return 0
    echo "  Dispatching $total jobs across ${NUM_GPUS} GPUs"
    for ((i = 0; i < total; i += NUM_GPUS)); do
        local pids=()
        local batch_end=$(( i + NUM_GPUS < total ? i + NUM_GPUS : total ))
        for ((j = 0; j < NUM_GPUS && i + j < total; j++)); do
            local gpu=${GPUS[$j]}
            local job_str="${_jobs[$((i + j))]}"
            ( export CUDA_VISIBLE_DEVICES="$gpu"; $job_str ) &
            pids+=($!)
        done
        local fail=0
        for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
        [ "$fail" -ne 0 ] && { echo "ERROR: Job failed. Check $LOG_DIR"; exit 1; }
        echo "  Batch done (${batch_end}/${total})"
    done
}

# -----------------------------------------------------------------------------
# 1. Generate embeddings (skip if all train/val/test .npz exist per task)
# -----------------------------------------------------------------------------
gen_embed_job() {
    local repr="$1"
    local out_dir="$DATA_DIR/embeddings/${repr}"
    local log="$LOG_DIR/embed_${repr}.log"
    local skip=1
    for task in "${TASKS[@]}"; do
        for split in train val test; do
            [ -f "${out_dir}/${task}/${split}.npz" ] || { skip=0; break 2; }
        done
    done
    if [ "$skip" -eq 1 ]; then
        echo "  [skip] embeddings ${repr} (all tasks have train/val/test .npz)" > "$log"
        return 0
    fi
    python "$SCRIPT_DIR/generate_embeddings.py" \
        --sft_dir "$DATA_DIR/sft/${repr}" \
        --output_dir "$out_dir" \
        > "$log" 2>&1
}

echo "=== Generating embeddings ==="
EMBED_JOBS=()
for repr in "${REPRESENTATIONS[@]}"; do EMBED_JOBS+=("gen_embed_job $repr"); done
dispatch_jobs EMBED_JOBS

# -----------------------------------------------------------------------------
# 2. Evaluate embeddings (full and n=40)
# -----------------------------------------------------------------------------
echo ""
echo "=== Evaluating embeddings (n=full) ==="
for repr in "${REPRESENTATIONS[@]}"; do
    echo "  [eval_embeddings] ${repr}"
    python "$SCRIPT_DIR/eval_embeddings.py" \
        --embeddings_dir "$DATA_DIR/embeddings/${repr}" \
        --output_dir "$RESULTS_DIR/embedding_${repr}_full"
done

echo ""
echo "=== Evaluating embeddings (n=40) ==="
for repr in "${REPRESENTATIONS[@]}"; do
    for task in "${TASKS[@]}"; do
        python "$SCRIPT_DIR/eval_embeddings.py" \
            --embeddings_dir "$DATA_DIR/embeddings/${repr}" \
            --output_dir "$RESULTS_DIR/embedding_${repr}_n40" \
            --tasks "$task" \
            --n_train 40 \
            --cohort_file "$DATA_DIR/rubric/${task}/cohort.json"
    done
done

# -----------------------------------------------------------------------------
# 3. Compute metrics for embedding result dirs
# -----------------------------------------------------------------------------
echo ""
echo "=== Computing metrics (embedding results) ==="
for repr in "${REPRESENTATIONS[@]}"; do
    for suffix in full n40; do
        dir="$RESULTS_DIR/embedding_${repr}_${suffix}"
        [ -d "$dir" ] || continue
        preds=$(find "$dir" -name "predictions.csv" 2>/dev/null | tr '\n' ' ')
        [ -n "$preds" ] || continue
        echo "  [metrics] embedding_${repr}_${suffix}"
        # shellcheck disable=SC2086
        python "$SCRIPT_DIR/compute_metrics.py" \
            --predictions $preds \
            --output_dir "$RESULTS_DIR/metrics/embedding_${repr}_${suffix}"
    done
done

echo ""
echo "Done. Results: $RESULTS_DIR/embedding_*"
