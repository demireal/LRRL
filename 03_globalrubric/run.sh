#!/usr/bin/env bash
# =============================================================================
# 03_globalrubric/run.sh -- Full global rubric pipeline: cohort -> rubric -> apply -> SFT.
#
# Assumes 01_serialize (naivetext) has been run.
#
# GPU Parallelization (Step 1 - build_cohort):
#   Jobs are dispatched in batches of NUM_GPUS (default: 4).
#   Each job is pinned to a single GPU via CUDA_VISIBLE_DEVICES.
#   15 tasks -> 4 + 4 + 4 + 3 batches.
#
# Configuration:
#   GPUS  -- space-separated GPU IDs (default: "0 1 2 3")
#   TASKS -- override task list (default: all 15)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"
RUBRIC_DIR="${DATA_DIR}/rubric"
LOG_DIR="${DATA_DIR}/logs/rubric"
mkdir -p "$LOG_DIR"

# --- GPU config ---
GPUS=(${GPUS:-0 1 2 3})
NUM_GPUS=${#GPUS[@]}

# --- Tasks ---
# Set NEW_TASKS_ONLY=1 to run only tasks whose names start with "new_".
DEFAULT_TASKS="guo_icu guo_los guo_readmission lab_thrombocytopenia lab_hyperkalemia lab_hypoglycemia lab_hyponatremia lab_anemia new_hypertension new_hyperlipidemia new_pancan new_celiac new_lupus new_acutemi chexpert"
if [ "${NEW_TASKS_ONLY:-0}" = "1" ]; then
    TASKS=($(echo "$DEFAULT_TASKS" | tr ' ' '\n' | grep '^new_'))
else
    TASKS=(${TASKS:-$DEFAULT_TASKS})
fi

echo "GPUs: ${GPUS[*]} (${NUM_GPUS} total)"
echo "Tasks: ${TASKS[*]} (${#TASKS[@]} total)"
echo ""

# =============================================================================
# Generic GPU-parallel dispatcher
# =============================================================================
dispatch_jobs() {
    local -n _jobs=$1
    local total=${#_jobs[@]}
    local batch_num=0

    if [ "$total" -eq 0 ]; then
        echo "  (no jobs to dispatch)"
        return 0
    fi

    echo "  Dispatching $total jobs across ${NUM_GPUS} GPUs"

    for ((i = 0; i < total; i += NUM_GPUS)); do
        batch_num=$((batch_num + 1))
        local pids=()
        local batch_end=$(( i + NUM_GPUS < total ? i + NUM_GPUS : total ))

        for ((j = 0; j < NUM_GPUS && i + j < total; j++)); do
            local gpu=${GPUS[$j]}
            local job_str="${_jobs[$((i + j))]}"
            # shellcheck disable=SC2086
            ( export CUDA_VISIBLE_DEVICES="$gpu"; $job_str ) &
            pids+=($!)
        done

        local fail=0
        for pid in "${pids[@]}"; do
            wait "$pid" || fail=1
        done

        if [ "$fail" -ne 0 ]; then
            echo "ERROR: One or more jobs in batch $batch_num failed. Check logs in $LOG_DIR"
            exit 1
        fi

        echo "  Batch ${batch_num} done (${batch_end}/${total})"
    done
}

# =============================================================================
# Step 1: Build cohorts (k-means + medoid selection) -- GPU-parallelized
# =============================================================================
build_cohort_job() {
    local task="$1"
    local train_path="$DATA_DIR/serialized/naivetext/${task}/train.json"
    local log="$LOG_DIR/build_cohort_${task}.log"

    if [ ! -f "$train_path" ]; then
        echo "  [skip] $task: no train.json" > "$log"
        return 0
    fi

    python "$SCRIPT_DIR/build_cohort.py" \
        --input_dir "$DATA_DIR/serialized/naivetext" \
        --output_dir "$RUBRIC_DIR" \
        --tasks "$task" \
        > "$log" 2>&1
}

echo "=== Step 1: Build cohorts (k-means + medoid selection) ==="
BUILD_COHORT_JOBS=()
for task in "${TASKS[@]}"; do
    BUILD_COHORT_JOBS+=("build_cohort_job $task")
done
dispatch_jobs BUILD_COHORT_JOBS

echo ""
echo "=== Step 2: Generate rubric instructions via GPT-5-mini ==="
python "$SCRIPT_DIR/create_rubric.py" \
    --cohort_dir "$RUBRIC_DIR" \
    --output_dir "$RUBRIC_DIR" \
    --tasks "${TASKS[@]}"

echo ""
echo "=== Step 3: Apply rubric to all patients via GPT-5-mini ==="
python "$SCRIPT_DIR/apply_rubric.py" \
    --rubric_dir "$RUBRIC_DIR" \
    --serialized_dir "$DATA_DIR/serialized/naivetext" \
    --output_dir "$RUBRIC_DIR/rubricified" \
    --tasks "${TASKS[@]}"

echo ""
echo "=== Step 4: Create SFT datasets from rubricified data ==="
python "$SCRIPT_DIR/create_globalrubric_sft.py" \
    --input_dir "$RUBRIC_DIR/rubricified" \
    --output_dir "$DATA_DIR/sft/global-rubric" \
    --tasks "${TASKS[@]}"

echo ""
echo "=== Done. Rubric data in $RUBRIC_DIR ==="
