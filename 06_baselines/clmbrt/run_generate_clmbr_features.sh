#!/bin/bash
# Generate CLMBR-T features for patients in our splits only.
# Uses generate_clmbr_t_features.py with --path_to_data_dir.
# Loads from data/sft/naivetext/{train,val,test}/{task}.json - no external label files needed.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCHMARK_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

# Default paths (EHRSHOT_ASSETS under benchmark repo root, same as ehrshot/bash_scripts/5_generate_clmbr_features.sh)
PATH_TO_DATABASE="${PATH_TO_DATABASE:-${BENCHMARK_ROOT}/EHRSHOT_ASSETS/femr/extract}"
PATH_TO_DATA_DIR="${PATH_TO_DATA_DIR:-${PROJECT_ROOT}}"
SFT_SUBDIR="${SFT_SUBDIR:-data/sft/naivetext}"
PATH_TO_FEATURES_DIR="${PATH_TO_FEATURES_DIR:-${BENCHMARK_ROOT}/EHRSHOT_ASSETS/features}"
PATH_TO_MODELS_DIR="${PATH_TO_MODELS_DIR:-${BENCHMARK_ROOT}/EHRSHOT_ASSETS/models/clmbr}"
BATCH_SIZE="${BATCH_SIZE:-16384}"
USE_GPU="${USE_GPU:-1}"  # Default: use GPU. Set USE_GPU=0 or pass --use_cpu for CPU.
OUTPUT_FILENAME="${OUTPUT_FILENAME:-clmbr_t_features.pkl}"
LOG_FILE="${LOG_FILE:-data/clmbrt/clmbr_feature_gen_$(date +%Y%m%d_%H%M%S).log}"

# Parse overrides from command line
while [[ $# -gt 0 ]]; do
    case $1 in
        --path_to_database) PATH_TO_DATABASE="$2"; shift 2 ;;
        --path_to_models_dir) PATH_TO_MODELS_DIR="$2"; shift 2 ;;
        --path_to_data_dir) PATH_TO_DATA_DIR="$2"; shift 2 ;;
        --sft_subdir) SFT_SUBDIR="$2"; shift 2 ;;
        --path_to_features_dir) PATH_TO_FEATURES_DIR="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --use_cpu) USE_CPU=1; shift ;;
        --use_gpu) USE_GPU=1; shift ;;
        --output_filename) OUTPUT_FILENAME="$2"; shift 2 ;;
        --log_file) LOG_FILE="$2"; shift 2 ;;
        --is_force_refresh) IS_FORCE_REFRESH=1; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "  --path_to_database PATH   (default: EHRSHOT_ASSETS/femr/extract)"
            echo "  --path_to_models_dir PATH (default: EHRSHOT_ASSETS/models/clmbr)"
            echo "  --path_to_data_dir PATH   (default: ehrshot-v2)"
            echo "  --sft_subdir PATH         (default: data/sft/naivetext, same as generate_embeddings)"
            echo "  --path_to_features_dir PATH (default: EHRSHOT_ASSETS/features)"
            echo "  --batch_size INT          (default: 16384 for GPU, 1024 for CPU)"
            echo "  --use_cpu                 Force CPU (slower)"
            echo "  --use_gpu                 Request GPU/CUDA (faster)"
            echo "  --output_filename NAME    (default: clmbr_t_features.pkl)"
            echo "  --is_force_refresh"
            exit 0 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# Validate required paths exist
if [ ! -d "$PATH_TO_DATABASE" ]; then
    echo "ERROR: Database not found: $PATH_TO_DATABASE"
    echo "Set PATH_TO_DATABASE or ensure EHRSHOT_ASSETS/femr/extract exists"
    exit 1
fi
if [ ! -d "$PATH_TO_MODELS_DIR" ] || [ ! -d "${PATH_TO_MODELS_DIR}/clmbr_model" ]; then
    echo "ERROR: Model not found: $PATH_TO_MODELS_DIR (expects clmbr_model, dictionary subdirs)"
    echo "Set PATH_TO_MODELS_DIR or ensure EHRSHOT_ASSETS/models/clmbr exists"
    exit 1
fi

mkdir -p "$PATH_TO_FEATURES_DIR"

# Use EHRSHOT_ENV (has femr/jax) - required for CLMBR subprocesses
for _d in "$HOME/miniconda3/envs/EHRSHOT_ENV" "$CONDA_PREFIX" "$HOME/anaconda3/envs/EHRSHOT_ENV"; do
    if [ -n "$_d" ] && [ -x "${_d}/bin/python" ]; then
        export PATH="${_d}/bin:$PATH"
        echo "Using Python: ${_d}/bin/python"
        break
    fi
done

echo "=========================================="
echo "Generate CLMBR-T features (splits only)"
echo "=========================================="
echo "Database:    ${PATH_TO_DATABASE}"
echo "Data dir:   ${PATH_TO_DATA_DIR}"
echo "Features:   ${PATH_TO_FEATURES_DIR}"
echo "Models:     ${PATH_TO_MODELS_DIR}"
echo "Output:     ${OUTPUT_FILENAME}"
echo "Log file:   ${LOG_FILE}"
echo "Batch size: ${BATCH_SIZE}"
echo "=========================================="

# When GPU requested, clear env vars and set LD_LIBRARY_PATH + PATH for CUDA 11 (EHRSHOT_ENV jaxlib 0.4.7+cuda11)
if [ -z "${USE_CPU}" ] && [ -n "${USE_GPU}" ]; then
    unset CUDA_VISIBLE_DEVICES
    unset JAX_PLATFORMS
    unset CLMBR_USE_CPU
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    CUDA11_BASE="$HOME/.cudnn86/nvidia"
    CUDA11_LIBS="${CUDA11_BASE}/cudnn/lib:${CUDA11_BASE}/cublas/lib:${CUDA11_BASE}/cuda_nvrtc/lib:${CUDA11_BASE}/cuda_runtime/lib"
    if [ -d "${CUDA11_BASE}/cudnn/lib" ]; then
        export LD_LIBRARY_PATH="${CUDA11_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        echo "GPU mode: set LD_LIBRARY_PATH with CUDA 11 libs"
    fi
    PTXAS_DIR="$HOME/.cudnn86/nvidia/cuda_nvcc/bin"
    if [ -d "$PTXAS_DIR" ]; then
        export PATH="${PTXAS_DIR}:${PATH}"
    fi
    echo "GPU mode: cleared CUDA_VISIBLE_DEVICES, JAX_PLATFORMS, CLMBR_USE_CPU"
fi

cd "$PROJECT_ROOT"
python "$SCRIPT_DIR/generate_clmbr_t_features.py" \
    --path_to_database "$PATH_TO_DATABASE" \
    --path_to_data_dir "$PATH_TO_DATA_DIR" \
    --sft_subdir "$SFT_SUBDIR" \
    --log_file "$LOG_FILE" \
    --path_to_features_dir "$PATH_TO_FEATURES_DIR" \
    --path_to_models_dir "$PATH_TO_MODELS_DIR" \
    --output_filename "$OUTPUT_FILENAME" \
    $([ -n "${BATCH_SIZE}" ] && echo "--batch_size ${BATCH_SIZE}") \
    $([ -n "${USE_CPU}" ] && echo --use_cpu) \
    $( [ -z "${USE_CPU}" ] && [ -n "${USE_GPU}" ] && echo --use_gpu) \
    $([ -n "${IS_FORCE_REFRESH}" ] && echo --is_force_refresh)

echo ""
echo "Features written to: ${PATH_TO_FEATURES_DIR}/${OUTPUT_FILENAME}"
echo "Use with evaluate_clmbr_all_tasks.py: --path_to_clmbr_features ${PATH_TO_FEATURES_DIR}/${OUTPUT_FILENAME}"
