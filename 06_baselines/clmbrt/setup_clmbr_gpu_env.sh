#!/bin/bash
# Setup script for clmbr-gpu conda environment.
# Run after: conda activate clmbr-gpu
# Installs femr, JAX with CUDA 11, and dependencies for GPU feature generation.

set -e

echo "=== Setting up clmbr-gpu environment ==="
echo "Python: $(which python)"
echo ""

# Core dependencies (from femr, generate_clmbr_t_features)
pip install --upgrade pip
pip install numpy pandas tqdm loguru

# femr + femr-cuda (matching versions for get_local_attention_data)
pip install femr==0.1.16 femr-cuda==0.1.16

# JAX 0.4.25 (has xla.register_translation; removed in 0.4.33) + CUDA 11 plugin
pip install jax==0.4.25 jaxlib==0.4.25
pip install jax-cuda11-plugin==0.4.25

# CUDA 11 toolkit libs (cuDNN 8.6, nvrtc, cuda_runtime, nvcc for ptxas)
# Plugin may pull some; we ensure full set for compatibility
pip install nvidia-cudnn-cu11==8.6.0.163 nvidia-cuda-nvrtc-cu11==11.8.89 nvidia-cuda-runtime-cu11==11.8.89 nvidia-cuda-nvcc-cu11==11.8.89

# Create symlinks for unversioned .so names (libnvrtc.so, libcudart.so)
CUDA11_NVRTC=$(python -c "import nvidia.cuda_nvrtc; import os; print(os.path.join(os.path.dirname(nvidia.cuda_nvrtc.__file__), 'lib'))" 2>/dev/null)
CUDA11_RUNTIME=$(python -c "import nvidia.cuda_runtime; import os; print(os.path.join(os.path.dirname(nvidia.cuda_runtime.__file__), 'lib'))" 2>/dev/null)
if [ -n "$CUDA11_NVRTC" ] && [ -d "$CUDA11_NVRTC" ] && [ ! -L "$CUDA11_NVRTC/libnvrtc.so" ]; then
    ln -sf libnvrtc.so.11.2 "$CUDA11_NVRTC/libnvrtc.so" 2>/dev/null || true
fi
if [ -n "$CUDA11_RUNTIME" ] && [ -d "$CUDA11_RUNTIME" ] && [ ! -L "$CUDA11_RUNTIME/libcudart.so" ]; then
    ln -sf libcudart.so.11.0 "$CUDA11_RUNTIME/libcudart.so" 2>/dev/null || true
fi

echo ""
echo "=== Verifying setup ==="
python -c "
import jax
import jaxlib
print('JAX:', jax.__version__, '| jaxlib:', jaxlib.__version__)
print('Devices:', jax.devices())
print('Backend:', jax.default_backend())
import femr.extension.jax as ext
print('femr.extension.jax has get_local_attention_data:', hasattr(ext, 'get_local_attention_data'))
from femr.models.scripts import compute_representations
print('femr compute_representations: OK')
print('SUCCESS')
" || { echo "Verification failed. Try --use_cpu as fallback."; exit 1; }

echo ""
echo "=== Done. Run: bash run_generate_clmbr_features.sh --is_force_refresh ==="
