#!/bin/bash
# One-shot script to create and setup the clmbr-gpu conda environment.
# Run from 06_baselines/clmbrt/: bash install_clmbr_gpu.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Find conda
CONDA=""
for c in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" "$(which conda 2>/dev/null)"; do
    [ -x "$c" ] && CONDA="$c" && break
done
[ -z "$CONDA" ] && { echo "ERROR: conda not found"; exit 1; }

echo "=== Creating conda environment clmbr-gpu ==="
if "$CONDA" env list | grep -qw "clmbr-gpu"; then
    echo "Environment clmbr-gpu already exists. Running setup..."
else
    # If conda create fails (e.g. corrupted pkgs cache), try: conda clean --all
    "$CONDA" env create -f environment_clmbr_gpu.yml -y || {
        echo "ERROR: conda env create failed. Try: conda clean --all, then re-run."
        exit 1
    }
fi

# Activate and run setup
echo "=== Installing packages ==="
source "$(dirname "$CONDA")/../etc/profile.d/conda.sh"
conda activate clmbr-gpu

# Run the pip installs from setup script
pip install --upgrade pip
pip install numpy pandas tqdm loguru
pip install femr==0.1.16 femr-cuda==0.1.16
pip install jax==0.4.25 jaxlib==0.4.25
pip install jax-cuda11-plugin==0.4.25
pip install nvidia-cudnn-cu11==8.6.0.163 nvidia-cuda-nvrtc-cu11==11.8.89 nvidia-cuda-runtime-cu11==11.8.89 nvidia-cuda-nvcc-cu11==11.8.89

# Create symlinks for unversioned libs
CUDA11_BASE=$(python -c "
try:
    import nvidia.cuda_nvrtc as m
    import os
    print(os.path.join(os.path.dirname(m.__file__), 'lib'))
except: print('')
")
if [ -n "$CUDA11_BASE" ] && [ -d "$CUDA11_BASE" ]; then
    (cd "$CUDA11_BASE" && ln -sf libnvrtc.so.11.2 libnvrtc.so 2>/dev/null || true)
fi
CUDA11_RT=$(python -c "
try:
    import nvidia.cuda_runtime as m
    import os
    print(os.path.join(os.path.dirname(m.__file__), 'lib'))
except: print('')
")
if [ -n "$CUDA11_RT" ] && [ -d "$CUDA11_RT" ]; then
    (cd "$CUDA11_RT" && ln -sf libcudart.so.11.0 libcudart.so 2>/dev/null || true)
fi

echo ""
echo "=== Verifying ==="
unset CUDA_VISIBLE_DEVICES
python -c "
import jax
import jaxlib
print('JAX:', jax.__version__, 'jaxlib:', jaxlib.__version__)
print('Devices:', jax.devices())
print('Backend:', jax.default_backend())
import femr.extension.jax as ext
assert hasattr(ext, 'get_local_attention_data'), 'femr-cuda missing get_local_attention_data'
from femr.models.scripts import compute_representations
print('femr: OK')
import jax.numpy as jnp
x = jnp.ones((2,2))
print('GPU compute:', float((x@x)[0,0]))
print('SUCCESS')
" || { echo "Verification failed."; exit 1; }

echo ""
echo "Done. To run CLMBR feature generation:"
echo "  conda activate clmbr-gpu"
echo "  bash run_generate_clmbr_features.sh --is_force_refresh"
