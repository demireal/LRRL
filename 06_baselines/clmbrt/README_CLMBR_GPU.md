# CLMBR-T GPU Setup

Fresh conda environment for CLMBR-T feature generation with GPU support.

## Quick Start

```bash
cd 06_baselines/clmbrt
bash install_clmbr_gpu.sh
```

If conda fails with `CondaVerificationError` (corrupted cache), run `conda clean --all` and retry.

This creates the `clmbr-gpu` environment and installs:
- femr 0.1.16 + femr-cuda 0.1.16 (matching versions)
- JAX 0.4.25 + jax-cuda11-plugin 0.4.25 (CUDA 11 GPU backend)
- CUDA 11 toolkit libs (cuDNN 8.6, nvrtc, cuda_runtime, nvcc)

## Run Feature Generation

```bash
conda activate clmbr-gpu
bash run_generate_clmbr_features.sh --is_force_refresh
```

The run script automatically prefers `clmbr-gpu` when it exists (over EHRSHOT_ENV).

## Manual Setup (alternative)

```bash
conda env create -f environment_clmbr_gpu.yml
conda activate clmbr-gpu
bash setup_clmbr_gpu_env.sh
```

## Alternative: Clone from EHRSHOT_ENV (if conda create fails)

```bash
conda create --clone EHRSHOT_ENV -n clmbr-gpu
conda activate clmbr-gpu
pip install femr==0.1.16 femr-cuda==0.1.16  # upgrade to matching versions
pip install jax==0.4.25 jaxlib==0.4.25 jax-cuda11-plugin==0.4.25
pip install nvidia-cudnn-cu11==8.6.0.163 nvidia-cuda-nvrtc-cu11==11.8.89 nvidia-cuda-runtime-cu11==11.8.89 nvidia-cuda-nvcc-cu11==11.8.89
# Create symlinks: ln -sf libnvrtc.so.11.2 $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so
#                 ln -sf libcudart.so.11.0 $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so
```

## Requirements

- CUDA 11 or 12 driver (GPU)
- EHRSHOT_ASSETS with femr extract, CLMBR model, and data
