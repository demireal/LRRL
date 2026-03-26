#!/usr/bin/env python3
"""
Generate CLMBR/CLMBR-T features for patients in our splits or from external label dirs.

Modes:
  1. --path_to_data_dir: Load (patient_id, label_time, label_value) from
     data/sft/naivetext/{train,val,test}/{task}.json for all 15 tasks.
     Only generates features for patients in our splits.

  2. --path_to_labels_dir: Load from external labeled_patients.csv files
     (legacy: {path}/{task}/labeled_patients.csv for acute_mi, hyperlipidemia, etc.)

The output pickle file will contain a dict with:
- 'data_matrix': Feature matrix (n_samples, n_features)
- 'patient_ids': Patient IDs (n_samples,)
- 'labeling_time': Labeling times (n_samples,)
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from loguru import logger

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Change to project root directory for imports
os.chdir(project_root)

# Import from ehrshot package and femr
from femr.labelers import load_labeled_patients

from config.tasks import ALL_TASK_NAMES

# Task name mappings (internal task names to dataset names) for legacy label dirs
TASK_MAPPINGS = {
    'acute_mi': 'new_acutemi',
    'hyperlipidemia': 'new_hyperlipidemia',
    'hypertension': 'new_hypertension',
    'pancreatic_cancer': 'new_pancan'
}

# Default list of tasks for legacy mode
DEFAULT_TASKS = ['acute_mi', 'hyperlipidemia', 'hypertension', 'pancreatic_cancer']

# Python with femr/jax for CLMBR subprocesses (avoids ModuleNotFoundError when run outside env)
# Set CLMBR_PYTHON env var to override, or ensure EHRSHOT_ENV conda env exists.


def _get_clmbr_python() -> str:
    """Python executable with femr/jax for CLMBR subprocesses."""
    for path in [
        os.environ.get("CLMBR_PYTHON", ""),
        os.path.expanduser("~/miniconda3/envs/EHRSHOT_ENV/bin/python"),
        (os.environ.get("CONDA_PREFIX", "") + "/bin/python") if os.environ.get("CONDA_PREFIX") else "",
        sys.executable,
    ]:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return sys.executable


def _binarize_label(task: str, raw_value) -> bool:
    """Convert raw label to bool. Same logic as 01_serialize/serialize.py _parse_label_value.

    Some tasks have non-binary raw labels:
    - lab_*: value is 0/1/2/3 (categorical). Only 1 = abnormal (True).
    - chexpert: value is int bitmask. 8192 = no finding (False); any other = finding (True).
    - guo_*, new_*: value is "True"/"False" or bool.
    """
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, (int, float)):
        raw_str = str(int(raw_value))
    else:
        raw_str = str(raw_value).strip()
    if task.startswith("lab_"):
        return int(raw_str) == 1
    if task == "chexpert":
        return int(raw_str) != 8192
    if raw_str in ("True", "False"):
        return raw_str == "True"
    return raw_str == "True" or raw_value in (True, 1, "true", "yes", "Yes")


def load_labels_from_naivetext_splits(
    path_to_data_dir: str,
    sft_subdir: str = "data/sft/naivetext",
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load labels from data/sft/naivetext/{split}/{task_name}.json.

    Each record has patient_id, label_time, label_value (ground truth). No dedup needed.
    """
    from tqdm import tqdm

    naivetext_root = Path(path_to_data_dir) / sft_subdir
    patient_ids: List[int] = []
    label_values: List[bool] = []
    label_times: List[str] = []

    files = []
    for split in ["train", "val", "test"]:
        split_dir = naivetext_root / split
        if not split_dir.exists():
            continue
        for task in ALL_TASK_NAMES:
            path = split_dir / f"{task}.json"
            if path.exists():
                files.append((split, task, path))

    for _split, _task, path in tqdm(files, desc="Loading labels"):
        with open(path) as f:
            data = json.load(f)
        for r in data:
            patient_ids.append(int(r["patient_id"]))
            label_values.append(bool(r["label_value"]))
            lt = r.get("label_time", r.get("prediction_time", ""))
            label_times.append(str(lt))

    if not patient_ids:
        return {}

    return {
        "naivetext_splits": (
            np.array(patient_ids),
            np.array(label_values, dtype=bool),
            np.array(label_times, dtype=object),
        )
    }


def load_all_task_labels(path_to_labels_dir: str) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load labels for all tasks.
    
    Returns:
        Dictionary mapping task names to (patient_ids, label_values, label_times) tuples
    """
    all_labels = {}
    
    for task_name in DEFAULT_TASKS:
        task_dir = TASK_MAPPINGS.get(task_name, task_name)
        label_file = os.path.join(path_to_labels_dir, task_dir, 'labeled_patients.csv')
        
        if not os.path.exists(label_file):
            logger.warning(f"Label file not found for {task_name}: {label_file}, skipping")
            continue
        
        labeled_patients = load_labeled_patients(label_file)
        patient_ids, label_values, label_times = labeled_patients.as_numpy_arrays()
        
        all_labels[task_name] = (patient_ids, label_values, label_times)
        
        logger.info(f"Loaded labels for {task_name}:")
        logger.info(f"  {len(patient_ids)} labels")
        logger.info(f"  Positive: {np.sum(label_values)}/{len(label_values)} ({np.mean(label_values)*100:.1f}%)")
    
    return all_labels


def create_combined_labeled_patients_csv(
    all_labels: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_path: str
) -> str:
    """Create a combined labeled_patients.csv file from all tasks.
    
    Format matches what clmbr_create_batches expects:
    - Columns: patient_id, prediction_time, value, label_type
    
    Returns:
        Path to the created CSV file
    """
    logger.info(f"Creating combined labeled_patients.csv from all tasks...")
    
    # Combine all labels; apply _binarize_label for legacy (raw lab/chexpert values)
    all_rows = []
    for task_name, (patient_ids, label_values, label_times) in all_labels.items():
        for i in range(len(patient_ids)):
            time_val = label_times[i]
            if isinstance(time_val, (str, np.str_)):
                prediction_time = str(time_val)
            elif isinstance(time_val, np.datetime64):
                dt = pd.Timestamp(time_val).to_pydatetime()
                dt = dt.replace(second=0, microsecond=0)
                prediction_time = dt.isoformat()
            elif hasattr(time_val, 'isoformat'):
                dt = time_val.replace(second=0, microsecond=0) if hasattr(time_val, 'replace') else time_val
                prediction_time = dt.isoformat()
            else:
                prediction_time = str(time_val)

            raw_val = label_values[i]
            if isinstance(raw_val, np.floating):
                raw_val = float(raw_val)
            elif isinstance(raw_val, np.integer):
                raw_val = int(raw_val)
            all_rows.append({
                'patient_id': int(patient_ids[i]),
                'prediction_time': prediction_time,
                'value': _binarize_label(task_name, raw_val),
                'label_type': 'boolean'
            })
    
    df = pd.DataFrame(all_rows, columns=['patient_id', 'prediction_time', 'value', 'label_type'])
    df.to_csv(output_path, index=False)
    logger.info(f"Created combined labeled_patients.csv with {len(df)} labels at: {output_path}")
    return output_path


def compute_clmbr_t_features(
    path_to_database: str,
    path_to_model_dir: str,
    all_labels: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    path_to_features_dir: str,
    is_force_refresh: bool = False,
    batch_size: int = None,
    use_cpu: bool = False,
    use_gpu: bool = False,
) -> Dict[str, np.ndarray]:
    """Compute CLMBR-T features for all tasks using command-line tools (like 5_generate_clmbr_features.py).
    
    Returns:
        Dictionary with keys: 'data_matrix', 'patient_ids', 'labeling_time'
    """
    import shutil
    import tempfile

    # Explicit --use_gpu takes priority over CLMBR_USE_CPU env var.
    # Only fall back to CLMBR_USE_CPU when neither --use_cpu nor --use_gpu was passed.
    env_wants_cpu = os.environ.get("CLMBR_USE_CPU", "").lower() in ("1", "true", "yes")
    if use_gpu:
        # Explicit GPU request overrides everything
        use_cpu = False
    elif use_cpu or (env_wants_cpu and not use_gpu):
        use_cpu = True
        use_gpu = False
    if use_gpu:
        # Clear any CPU-forcing env so subprocess sees GPU
        os.environ.pop("JAX_PLATFORMS", None)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        # jaxlib 0.4.7+cuda11 in EHRSHOT_ENV needs cuDNN 8.6 + CUDA 11 runtime (from ~/.cudnn86)
        cuda11_base = os.path.expanduser("~/.cudnn86/nvidia")
        cuda11_lib_dirs = [
            os.path.join(cuda11_base, "cudnn", "lib"),
            os.path.join(cuda11_base, "cublas", "lib"),
            os.path.join(cuda11_base, "cuda_nvrtc", "lib"),
            os.path.join(cuda11_base, "cuda_runtime", "lib"),
        ]
        extra_ld = ":".join(p for p in cuda11_lib_dirs if os.path.isdir(p))
        if extra_ld:
            cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = f"{extra_ld}:{cur_ld}" if cur_ld else extra_ld
            logger.info(f"Added cuDNN 8.6 libs to LD_LIBRARY_PATH: {extra_ld}")
        logger.info("Requesting GPU execution (--use_gpu)")
    if use_cpu:
        os.environ['JAX_PLATFORMS'] = 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        logger.info("Forcing CPU execution (CLMBR_USE_CPU=1)")
    
    # Check if model directory exists
    if not os.path.exists(path_to_model_dir):
        raise FileNotFoundError(f"Model directory not found: {path_to_model_dir}")
    
    # Find the model and dictionary paths
    # Assuming the model directory structure is: {path_to_model_dir}/clmbr_model and {path_to_model_dir}/dictionary
    path_to_model = os.path.join(path_to_model_dir, "clmbr_model")
    path_to_dictionary = os.path.join(path_to_model_dir, "dictionary")
    
    if not os.path.exists(path_to_model):
        raise FileNotFoundError(f"Model not found: {path_to_model}")
    if not os.path.exists(path_to_dictionary):
        raise FileNotFoundError(f"Dictionary not found: {path_to_dictionary}")
    
    # Create temporary directory for batches and combined labels
    temp_dir = os.path.join(path_to_features_dir, "clmbr_t_temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create combined labeled_patients.csv
    combined_labels_path = os.path.join(temp_dir, "labeled_patients.csv")
    create_combined_labeled_patients_csv(all_labels, combined_labels_path)
    
    # Paths for batches and representations
    path_to_batches = os.path.join(temp_dir, "batches")
    path_to_representations = os.path.join(temp_dir, "representations.pkl")
    
    # Find clmbr_create_batches command
    clmbr_batches_cmd = shutil.which("clmbr_create_batches")
    if clmbr_batches_cmd is None:
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        if conda_prefix and os.path.exists(f"{conda_prefix}/bin/clmbr_create_batches"):
            clmbr_batches_cmd = f"{conda_prefix}/bin/clmbr_create_batches"
        elif os.path.exists(os.path.expanduser("~/miniconda3/envs/EHRSHOT_ENV/bin/clmbr_create_batches")):
            clmbr_batches_cmd = os.path.expanduser("~/miniconda3/envs/EHRSHOT_ENV/bin/clmbr_create_batches")
        else:
            clmbr_batches_cmd = "clmbr_create_batches"
    
    # Determine batch size: 16384 is femr default, fits A100 80GB; 131072 OOMs (~384GB); 1024 for CPU
    if batch_size is None:
        batch_size = 1024 if use_cpu else 16384
        logger.info(f"Using batch_size={batch_size} ({'CPU' if use_cpu else 'GPU'})")
    
    # Check if batches exist and if they have the right batch size
    batches_need_recreation = False
    if os.path.exists(path_to_batches) and not is_force_refresh:
        # Try to detect batch size from log file
        log_file = os.path.join(path_to_batches, "log")
        logger.info(f"Checking for existing batches at: {path_to_batches}")
        logger.info(f"Looking for log file at: {log_file}")
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    first_line = f.readline()
                    logger.debug(f"First line of log file: {first_line[:200]}...")
                    if 'batch_size=' in first_line:
                        import re
                        match = re.search(r'batch_size=(\d+)', first_line)
                        if match:
                            existing_batch_size = int(match.group(1))
                            logger.info(f"Detected existing batch_size={existing_batch_size} from log file")
                            logger.info(f"Target batch_size={batch_size}")
                            # If existing batch size is too large for CPU and we're using CPU, recreate
                            if existing_batch_size > 2048 and batch_size <= 2048:
                                logger.warning(f"Existing batches have batch_size={existing_batch_size}, but we need {batch_size} for CPU execution")
                                logger.warning("Automatically recreating batches with smaller batch size...")
                                batches_need_recreation = True
                            else:
                                logger.info(f"Existing batch_size={existing_batch_size} is compatible with target batch_size={batch_size}")
                        else:
                            logger.warning(f"Could not find batch_size in log file first line")
                    else:
                        logger.warning(f"Log file doesn't contain 'batch_size=' in first line")
            except Exception as e:
                logger.warning(f"Could not parse batch size from log file: {e}, will check if recreation is needed")
                import traceback
                logger.debug(traceback.format_exc())
        else:
            logger.warning(f"Log file not found at: {log_file}")
        
        if not batches_need_recreation:
            logger.warning(f"Batches already exist at: {path_to_batches}")
            logger.warning("If you encounter out-of-memory errors, recreate batches with smaller batch size:")
            logger.warning("  Use --is_force_refresh --batch_size 1024 (for CPU execution)")
    
    # Generate batches (Phase 1/2)
    if is_force_refresh or not os.path.exists(path_to_batches) or batches_need_recreation:
        logger.info("Phase 1/2: Creating CLMBR batches...")
        if os.path.exists(path_to_batches):
            logger.info(f"Removing existing batches directory (force_refresh={is_force_refresh}, need_recreation={batches_need_recreation})")
            shutil.rmtree(path_to_batches)
        logger.info(f"Creating CLMBR batches at: {path_to_batches}")
        logger.info(f"Using batch size: {batch_size}")
        # Use Python with femr/jax (lrshi EHRSHOT_ENV) to avoid ModuleNotFoundError
        python_exe = _get_clmbr_python()
        logger.info(f"Using Python for CLMBR: {python_exe}")
        cmd_args = [
            python_exe,
            clmbr_batches_cmd,
            path_to_batches,
            "--data_path", path_to_database,
            "--dictionary", path_to_dictionary,
            "--task", "labeled_patients",
            "--batch_size", str(batch_size),
            "--val_start", "70",
            "--labeled_patients_path", combined_labels_path,
        ]
        logger.info(f"Running: {' '.join(cmd_args)}")
        batch_env = os.environ.copy()
        batch_env["PYTHONNOUSERSITE"] = "1"  # Use env's femr/jax, not ~/.local
        result = subprocess.run(cmd_args, env=batch_env, capture_output=False)
        exit_code = result.returncode
        if exit_code != 0:
            raise RuntimeError(f"clmbr_create_batches failed with exit code {exit_code}")
        logger.success(f"Created CLMBR batches")
    else:
        logger.info(f"Batches already exist at: {path_to_batches}, skipping creation")
    
    # Find clmbr_compute_representations command
    clmbr_cmd = shutil.which("clmbr_compute_representations")
    if clmbr_cmd is None:
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        if conda_prefix and os.path.exists(f"{conda_prefix}/bin/clmbr_compute_representations"):
            clmbr_cmd = f"{conda_prefix}/bin/clmbr_compute_representations"
        elif os.path.exists(os.path.expanduser("~/miniconda3/envs/EHRSHOT_ENV/bin/clmbr_compute_representations")):
            clmbr_cmd = os.path.expanduser("~/miniconda3/envs/EHRSHOT_ENV/bin/clmbr_compute_representations")
        else:
            clmbr_cmd = "clmbr_compute_representations"
    
    # Generate representations (Phase 2/2)
    if is_force_refresh or not os.path.exists(path_to_representations):
        logger.info("Phase 2/2: Computing CLMBR representations (this may take a while)...")
        logger.info("  Stages: (1) Load model+batches (2) JAX compile one-time (3) Compute embeddings per batch")
        logger.info(f"  Output: {path_to_representations}")
        # Inline wrapper: patches for EHRSHOT_ENV compatibility
        # - jnp.DeviceArray -> jax.Array
        # - When femr-cuda 0.0.20 lacks get_local_attention_data, patch to use Python fallback on GPU
        _wrapper = r'''
import sys
import jax.numpy as _jnp
import jax as _jax
_jnp.DeviceArray = _jax.Array
sys.argv = ["clmbr_compute_representations"] + sys.argv[1:]
# Patch femr.jax: use Python fallback when get_local_attention_data is missing (femr-cuda 0.0.20)
import femr.extension.jax as _ext_jax
if not hasattr(_ext_jax, "get_local_attention_data"):
    import femr.jax as _femr_jax
    from jax.interpreters import xla
    from jax import xla_computation
    from jax.lib import xla_client
    def _patched_forward(ctx, avals_in, avals_out, queries, keys, values, length, *, attention_width=None, causal=None, aw=None, **kwargs):
        w = attention_width if attention_width is not None else aw
        c = causal
        comp = xla_computation(_femr_jax.local_attention_fallback, static_argnums=(4, 5))(
            *avals_in, w, c
        )
        res = xla_client.ops.Call(ctx.builder, comp, [queries, keys, values, length])
        return [xla_client.ops.GetTupleElement(res, i) for i in range(len(avals_out))]
    def _patched_backward(ctx, avals_in, avals_out, queries, keys, values, length, attention, g, *, attention_width=None, causal=None, aw=None, **kwargs):
        w = attention_width if attention_width is not None else aw
        c = causal
        comp = xla_computation(_femr_jax.local_attention_backward_fallback, static_argnums=(6, 7))(
            *avals_in, w, c
        )
        res = xla_client.ops.Call(ctx.builder, comp, [queries, keys, values, length, attention, g])
        return [xla_client.ops.GetTupleElement(res, i) for i in range(len(avals_out))]
    xla.register_translation(_femr_jax.local_attention_forward_p, _patched_forward)
    xla.register_translation(_femr_jax.local_attention_backward_p, _patched_backward)
# Replace femr's cryptic "Compiling the transformer..." with a clearer one-time message
import femr.models.transformer as _trans
_orig_transformer_call = _trans.EHRTransformer.__call__
_compile_logged = [False]
def _trans_call(self, batch, is_training=False, no_task=False):
    if not _compile_logged[0]:
        _compile_logged[0] = True
        print("[Stage 2/2] JAX compiling CLMBR-T forward pass (one-time, 2-5 min)...", flush=True)
    return _orig_transformer_call(self, batch, is_training, no_task)
_trans.EHRTransformer.__call__ = _trans_call
# Progress bar: store tqdm in module-level dict (BatchLoader is C extension, no dynamic attrs)
import femr.extension.dataloader as _batch_loader
_orig_get_batch = _batch_loader.BatchLoader.get_batch
_clmbr_pbar = {}
def _tqdm_get_batch(self, split, index):
    k = id(self)
    if k not in _clmbr_pbar:
        n = sum(self.get_number_of_batches(s) for s in ("train", "dev", "test"))
        _clmbr_pbar[k] = __import__("tqdm").tqdm(total=n, desc="Computing embeddings", unit="batch")
    out = _orig_get_batch(self, split, index)
    _clmbr_pbar[k].update(1)
    return out
_batch_loader.BatchLoader.get_batch = _tqdm_get_batch
print("[Stage 2/2] Loading model and batches...", flush=True)
from femr.models.scripts import compute_representations
sys.exit(compute_representations())
'''
        python_exe = _get_clmbr_python()
        logger.info(f"Using Python for compute_representations: {python_exe}")
        rep_cmd_args = [
            python_exe,
            "-c",
            _wrapper,
            path_to_representations,
            "--data_path", path_to_database,
            "--batches_path", path_to_batches,
            "--model_dir", path_to_model,
        ]
        logger.info(f"Running: {python_exe} -c '...' {path_to_representations} ...")
        env = os.environ.copy()
        # Use env packages (JAX 0.4.8, femr). femr-cuda 0.0.20 lacks get_local_attention_data;
        # we patch local_attention to use Python fallback when missing.
        env["PYTHONNOUSERSITE"] = "1"
        if use_gpu:
            env.pop("JAX_PLATFORMS", None)
            env.pop("CUDA_VISIBLE_DEVICES", None)
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            # Use CUDA 11 ptxas/nvlink (CUDA 12 produces cubin 129; jaxlib expects 128)
            ptxas_dir = os.path.expanduser("~/.cudnn86/nvidia/cuda_nvcc/bin")
            if os.path.isdir(ptxas_dir):
                env["PATH"] = ptxas_dir + ":" + env.get("PATH", "")
        # Use idle CPUs: set thread counts so Eigen/OpenMP/MKL use all available cores
        if use_cpu:
            n_cpu = os.cpu_count() or 64
            env["OMP_NUM_THREADS"] = str(n_cpu)
            env["MKL_NUM_THREADS"] = str(n_cpu)
            env["OPENBLAS_NUM_THREADS"] = str(n_cpu)
            env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={min(n_cpu, 16)}").strip()
            logger.info(f"CPU parallelism: OMP_NUM_THREADS={n_cpu}, XLA host devices={min(n_cpu, 16)}")
        result = subprocess.run(
            rep_cmd_args,
            env=env,
            capture_output=False,
        )
        
        if result.returncode != 0:
            logger.error(f"clmbr_compute_representations failed with exit code {result.returncode}")
            logger.error("This might be due to:")
            logger.error("  1. CUDA/cuDNN issues - try --use_cpu")
            logger.error("  2. Out of memory - try --batch_size 1024 or --use_cpu")
            logger.error("  Solution: --is_force_refresh --batch_size 1024 --use_cpu")
            raise RuntimeError(f"clmbr_compute_representations failed with exit code {result.returncode}")
        logger.success(f"Computed CLMBR representations")
    else:
        logger.info(f"Representations already exist at: {path_to_representations}, skipping computation")
    
    # Load the representations
    logger.info(f"Loading representations from: {path_to_representations}")
    with open(path_to_representations, 'rb') as f:
        raw_feats = pickle.load(f)
    
    # Handle different pickle formats (from evaluate_multi_task_clmbr.py)
    if isinstance(raw_feats, dict):
        feature_matrix = raw_feats['data_matrix']
        feature_patient_ids = raw_feats['patient_ids']
        feature_times = raw_feats['labeling_time']
    elif len(raw_feats) == 4:
        # Format: (feature_matrix, patient_ids, label_values, times)
        feature_matrix, feature_patient_ids, _, feature_times = raw_feats
    elif len(raw_feats) == 5:
        # Format: (feature_matrix, patient_ids, label_values, times, tasks)
        feature_matrix, feature_patient_ids, _, feature_times, _ = raw_feats
    else:
        raise ValueError(f"Unexpected format in features file. Got {type(raw_feats)} with {len(raw_feats) if hasattr(raw_feats, '__len__') else 'N/A'} elements")
    
    # Convert to numpy arrays if needed
    if not isinstance(feature_matrix, np.ndarray):
        feature_matrix = np.array(feature_matrix)
    if not isinstance(feature_patient_ids, np.ndarray):
        feature_patient_ids = np.array(feature_patient_ids)
    if not isinstance(feature_times, np.ndarray):
        feature_times = np.array(feature_times)
    
    # Ensure feature_times are datetime-like for consistency
    if not np.issubdtype(feature_times.dtype, np.datetime64):
        try:
            feature_times = pd.to_datetime(feature_times).values.astype('datetime64[us]')
        except Exception as e:
            logger.warning(f"Could not convert times to datetime64: {e}, keeping original format")
    
    logger.info(f"Loaded features:")
    logger.info(f"  Feature matrix shape: {feature_matrix.shape}")
    logger.info(f"  Patient IDs shape: {feature_patient_ids.shape}")
    logger.info(f"  Times shape: {feature_times.shape}")
    
    return {
        'data_matrix': feature_matrix,
        'patient_ids': feature_patient_ids,
        'labeling_time': feature_times
    }


def save_features(features: Dict[str, np.ndarray], output_path: str):
    """Save features to pickle file."""
    logger.info(f"Saving features to: {output_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(features, f)
    
    logger.success(f"Features saved to: {output_path}")
    logger.info(f"  Feature matrix shape: {features['data_matrix'].shape}")
    logger.info(f"  Patient IDs shape: {features['patient_ids'].shape}")
    logger.info(f"  Times shape: {features['labeling_time'].shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CLMBR-T features for all tasks"
    )
    parser.add_argument(
        "--path_to_database",
        required=True,
        type=str,
        help="Path to FEMR patient database (directory containing data/*.parquet files)"
    )
    parser.add_argument(
        "--path_to_labels_dir",
        type=str,
        default=None,
        help="Path to directory containing labeled_patients.csv files (legacy; used when --path_to_data_dir not set)"
    )
    parser.add_argument(
        "--path_to_data_dir",
        type=str,
        default=None,
        help="Path to ehrshot-v2 data root. When set, loads from {sft_subdir}/{train,val,test}/{task}.json"
    )
    parser.add_argument(
        "--sft_subdir",
        type=str,
        default="data/sft/naivetext",
        help="Subdir under path_to_data_dir for SFT JSONs (default: data/sft/naivetext). Same as generate_embeddings --sft_dir relative to project root"
    )
    parser.add_argument(
        "--path_to_features_dir",
        required=True,
        type=str,
        help="Path to directory where features will be saved"
    )
    parser.add_argument(
        "--path_to_models_dir",
        required=True,
        type=str,
        help="Path to directory where CLMBR-T model is saved"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="clmbr_t_features.pkl",
        help="Output filename for features (default: clmbr_t_features.pkl)"
    )
    parser.add_argument(
        "--is_force_refresh",
        action='store_true',
        default=False,
        help="If set, overwrite existing output file"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for CLMBR batch creation (default: 16384 for GPU, 1024 for CPU)"
    )
    parser.add_argument(
        "--use_cpu",
        action='store_true',
        help="Force CPU execution (smaller batch size, slower)"
    )
    parser.add_argument(
        "--use_gpu",
        action='store_true',
        help="Request GPU/CUDA (faster, batch 16384). Use when GPU is available."
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Path to log file. If set, duplicate all log output to this file"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.log_file:
        log_path = args.log_file
        if not os.path.isabs(log_path):
            proot = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            log_path = os.path.join(proot, log_path)
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        logger.add(log_path, level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}")
        logger.info(f"Logging to {log_path}")

    # Validate: need either path_to_data_dir or path_to_labels_dir
    if not args.path_to_data_dir and not args.path_to_labels_dir:
        raise ValueError("Provide --path_to_data_dir (naivetext splits) or --path_to_labels_dir (legacy label dirs)")
    if args.path_to_data_dir and args.path_to_labels_dir:
        logger.warning("Both --path_to_data_dir and --path_to_labels_dir set; using --path_to_data_dir")
        args.path_to_labels_dir = None

    # Convert relative paths to absolute
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if not os.path.isabs(args.path_to_database):
        args.path_to_database = os.path.join(project_root, args.path_to_database)

    if args.path_to_labels_dir and not os.path.isabs(args.path_to_labels_dir):
        args.path_to_labels_dir = os.path.join(project_root, args.path_to_labels_dir)

    if args.path_to_data_dir and not os.path.isabs(args.path_to_data_dir):
        args.path_to_data_dir = os.path.join(project_root, args.path_to_data_dir)

    if not os.path.isabs(args.path_to_features_dir):
        args.path_to_features_dir = os.path.join(project_root, args.path_to_features_dir)

    if not os.path.isabs(args.path_to_models_dir):
        args.path_to_models_dir = os.path.join(project_root, args.path_to_models_dir)

    output_path = os.path.join(args.path_to_features_dir, args.output_filename)

    logger.info("=" * 60)
    logger.info("CLMBR-T Feature Generation")
    logger.info("=" * 60)
    logger.info(f"Database path: {args.path_to_database}")
    if args.path_to_data_dir and not args.path_to_labels_dir:
        logger.info(f"Data dir (naivetext splits): {args.path_to_data_dir}")
    else:
        logger.info(f"Labels directory: {args.path_to_labels_dir}")
    logger.info(f"Features directory: {args.path_to_features_dir}")
    logger.info(f"Model directory: {args.path_to_models_dir}")
    logger.info(f"Output file: {output_path}")
    logger.info("=" * 60)

    # Check if output already exists
    if os.path.exists(output_path) and not args.is_force_refresh:
        logger.warning(f"Output file already exists: {output_path}")
        logger.warning("Use --is_force_refresh to overwrite")
        return

    # Load labels for all tasks
    logger.info("Loading labels...")
    if args.path_to_data_dir and not args.path_to_labels_dir:
        all_labels = load_labels_from_naivetext_splits(
            args.path_to_data_dir, sft_subdir=args.sft_subdir
        )
        logger.info(f"Loaded {sum(len(v[0]) for v in all_labels.values())} patient-time pairs from naivetext splits")
    else:
        all_labels = load_all_task_labels(args.path_to_labels_dir)
    
    if len(all_labels) == 0:
        raise ValueError("No labels loaded! Check that label files exist in the labels directory.")
    
    # Compute CLMBR-T features
    logger.info("Computing CLMBR-T features...")
    features = compute_clmbr_t_features(
        args.path_to_database,
        args.path_to_models_dir,
        all_labels,
        args.path_to_features_dir,
        is_force_refresh=args.is_force_refresh,
        batch_size=args.batch_size,
        use_cpu=args.use_cpu,
        use_gpu=args.use_gpu,
    )
    
    # Save features
    save_features(features, output_path)
    
    logger.success("Done!")


if __name__ == "__main__":
    main()
