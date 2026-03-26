Adaptad from https://github.com/stefanhgm/ehrshot-benchmark

# CLMBR-T Multi-Task Evaluation Pipeline

This pipeline evaluates CLMBR-T features on all 15 tasks using logistic regression, with the same output format as `05_eval/eval_embeddings.py`.

## Overview

1. **Generate CLMBR features** using `generate_clmbr_t_features.py` (patients in splits only)
2. **Evaluate** all 15 tasks with `evaluate_clmbr_all_tasks.py` (n=40 cohort or full train)
3. **Compute summary metrics** (per_task_metrics.json, summary.json) via `compute_metrics.py`

## Step 1: Generate CLMBR Features

Uses `generate_clmbr_t_features.py` (in this dir) with `--path_to_data_dir`. Only generates features for patient-time pairs in our splits (`data/sft/naivetext/{train,val,test}/{task}.json` for all 15 tasks).

**Default paths** (from `EHRSHOT_ASSETS` under benchmark repo root, same as `ehrshot/bash_scripts/5_generate_clmbr_features.sh`):
- Database: `EHRSHOT_ASSETS/femr/extract`
- Models: `EHRSHOT_ASSETS/models/clmbr` (must contain `clmbr_model`, `dictionary` subdirs)
- Features output: `EHRSHOT_ASSETS/features/clmbr_t_features.pkl`

```bash
cd ehrshot-v2

# With EHRSHOT_ASSETS present at ../EHRSHOT_ASSETS:
bash 06_baselines/clmbrt/run_generate_clmbr_features.sh
```

Override with env vars: `PATH_TO_DATABASE`, `PATH_TO_MODELS_DIR`, `PATH_TO_DATA_DIR`, `PATH_TO_FEATURES_DIR`, `BATCH_SIZE`, `OUTPUT_FILENAME`, `IS_FORCE_REFRESH=1`

Output: `EHRSHOT_ASSETS/features/clmbr_t_features.pkl` (default), plus `clmbr_t_temp/` in the features dir.

## Step 2: Run Evaluation (all 15 tasks)

Uses `data/sft/naivetext/{split}/{task}.json` and (optionally) `data/rubric/{task}/cohort.json` for n=40.

```bash
cd ehrshot-v2
bash 06_baselines/clmbrt/run_evaluate_all_tasks.sh
```

**Defaults:**
- Features: `EHRSHOT_ASSETS/features/clmbr_t_features.pkl` (from Step 1)
- n=40 cohort from `data/rubric/{task}/cohort.json`
- Output: `data/results/clmbrt_n40/{task}/metrics.json`, `predictions.csv`
- Summary: `data/results/metrics/clmbrt_n40/summary.json`, `per_task_metrics.json`

**Full train (no n=40):**
```bash
N_TRAIN= PATH_TO_OUTPUT_DIR=data/results/clmbrt_full bash 06_baselines/clmbrt/run_evaluate_all_tasks.sh
```

**Single task:**
```bash
python 06_baselines/clmbrt/evaluate_clmbr_all_tasks.py \
    --task_name chexpert \
    --path_to_clmbr_features ../EHRSHOT_ASSETS/features/clmbr_t_features.pkl \
    --path_to_data_dir . \
    --path_to_output_dir data/results/clmbrt_n40 \
    --n_train 40 \
    --cohort_dir data/rubric
```

## Arguments (evaluate_clmbr_all_tasks.py)

- `--task_name`: One of the 15 tasks
- `--path_to_clmbr_features`: Path to CLMBR features pickle
- `--path_to_data_dir`: Root containing `data/sft/naivetext/{split}/{task}.json`
- `--path_to_output_dir`: Output directory (writes `{task}/metrics.json`, `predictions.csv`)
- `--n_train`: If set (e.g. 40), filter train to cohort. Requires `--cohort_dir`
- `--cohort_dir`: Dir with `{task}/cohort.json` (e.g. `data/rubric`)

## Output (matches 05_eval/eval_embeddings.py)

### Per-task
`data/results/clmbrt_n40/{task}/`
- `metrics.json` — best_c, val_auroc, test_auroc/auprc with CIs
- `predictions.csv` — patient_id, label_time, ground_truth, probability_score, target_task

### Summary metrics (data/results/metrics/clmbrt_n40/)
- `per_task_metrics.json` — AUROC/AUPRC with bootstrap CIs per task
- `summary.json` — overall and group-level (guo, lab, new, chexpert) metrics

Same layout as `data/results/metrics/embedding_global-rubric_full/`.

## Notes

- Best C is selected on the validation set; the model is refit on train and evaluated on test only
- ehrshot imports: `evaluate_clmbr_all_tasks.py` adds the parent dir (ehrshot-benchmark) to `sys.path` so `from ehrshot.utils import compute_feature_label_alignment` resolves

## CLMBR Feature Generation: Environment Requirements

Feature generation (`run_generate_clmbr_features.sh`) requires femr, JAX, optax, dm-haiku. Known working combo from this project:

```
jax==0.4.33  jaxlib==0.4.33  optax==0.1.4  femr-cuda==0.0.20  dm-haiku
```

**Common issues:**
- `ModuleNotFoundError: jaxlib`: Ensure `conda activate EHRSHOT_ENV` and install jax+jaxlib in that env
- `~/.local` vs conda: `clmbr_create_batches` / `clmbr_compute_representations` in `~/.local/bin` use shebang `#!/usr/bin/python3`; the script invokes them via `sys.executable` (your conda Python) so the correct env is used
- `jnp.DeviceArray` / `xla.register_translation` errors: femr 0.0.20 is incompatible with JAX 0.4.33 (deprecated APIs removed). Use a clean env with only the packages above, or run the original `ehrshot/bash_scripts/5_generate_clmbr_features.sh` (SLURM) which may have a working environment
