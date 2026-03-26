#!/usr/bin/env python3
"""
Compute AUROC and AUPRC with bootstrap 95% confidence intervals.

Reads predictions.csv files produced by eval_direct.py, eval_reasoning.py,
or eval_embeddings.py, and computes per-task and aggregate metrics.

Inputs:
  --predictions : One or more paths to predictions.csv files.
  --output_dir  : Where to write metrics.

Outputs:
  {output_dir}/per_task_metrics.json   -- per-task AUROC / AUPRC with CIs
  {output_dir}/summary.json            -- mean across tasks

predictions.csv format:
  patient_id, label_time, ground_truth, probability_score, target_task

Connects to:
  - Upstream: eval_direct.py, eval_reasoning.py, eval_embeddings.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.tasks import SEED

BOOTSTRAP_N = 1000


def _bootstrap_samples(y_true, y_score, fn, n=BOOTSTRAP_N, seed=SEED):
    """
    Return the full vector of bootstrap metric values (length n) with NaNs for
    iterations where the metric is undefined (e.g., only one class present).
    """
    rng = np.random.RandomState(seed)
    vals = np.full(n, np.nan, dtype=float)
    for i in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        vals[i] = fn(yt, ys)
    return vals


def _summary_from_samples(samples: np.ndarray):
    """Compute mean and 95% CI from a vector of bootstrap samples (with NaNs)."""
    if samples.size == 0 or np.all(np.isnan(samples)):
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
    arr = samples[~np.isnan(samples)]
    if arr.size == 0:
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "ci_lo": float(np.percentile(arr, 2.5)),
        "ci_hi": float(np.percentile(arr, 97.5)),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", nargs="+", required=True,
                   help="Path(s) to predictions.csv file(s)")
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()

    # Load and concatenate all prediction files
    dfs = []
    for path in args.predictions:
        dfs.append(pd.read_csv(path))
    df = pd.concat(dfs, ignore_index=True)

    tasks = sorted(df["target_task"].unique())
    per_task = {}
    # Store full bootstrap samples per task so we can compute group-level CIs
    task_samples = {}  # task -> {"auroc": np.ndarray, "auprc": np.ndarray}

    for task in tasks:
        sub = df[df["target_task"] == task]
        y_true = sub["ground_truth"].values
        y_score = sub["probability_score"].values

        if len(np.unique(y_true)) < 2:
            logger.warning(f"  {task}: only one class, skipping")
            continue

        auroc_samples = _bootstrap_samples(y_true, y_score, roc_auc_score)
        auprc_samples = _bootstrap_samples(y_true, y_score, average_precision_score)

        auroc = _summary_from_samples(auroc_samples)
        auprc = _summary_from_samples(auprc_samples)

        task_samples[task] = {
            "auroc": auroc_samples,
            "auprc": auprc_samples,
        }

        per_task[task] = {
            "n": len(sub),
            "auroc": auroc,
            "auprc": auprc,
        }
        logger.info(f"  {task}: AUROC={auroc['mean']:.4f} "
                    f"[{auroc['ci_lo']:.4f}, {auroc['ci_hi']:.4f}]  "
                    f"AUPRC={auprc['mean']:.4f} "
                    f"[{auprc['ci_lo']:.4f}, {auprc['ci_hi']:.4f}]")

    # ------------------------------------------------------------------
    # Aggregate metrics: task groups and overall (per-iteration averaging)
    # ------------------------------------------------------------------
    # Define groups by task name prefixes
    GROUP_DEFS = {
        "guo": ["guo_icu", "guo_los", "guo_readmission"],
        "new": [
            "new_hypertension",
            "new_hyperlipidemia",
            "new_pancan",
            "new_celiac",
            "new_lupus",
            "new_acutemi",
        ],
        "lab": [
            "lab_thrombocytopenia",
            "lab_hyperkalemia",
            "lab_hypoglycemia",
            "lab_hyponatremia",
            "lab_anemia",
        ],
        "chexpert": ["chexpert"],
    }

    def _aggregate_group(samples_dict, members):
        """Compute group-level AUROC/AUPRC summaries from per-task samples."""
        present = [m for m in members if m in samples_dict]
        if not present:
            return None
        out = {}
        for metric in ("auroc", "auprc"):
            mats = []
            for t in present:
                arr = samples_dict[t][metric]
                if arr.size == 0:
                    continue
                mats.append(arr)
            if not mats:
                out[metric] = {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
                continue
            stacked = np.vstack(mats)  # (n_tasks_in_group, n_boot)
            # Average across tasks for each bootstrap iteration
            group_samples = np.nanmean(stacked, axis=0)
            out[metric] = _summary_from_samples(group_samples)
        return out

    groups = {}
    for name, members in GROUP_DEFS.items():
        agg = _aggregate_group(task_samples, members)
        if agg is not None:
            groups[name] = {
                "tasks": [m for m in members if m in task_samples],
                "auroc": agg["auroc"],
                "auprc": agg["auprc"],
            }

    # Overall aggregate: average across all tasks per bootstrap iteration
    overall = {"auroc": {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0},
               "auprc": {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}}
    if task_samples:
        for metric in ("auroc", "auprc"):
            mats = [v[metric] for v in task_samples.values() if v[metric].size > 0]
            if mats:
                stacked = np.vstack(mats)
                overall_samples = np.nanmean(stacked, axis=0)
                overall[metric] = _summary_from_samples(overall_samples)

    # Backwards-compatible top-level means (no CIs); keep for existing consumers
    summary = {
        "n_tasks": len(per_task),
        "mean_auroc": float(overall["auroc"]["mean"]),
        "mean_auprc": float(overall["auprc"]["mean"]),
        "overall": overall,
        "groups": groups,
    }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_task_metrics.json", "w") as f:
        json.dump(per_task, f, indent=2)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nSummary: mean AUROC={summary['mean_auroc']:.4f}  "
                f"mean AUPRC={summary['mean_auprc']:.4f}")
    logger.success(f"Metrics saved to {out}")


if __name__ == "__main__":
    main()
