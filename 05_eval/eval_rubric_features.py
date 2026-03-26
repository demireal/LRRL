#!/usr/bin/env python3
"""
Evaluate rubric feature vectors (produced by 03_globalrubric/feature_extractors/)
using Lasso (L1-regularized logistic regression) and XGBoost classifiers.

For each task:
  - Loads {features_dir}/{task}/{split}.npz (train / val / test)
  - Imputes NaN with column means from train
  - Lasso sweep: C_VALUES = np.logspace(-7, -1, 15), L1, pick best C by val AUROC
  - XGBoost grid search: n_estimators, max_depth, learning_rate, subsample
  - Reports test AUROC for both; saves {output_dir}/{task}/metrics.json

CLI:
  python 05_eval/eval_rubric_features.py \\
    --features_dir data/rubric_features_auto \\
    --output_dir   data/results/rubric_features_auto_full \\
    --tasks guo_icu
"""

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("xgboost not installed — XGBoost evaluation will be skipped")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.tasks import ALL_TASK_NAMES

C_VALUES = np.logspace(-7, -1, 15).tolist()

XGB_GRID = {
    "n_estimators":   [50, 100, 300],
    "max_depth":      [2, 3, 5, 7],
    "learning_rate":  [0.01, 0.05],
    "subsample":      [0.8],
    "min_child_weight": [1, 5],
}


def subsample_balanced(X: np.ndarray, y: np.ndarray,
                        n_pos: int = 20, n_neg: int = 20,
                        seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Subsample to exactly n_pos positives and n_neg negatives."""
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    sel_pos = rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False)
    sel_neg = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
    sel = np.concatenate([sel_pos, sel_neg])
    rng.shuffle(sel)
    return X[sel], y[sel]


def load_cohort_mask(cohort_dir: Path, task: str,
                     patient_ids: np.ndarray, prediction_times: np.ndarray) -> np.ndarray:
    """Return boolean mask selecting the 40 cohort patients from a feature array."""
    cohort_path = cohort_dir / task / "cohort.json"
    with open(cohort_path) as f:
        cohort = json.load(f)
    cohort_keys = {(int(c["patient_id"]), str(c["prediction_time"])) for c in cohort}
    mask = np.array([
        (int(pid), str(pt)) in cohort_keys
        for pid, pt in zip(patient_ids, prediction_times)
    ])
    return mask


def load_split(features_dir: Path, task: str, split: str):
    path = features_dir / task / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    d = np.load(path, allow_pickle=True)
    X = d["embeddings"].astype(np.float32)
    y = d["labels"].astype(int)
    patient_ids = d["patient_ids"] if "patient_ids" in d else np.arange(len(y))
    prediction_times = d["prediction_times"] if "prediction_times" in d else np.array([""] * len(y))
    return X, y, patient_ids, prediction_times


def impute(X_train, X_val, X_test):
    col_means = np.nanmean(X_train, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)

    def _fill(X):
        X = X.copy().astype(np.float64)
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        return X

    return _fill(X_train), _fill(X_val), _fill(X_test), col_means


def eval_lasso(X_tr, y_tr, X_val, y_val, X_te, y_te):
    scaler = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s  = scaler.transform(X_te)

    best_c, best_val_auc = None, -1.0
    sweep = []
    for c in C_VALUES:
        clf = LogisticRegression(C=c, penalty="l1", solver="saga",
                                 max_iter=10000, random_state=42)
        clf.fit(X_tr_s, y_tr)
        val_auc = roc_auc_score(y_val, clf.predict_proba(X_val_s)[:, 1])
        sweep.append({"C": c, "val_auroc": val_auc})
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_c = c

    clf = LogisticRegression(C=best_c, penalty="l1", solver="saga",
                             max_iter=10000, random_state=42)
    clf.fit(X_tr_s, y_tr)
    test_proba = clf.predict_proba(X_te_s)[:, 1]
    test_auc = roc_auc_score(y_te, test_proba)
    return {
        "best_C": best_c,
        "val_auroc": best_val_auc,
        "test_auroc": test_auc,
        "test_proba": test_proba,
        "sweep": sweep,
    }


def eval_xgb(X_tr, y_tr, X_val, y_val, X_te, y_te):
    if not HAS_XGB:
        return None

    keys = list(XGB_GRID.keys())
    values = list(XGB_GRID.values())

    best_params, best_val_auc = None, -1.0
    for combo in product(*values):
        params = dict(zip(keys, combo))
        clf = XGBClassifier(
            **params,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=4,
            verbosity=0,
        )
        clf.fit(X_tr, y_tr)
        val_auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_params = params

    clf = XGBClassifier(
        **best_params,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=4,
        verbosity=0,
    )
    clf.fit(X_tr, y_tr)
    test_proba = clf.predict_proba(X_te)[:, 1]
    test_auc = roc_auc_score(y_te, test_proba)
    return {
        "best_params": best_params,
        "val_auroc": best_val_auc,
        "test_proba": test_proba,
        "test_auroc": test_auc,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features_dir", required=True,
                   help="Root features directory (data/rubric_features_auto)")
    p.add_argument("--output_dir", required=True,
                   help="Where to write per-task metrics.json and predictions.csv")
    p.add_argument("--metrics_dir", default=None,
                   help="If set, write Lasso per_task_metrics.json here. "
                        "XGB goes to same path + '_xgb' unless --xgb_metrics_dir is set.")
    p.add_argument("--xgb_metrics_dir", default=None,
                   help="Override directory for XGB full metrics (default: metrics_dir + '_xgb').")
    p.add_argument("--n40_metrics_dir", default=None,
                   help="Override directory for Lasso n=40 metrics (default: metrics_dir + '_n40').")
    p.add_argument("--n40_xgb_metrics_dir", default=None,
                   help="Override directory for XGB n=40 metrics (default: metrics_dir + '_xgb_n40').")
    p.add_argument("--cohort_dir", default=None,
                   help="If set, also run n=40 eval using cohort patients from "
                        "{cohort_dir}/{task}/cohort.json as the training set.")
    p.add_argument("--n40_only", action="store_true",
                   help="If set, skip full-training eval and only run n=40.")
    p.add_argument("--xgb_only", action="store_true",
                   help="If set, skip Lasso and only run XGBoost.")
    p.add_argument("--fixed_params_full", default=None,
                   help="Path to per_task_metrics.json with saved XGB best_params for full training.")
    p.add_argument("--fixed_params_n40", default=None,
                   help="Path to per_task_metrics.json with saved XGB best_params for n=40 training.")
    p.add_argument("--tasks", nargs="+", default=ALL_TASK_NAMES)
    return p.parse_args()


def _bootstrap_ci(y_true, y_score, fn, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        vals.append(fn(yt, ys))
    if not vals:
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
    arr = np.array(vals)
    return {
        "mean": float(np.mean(arr)),
        "ci_lo": float(np.percentile(arr, 2.5)),
        "ci_hi": float(np.percentile(arr, 97.5)),
    }


def main():
    args = parse_args()
    features_dir = Path(args.features_dir)
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort_dir   = Path(args.cohort_dir) if args.cohort_dir else None

    fixed_params_full = {}
    if args.fixed_params_full:
        with open(args.fixed_params_full) as f:
            _fp = json.load(f)
        fixed_params_full = {t: v["best_params"] for t, v in _fp.items() if "best_params" in v}
        logger.info(f"Loaded fixed XGB params for {len(fixed_params_full)} tasks (full)")

    fixed_params_n40 = {}
    if args.fixed_params_n40:
        with open(args.fixed_params_n40) as f:
            _fp = json.load(f)
        fixed_params_n40 = {t: v["best_params"] for t, v in _fp.items() if "best_params" in v}
        logger.info(f"Loaded fixed XGB params for {len(fixed_params_n40)} tasks (n=40)")

    metrics_dir = Path(args.metrics_dir) if args.metrics_dir else None
    if metrics_dir:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        xgb_metrics_dir = (Path(args.xgb_metrics_dir) if args.xgb_metrics_dir
                           else metrics_dir.parent / (metrics_dir.name + "_xgb"))
        n40_metrics_dir = (Path(args.n40_metrics_dir) if args.n40_metrics_dir
                           else metrics_dir.parent / (metrics_dir.name + "_n40"))
        n40_xgb_metrics_dir = (Path(args.n40_xgb_metrics_dir) if args.n40_xgb_metrics_dir
                                else metrics_dir.parent / (metrics_dir.name + "_xgb_n40"))
        xgb_metrics_dir.mkdir(parents=True, exist_ok=True)
        if cohort_dir:
            n40_metrics_dir.mkdir(parents=True, exist_ok=True)
            n40_xgb_metrics_dir.mkdir(parents=True, exist_ok=True)
    else:
        xgb_metrics_dir = n40_metrics_dir = n40_xgb_metrics_dir = None

    summary_rows = []
    per_task_lasso    = {}
    per_task_xgb      = {}
    per_task_lasso_n40  = {}
    per_task_xgb_n40    = {}
    raw_lasso    = {}
    raw_xgb      = {}
    raw_lasso_n40  = {}
    raw_xgb_n40    = {}

    for task in args.tasks:
        logger.info(f"\n{'='*60}\nEvaluating: {task}\n{'='*60}")

        try:
            X_tr, y_tr, tr_pids, tr_ptimes = load_split(features_dir, task, "train")
            X_val, y_val, _, _             = load_split(features_dir, task, "val")
            X_te, y_te, pids, ptimes       = load_split(features_dir, task, "test")
        except FileNotFoundError as e:
            logger.warning(f"  Skipping {task}: {e}")
            continue

        logger.info(f"  shapes — train:{X_tr.shape}, val:{X_val.shape}, test:{X_te.shape}")
        logger.info(f"  labels — train pos:{y_tr.sum()}, val pos:{y_val.sum()}, test pos:{y_te.sum()}")

        X_tr, X_val, X_te, _ = impute(X_tr, X_val, X_te)

        y_te_arr = np.array(y_te)

        task_out = output_dir / task
        task_out.mkdir(parents=True, exist_ok=True)

        lasso_res = xgb_res = None
        if not args.n40_only:
            if not args.xgb_only:
                logger.info("  Running Lasso sweep ...")
                lasso_res = eval_lasso(X_tr, y_tr, X_val, y_val, X_te, y_te)
                logger.info(f"  Lasso  best_C={lasso_res['best_C']:.2e}  "
                            f"val_auroc={lasso_res['val_auroc']:.4f}  "
                            f"test_auroc={lasso_res['test_auroc']:.4f}")

            if HAS_XGB:
                if task in fixed_params_full:
                    fp = fixed_params_full[task]
                    logger.info(f"  Using fixed XGB params: {fp}")
                    clf = XGBClassifier(**fp, use_label_encoder=False,
                                        eval_metric="logloss", verbosity=0, random_state=42)
                    clf.fit(X_tr, y_tr)
                    xgb_proba_fixed = clf.predict_proba(X_te)[:, 1]
                    test_auroc = float(roc_auc_score(y_te, xgb_proba_fixed))
                    logger.info(f"  XGB (fixed) test_auroc={test_auroc:.4f}")
                    xgb_res = {"best_params": fp, "val_auroc": float("nan"),
                               "test_auroc": test_auroc, "test_proba": xgb_proba_fixed.tolist()}
                else:
                    logger.info("  Running XGBoost grid search ...")
                    xgb_res = eval_xgb(X_tr, y_tr, X_val, y_val, X_te, y_te)
                    logger.info(f"  XGB best_params={xgb_res['best_params']}  "
                                f"val_auroc={xgb_res['val_auroc']:.4f}  "
                                f"test_auroc={xgb_res['test_auroc']:.4f}")

            if lasso_res:
                lasso_proba = np.array(lasso_res["test_proba"])
                pd.DataFrame({
                    "patient_id": pids, "label_time": ptimes,
                    "ground_truth": y_te, "probability_score": lasso_proba,
                    "target_task": task,
                }).to_csv(task_out / "predictions_lasso.csv", index=False)

            if xgb_res:
                xgb_proba = np.array(xgb_res["test_proba"])
                pd.DataFrame({
                    "patient_id": pids, "label_time": ptimes,
                    "ground_truth": y_te, "probability_score": xgb_proba,
                    "target_task": task,
                }).to_csv(task_out / "predictions_xgb.csv", index=False)

            save_lasso = ({k: v for k, v in lasso_res.items() if k != "test_proba"}
                          if lasso_res else None)
            save_xgb   = ({k: v for k, v in xgb_res.items()  if k != "test_proba"}
                          if xgb_res else None)
            metrics = {
                "task": task,
                "n_features": X_tr.shape[1],
                "n_train": int(X_tr.shape[0]),
                "n_val":   int(X_val.shape[0]),
                "n_test":  int(X_te.shape[0]),
                "lasso": save_lasso,
                "xgboost": save_xgb,
            }
            with open(task_out / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            logger.success(f"  saved -> {task_out}/metrics.json"
                           + ("  predictions_lasso.csv" if lasso_res else "")
                           + ("  predictions_xgb.csv" if xgb_res else ""))

            if lasso_res:
                lasso_proba = np.array(lasso_res["test_proba"])
                lasso_auroc_ci = _bootstrap_ci(y_te_arr, lasso_proba, roc_auc_score)
                lasso_auprc_ci = _bootstrap_ci(y_te_arr, lasso_proba, average_precision_score)
                per_task_lasso[task] = {"n": int(len(y_te)), "auroc": lasso_auroc_ci, "auprc": lasso_auprc_ci}
                raw_lasso[task] = (y_te_arr, lasso_proba)

            if xgb_res:
                xgb_auroc_ci = _bootstrap_ci(y_te_arr, xgb_proba, roc_auc_score)
                xgb_auprc_ci = _bootstrap_ci(y_te_arr, xgb_proba, average_precision_score)
                per_task_xgb[task] = {
                    "n": int(len(y_te)),
                    "auroc": xgb_auroc_ci,
                    "auprc": xgb_auprc_ci,
                    "best_params": xgb_res["best_params"],
                }
                raw_xgb[task] = (y_te_arr, xgb_proba)

            row = {"task": task}
            if lasso_res:
                row["lasso_test_auroc"] = lasso_res["test_auroc"]
                row["lasso_best_C"] = lasso_res["best_C"]
            if xgb_res:
                row["xgb_test_auroc"] = xgb_res["test_auroc"]
            summary_rows.append(row)

        # n=40 eval using cohort patients as training set
        if cohort_dir:
            cohort_path = cohort_dir / task / "cohort.json"
            if not cohort_path.exists():
                logger.warning(f"  n=40: cohort not found at {cohort_path}, skipping")
            else:
                mask = load_cohort_mask(cohort_dir, task, tr_pids, tr_ptimes)
                n_matched = mask.sum()
                if n_matched < 2:
                    logger.warning(f"  n=40: only {n_matched} cohort patients matched in train, skipping")
                else:
                    X_tr40, y_tr40 = subsample_balanced(X_tr[mask], y_tr[mask])
                    logger.info(f"  n=40 train: {len(y_tr40)} patients "
                                f"({y_tr40.sum()} pos, {(~y_tr40.astype(bool)).sum()} neg) "
                                f"[subsampled from {n_matched} cohort matches]")

                    lasso40 = eval_lasso(X_tr40, y_tr40, X_val, y_val, X_te, y_te)
                    logger.info(f"  n=40 Lasso  best_C={lasso40['best_C']:.2e}  "
                                f"val_auroc={lasso40['val_auroc']:.4f}  "
                                f"test_auroc={lasso40['test_auroc']:.4f}")

                    xgb40 = None
                    if HAS_XGB:
                        if task in fixed_params_n40:
                            fp40 = fixed_params_n40[task]
                            logger.info(f"  Using fixed n=40 XGB params: {fp40}")
                            clf40 = XGBClassifier(**fp40, use_label_encoder=False,
                                                  eval_metric="logloss", verbosity=0, random_state=42)
                            clf40.fit(X_tr40, y_tr40)
                            xgb40_proba_fixed = clf40.predict_proba(X_te)[:, 1]
                            test_auroc40 = float(roc_auc_score(y_te, xgb40_proba_fixed))
                            logger.info(f"  n=40 XGB (fixed) test_auroc={test_auroc40:.4f}")
                            xgb40 = {"best_params": fp40, "val_auroc": float("nan"),
                                     "test_auroc": test_auroc40, "test_proba": xgb40_proba_fixed.tolist()}
                        else:
                            xgb40 = eval_xgb(X_tr40, y_tr40, X_val, y_val, X_te, y_te)
                            logger.info(f"  n=40 XGB best_params={xgb40['best_params']}  "
                                        f"val_auroc={xgb40['val_auroc']:.4f}  "
                                        f"test_auroc={xgb40['test_auroc']:.4f}")

                    lasso40_proba = np.array(lasso40["test_proba"])
                    lasso40_auroc_ci = _bootstrap_ci(y_te_arr, lasso40_proba, roc_auc_score)
                    lasso40_auprc_ci = _bootstrap_ci(y_te_arr, lasso40_proba, average_precision_score)
                    per_task_lasso_n40[task] = {"n": int(len(y_te)),
                                                "auroc": lasso40_auroc_ci, "auprc": lasso40_auprc_ci}
                    raw_lasso_n40[task] = (y_te_arr, lasso40_proba)

                    if xgb40:
                        xgb40_proba = np.array(xgb40["test_proba"])
                        per_task_xgb_n40[task] = {
                            "n": int(len(y_te)),
                            "auroc": _bootstrap_ci(y_te_arr, xgb40_proba, roc_auc_score),
                            "auprc": _bootstrap_ci(y_te_arr, xgb40_proba, average_precision_score),
                            "best_params": xgb40["best_params"],
                        }
                        raw_xgb_n40[task] = (y_te_arr, xgb40_proba)

    # Save standard per_task_metrics.json + summary.json
    def _save_metrics_dir(mdir: Path, per_task: dict, label: str,
                          raw_preds: dict | None = None, n_boot: int = 1000, seed: int = 42):
        mdir.mkdir(parents=True, exist_ok=True)
        out_path = mdir / "per_task_metrics.json"
        with open(out_path, "w") as f:
            json.dump(per_task, f, indent=2)
        logger.success(f"Saved {label} per_task_metrics -> {out_path}")

        GROUPS = {
            "guo":      ["guo_icu", "guo_los", "guo_readmission"],
            "lab":      ["lab_thrombocytopenia", "lab_hyperkalemia",
                         "lab_hypoglycemia", "lab_hyponatremia", "lab_anemia"],
            "new":      ["new_hypertension", "new_hyperlipidemia", "new_pancan",
                         "new_celiac", "new_lupus", "new_acutemi"],
            "chexpert": ["chexpert"],
        }

        def _joint_bootstrap_mean(task_list, metric_fn, rng):
            tasks_with_data = [t for t in task_list if t in (raw_preds or {}) and t in per_task]
            if not tasks_with_data:
                vals = [per_task[t]["auroc"]["mean"] if metric_fn is roc_auc_score
                        else per_task[t]["auprc"]["mean"]
                        for t in task_list if t in per_task]
                m = float(np.mean(vals)) if vals else 0.0
                return {"mean": m, "ci_lo": m, "ci_hi": m}

            round_means = []
            for _ in range(n_boot):
                task_scores = []
                for t in tasks_with_data:
                    yt, ys = raw_preds[t]
                    idx = rng.choice(len(yt), len(yt), replace=True)
                    yt_b, ys_b = yt[idx], ys[idx]
                    if len(np.unique(yt_b)) < 2:
                        continue
                    task_scores.append(metric_fn(yt_b, ys_b))
                if task_scores:
                    round_means.append(float(np.mean(task_scores)))

            if not round_means:
                vals = [per_task[t]["auroc"]["mean"] if metric_fn is roc_auc_score
                        else per_task[t]["auprc"]["mean"]
                        for t in tasks_with_data]
                m = float(np.mean(vals))
                return {"mean": m, "ci_lo": m, "ci_hi": m}

            arr = np.array(round_means)
            return {
                "mean":  float(np.mean(arr)),
                "ci_lo": float(np.percentile(arr, 2.5)),
                "ci_hi": float(np.percentile(arr, 97.5)),
            }

        rng = np.random.RandomState(seed)
        all_tasks = list(per_task.keys())
        overall_auroc = _joint_bootstrap_mean(all_tasks, roc_auc_score, rng)
        overall_auprc = _joint_bootstrap_mean(all_tasks, average_precision_score, rng)

        groups_out = {}
        for grp, grp_tasks in GROUPS.items():
            present = [t for t in grp_tasks if t in per_task]
            if not present:
                continue
            groups_out[grp] = {
                "tasks": present,
                "auroc": _joint_bootstrap_mean(present, roc_auc_score, rng),
                "auprc": _joint_bootstrap_mean(present, average_precision_score, rng),
            }

        summary = {
            "n_tasks": len(all_tasks),
            "mean_auroc": overall_auroc["mean"],
            "mean_auprc": overall_auprc["mean"],
            "overall": {"auroc": overall_auroc, "auprc": overall_auprc},
            "groups": groups_out,
        }
        summary_path = mdir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.success(f"Saved {label} summary (mean AUROC={overall_auroc['mean']:.4f}) -> {summary_path}")

    def _merge_existing(mdir: Path, new_data: dict) -> dict:
        """Load existing per_task_metrics.json and merge new results into it."""
        existing_path = mdir / "per_task_metrics.json"
        if existing_path.exists():
            with open(existing_path) as f:
                existing = json.load(f)
            existing.update(new_data)
            return existing
        return new_data

    if metrics_dir and per_task_lasso:
        merged_lasso = _merge_existing(metrics_dir, per_task_lasso)
        _save_metrics_dir(metrics_dir, merged_lasso, "Lasso", raw_preds=raw_lasso)

    if xgb_metrics_dir and per_task_xgb:
        merged_xgb = _merge_existing(xgb_metrics_dir, per_task_xgb)
        _save_metrics_dir(xgb_metrics_dir, merged_xgb, "XGB", raw_preds=raw_xgb)

    if n40_metrics_dir and per_task_lasso_n40:
        merged_lasso_n40 = _merge_existing(n40_metrics_dir, per_task_lasso_n40)
        _save_metrics_dir(n40_metrics_dir, merged_lasso_n40, "Lasso n=40", raw_preds=raw_lasso_n40)

    if n40_xgb_metrics_dir and per_task_xgb_n40:
        merged_xgb_n40 = _merge_existing(n40_xgb_metrics_dir, per_task_xgb_n40)
        _save_metrics_dir(n40_xgb_metrics_dir, merged_xgb_n40, "XGB n=40", raw_preds=raw_xgb_n40)

    if summary_rows:
        print("\n" + "="*70)
        print(f"{'Task':<25}", end="")
        if not args.xgb_only:
            print(f" {'Lasso AUROC':>15} {'Lasso best C':>12}", end="")
        if HAS_XGB:
            print(f" {'XGB AUROC':>15}", end="")
        print()
        print("-"*70)
        for r in summary_rows:
            print(f"{r['task']:<25}", end="")
            if "lasso_test_auroc" in r:
                print(f" {r['lasso_test_auroc']:>15.4f} {r['lasso_best_C']:>12.2e}", end="")
            if HAS_XGB and "xgb_test_auroc" in r:
                print(f" {r['xgb_test_auroc']:>15.4f}", end="")
            print()
        print("="*70)

    logger.success("Done.")


if __name__ == "__main__":
    main()
