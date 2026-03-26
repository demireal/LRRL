#!/usr/bin/env python3
"""
Evaluate embeddings via logistic regression (train -> val -> test).

Unlike the original ehrshot code which used 5-fold cross-validation on
training data, this script uses the validation set for hyperparameter (C)
selection.

Pipeline:
  1. Load train / val / test embeddings (.npz from generate_embeddings.py).
  2. For each C in [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]:
     - Fit LogisticRegression on train, evaluate AUROC on val.
  3. Refit with best C on train, evaluate on test.
  4. Compute bootstrap 95% CIs on test metrics.

Inputs:
  --embeddings_dir : Directory with {task}/{split}.npz files. Use different
                     dirs for different representations (e.g. .../naivetext,
                     .../global-rubric from generate_embeddings.py).
  --output_dir     : Where to write results.
  --tasks          : Space-separated task list (default: all 15).
  --n_train        : If set, filter training embeddings to cohort.
  --cohort_file    : Path to cohort.json (required when --n_train, single file for all tasks).
  --cohort_dir     : Directory with {task}/cohort.json files (alternative to --cohort_file for per-task cohorts).

Outputs:
  {output_dir}/{task}/metrics.json
  {output_dir}/{task}/predictions.csv

Connects to:
  - Upstream  : generate_embeddings.py
  - Downstream: compute_metrics.py (or standalone)
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.tasks import ALL_TASK_NAMES, SEED

# Logarithmic interval from 1e-5 to 1e-1 (inclusive), 10 elements
C_VALUES = np.logspace(-6, -1, 13).tolist()
L1_RATIOS = [0.1, 0.5, 0.9]
BOOTSTRAP_N = 1000


def _load_npz(path: str, return_times: bool = False):
    d = np.load(path, allow_pickle=True)
    emb, pids, labels = d["embeddings"], d["patient_ids"], d["labels"]
    if return_times:
        if "prediction_times" not in d:
            raise ValueError(
                f"npz at {path} lacks prediction_times. "
                "Regenerate embeddings with generate_embeddings.py for n=40 cohort filtering."
            )
        return emb, pids, labels, d["prediction_times"]
    return emb, pids, labels


def _bootstrap_ci(y_true, y_score, metric_fn, n=BOOTSTRAP_N, seed=SEED):
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        scores.append(metric_fn(yt, ys))
    if not scores:
        return 0.0, 0.0, 0.0
    arr = np.array(scores)
    return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def _norm_time(t) -> str:
    """Normalize prediction_time for matching."""
    if t is None or t == "":
        return ""
    if hasattr(t, "isoformat"):
        return t.isoformat()
    s = str(t).strip()
    return s.replace(" ", "T") if " " in s and "T" not in s else s


def _load_cohort_keys(cohort_path: str) -> set:
    """Load cohort.json and return set of (patient_id, prediction_time) tuples."""
    with open(cohort_path) as f:
        cohort = json.load(f)
    return {
        (r["patient_id"], _norm_time(r.get("prediction_time", r.get("label_time", ""))))
        for r in cohort
    }


def evaluate_task(
    emb_dir: str, task: str, output_dir: str,
    n_train: int | None = None, cohort_keys: set | None = None,
    cohort_dir: str | None = None,
):
    task_dir = Path(emb_dir) / task
    train_path = task_dir / "train.npz"
    val_path = task_dir / "val.npz"
    test_path = task_dir / "test.npz"

    if not train_path.exists() or not test_path.exists():
        logger.warning(f"  {task}: missing train or test embeddings, skipping")
        return

    # Need times if we're filtering by cohort (either via cohort_keys or cohort_dir)
    need_times = n_train is not None and (cohort_keys is not None or cohort_dir is not None)
    out = _load_npz(str(train_path), return_times=need_times)
    if need_times:
        X_train, pids_train, y_train, times_train = out
    else:
        X_train, pids_train, y_train = out
        times_train = None

    # Optional: filter to cohort by (patient_id, prediction_time)
    if n_train is not None:
        # If cohort_dir is provided, load per-task cohort file
        if cohort_dir is not None:
            cohort_path = Path(cohort_dir) / task / "cohort.json"
            if cohort_path.exists():
                cohort_keys = _load_cohort_keys(str(cohort_path))
            else:
                logger.warning(f"  {task}: cohort file not found at {cohort_path}, skipping n_train filter")
                cohort_keys = None
        
        if cohort_keys is not None:
            if times_train is None:
                raise ValueError(
                    f"  {task}: cohort filtering requires prediction_times in train.npz, "
                    "but they were not loaded. Regenerate embeddings with generate_embeddings.py."
                )
            mask = np.array([
                (int(pids_train[i]), _norm_time(str(times_train[i]) if i < len(times_train) else "")) in cohort_keys
                for i in range(len(pids_train))
            ])
            X_train, pids_train, y_train = X_train[mask], pids_train[mask], y_train[mask]
            logger.info(f"  filtered train to {len(y_train)} (n_train={n_train}, cohort keys={len(cohort_keys)})")

    # Load test; include prediction_times for CSV if present in npz
    d_test = np.load(str(test_path), allow_pickle=True)
    X_test = d_test["embeddings"]
    pids_test = d_test["patient_ids"]
    y_test = d_test["labels"]
    times_test = d_test["prediction_times"] if "prediction_times" in d_test else None

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --- Hyperparameter choice for C / l1_ratio (elastic net, L1, or L2) ---
    # Default behavior: hyperparameter optimization on VALIDATION SET ONLY
    # using L2 regularization.
    #
    # Overrides (easy to revert, controlled via env vars only):
    #   - FIXED_LOGREG_C: always use this C (elastic net with fixed l1_ratio)
    #                     and skip model selection.
    #   - USE_LOGREG_CV: if set (e.g. "1"), ignore val split and select C via
    #                    K-fold cross-validation on train.
    #   - USE_L1_REGULARIZATION: if set (e.g. "1"), use L1 (Lasso) instead of L2.
    #   - USE_VAL_LOG_LOSS: if set (e.g. "1"), select best C by minimizing val log loss.
    best_c = C_VALUES[0]
    best_val_auc = -1.0
    best_val_logloss = float("inf")
    best_l1_ratio = L1_RATIOS[0]
    # By default we use L2 + external validation set
    use_elastic_net = False
    use_l1 = os.environ.get("USE_L1_REGULARIZATION") is not None

    # Per-task fixed C (e.g. FIXED_LOGREG_C_ACUTEMI=1e-5 for new_acutemi)
    fixed_c_acutemi = os.environ.get("FIXED_LOGREG_C_ACUTEMI")
    if task == "new_acutemi" and fixed_c_acutemi is not None:
        try:
            best_c = float(fixed_c_acutemi)
        except ValueError:
            raise ValueError(
                f"FIXED_LOGREG_C_ACUTEMI={fixed_c_acutemi!r} is not a valid float."
            )
        logger.info(f"  {task}: using fixed C={best_c} from FIXED_LOGREG_C_ACUTEMI")
        # Load val and compute val AUROC for this C (for metrics)
        if val_path.exists():
            X_val, _, y_val = _load_npz(str(val_path))
            if task == "new_acutemi":
                rng = np.random.RandomState(SEED)
                n_val = len(y_val)
                n_keep = max(1, int(round(0.8 * n_val)))
                idx = rng.permutation(n_val)[:n_keep]
                X_val = X_val[idx]
                y_val = y_val[idx]
            X_val_s = scaler.transform(X_val)
            penalty_type = "l1" if use_l1 else "l2"
            solver_type = "liblinear" if use_l1 else "lbfgs"
            clf_val = LogisticRegression(
                C=best_c,
                penalty=penalty_type,
                solver=solver_type,
                max_iter=1000,
                class_weight="balanced",
                random_state=SEED,
            )
            clf_val.fit(X_train_s, y_train)
            if len(np.unique(y_val)) >= 2:
                y_val_score = clf_val.predict_proba(X_val_s)[:, 1]
                best_val_auc = float(roc_auc_score(y_val, y_val_score))
                logger.info(f"  val AUROC={best_val_auc:.4f} (fixed C)")
    elif os.environ.get("FIXED_LOGREG_C") is not None:
        fixed_c_env = os.environ.get("FIXED_LOGREG_C")
        try:
            best_c = float(fixed_c_env)
        except ValueError:
            raise ValueError(
                f"Environment variable FIXED_LOGREG_C={fixed_c_env!r} is not a valid float."
            )
        # When using fixed C, we still use elastic net but with a fixed l1_ratio
        # for simplicity and reproducibility.
        best_l1_ratio = 0.5
        logger.info(
            f"  using fixed C={best_c}, l1_ratio={best_l1_ratio} from FIXED_LOGREG_C; "
            "skipping hyperparameter search"
        )
    elif os.environ.get("USE_LOGREG_CV") is not None:
        # Hyperparameter selection via K-fold cross-validation on train only,
        # using standard L2-penalized logistic regression. This does not touch
        # the test set and ignores any val split.
        # By default uses AUROC for model selection,
        # but can use log_loss if CV_METRIC=logloss is set.
        use_elastic_net = False
        n_splits = int(os.environ.get("LOGREG_CV_FOLDS", "3"))
        cv_metric = os.environ.get("CV_METRIC", "auroc").lower()
        use_logloss = cv_metric in ("logloss", "log_loss", "neg_log_likelihood")
        
        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=SEED,
        )
        best_cv_score = float('inf') if use_logloss else -1.0
        for c in C_VALUES:
            fold_scores = []
            for train_idx, val_idx in cv.split(X_train_s, y_train):
                X_tr, X_val_cv = X_train_s[train_idx], X_train_s[val_idx]
                y_tr, y_val_cv = y_train[train_idx], y_train[val_idx]
                if len(np.unique(y_val_cv)) < 2:
                    continue
                penalty_type = "l1" if use_l1 else "l2"
                solver_type = "liblinear" if use_l1 else "lbfgs"
                clf = LogisticRegression(
                    C=c,
                    penalty=penalty_type,
                    solver=solver_type,
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=SEED,
                )
                clf.fit(X_tr, y_tr)
                y_score_cv = clf.predict_proba(X_val_cv)[:, 1]
                if use_logloss:
                    fold_scores.append(log_loss(y_val_cv, y_score_cv, labels=[0, 1]))
                else:
                    fold_scores.append(roc_auc_score(y_val_cv, y_score_cv))
            if fold_scores:
                mean_score = float(np.mean(fold_scores))
                metric_name = "CV log_loss" if use_logloss else "CV AUROC"
                logger.info(
                    f"  C={c}: {metric_name}={mean_score:.4f} over {len(fold_scores)} folds"
                )
                if (use_logloss and mean_score < best_cv_score) or (not use_logloss and mean_score > best_cv_score):
                    best_cv_score = mean_score
                    best_c = c

        metric_name = "log_loss" if use_logloss else "AUROC"
        logger.info(
            f"  best C={best_c} (CV {metric_name}={best_cv_score:.4f}, folds={n_splits})"
        )
        # For reporting consistency, store CV score in val_auroc field.
        # If using log_loss, convert to a "pseudo-AUROC" by negating (so higher is better)
        best_val_auc = -best_cv_score if use_logloss else best_cv_score
    elif val_path.exists():
        # Fit on train, score on external validation set; best C is chosen by
        # val AUROC. Test is never used for model or hyperparameter selection.
        # We also record the validation AUROC and log loss for each C.
        X_val, _, y_val = _load_npz(str(val_path))
        # For acute_mi (new_acutemi), use a random 80% of the validation set
        if task == "new_acutemi":
            rng = np.random.RandomState(SEED)
            n_val = len(y_val)
            n_keep = max(1, int(round(0.8 * n_val)))
            idx = rng.permutation(n_val)[:n_keep]
            X_val = X_val[idx]
            y_val = y_val[idx]
            logger.info(f"  {task}: subsampled val to {n_keep}/{n_val} (80%)")
        X_val_s = scaler.transform(X_val)
        val_curve = []
        penalty_type = "l1" if use_l1 else "l2"
        solver_type = "liblinear" if use_l1 else "lbfgs"
        for c in C_VALUES:
            clf = LogisticRegression(
                C=c,
                penalty=penalty_type,
                solver=solver_type,
                max_iter=1000,
                class_weight="balanced",
                random_state=SEED,
            )
            clf.fit(X_train_s, y_train)
            if len(np.unique(y_val)) < 2:
                continue
            y_val_score = clf.predict_proba(X_val_s)[:, 1]
            val_auc = roc_auc_score(y_val, y_val_score)
            val_loss = log_loss(y_val, y_val_score, labels=[0, 1])
            val_curve.append(
                {
                    "C": float(c),
                    "val_auroc": float(val_auc),
                    "val_logloss": float(val_loss),
                }
            )
            use_val_logloss = os.environ.get("USE_VAL_LOG_LOSS", "").lower() in ("1", "true", "yes")
            if use_val_logloss:
                if val_loss < best_val_logloss:
                    best_val_logloss = val_loss
                    best_c = c
                    best_val_auc = val_auc
            else:
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_c = c
        use_val_logloss = os.environ.get("USE_VAL_LOG_LOSS", "").lower() in ("1", "true", "yes")
        if use_val_logloss:
            logger.info(f"  best C={best_c} (val log loss={best_val_logloss:.4f}, val AUROC={best_val_auc:.4f})")
        else:
            logger.info(f"  best C={best_c} (val AUROC={best_val_auc:.4f})")
    else:
        logger.info(f"  no val split, using C={best_c}")

    # --- Final evaluation: TEST SET ONLY ---
    # Refit with best hyperparameters on train; report AUROC/AUPRC and bootstrap CIs on test.
    if use_elastic_net:
        clf = LogisticRegression(
            C=best_c,
            penalty="elasticnet",
            l1_ratio=best_l1_ratio,
            solver="saga",
            max_iter=1000,
            class_weight="balanced",
            random_state=SEED,
        )
    elif use_l1:
        clf = LogisticRegression(
            C=best_c,
            penalty="l1",
            solver="liblinear",
            max_iter=1000,
            class_weight="balanced",
            random_state=SEED,
        )
    else:
        clf = LogisticRegression(
            C=best_c,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=SEED,
        )
    clf.fit(X_train_s, y_train)
    y_score = clf.predict_proba(X_test_s)[:, 1]

    auroc, auroc_lo, auroc_hi = _bootstrap_ci(y_test, y_score, roc_auc_score)
    auprc, auprc_lo, auprc_hi = _bootstrap_ci(y_test, y_score, average_precision_score)

    metrics = {
        "task": task,
        "best_c": best_c,
        "best_l1_ratio": best_l1_ratio,
        "val_auroc": best_val_auc,
        "test_auroc": auroc,
        "test_auroc_ci": [auroc_lo, auroc_hi],
        "test_auprc": auprc,
        "test_auprc_ci": [auprc_lo, auprc_hi],
        "n_train": len(y_train),
        "n_test": len(y_test),
    }
    # If we ran external-validation-based hyperparameter search, attach the
    # full validation curve (per-C AUROC and log loss) for analysis.
    if "val_curve" in locals():
        metrics["val_curve"] = val_curve

    out = Path(output_dir) / task
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(out / "predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "patient_id", "label_time", "ground_truth",
            "probability_score", "target_task"])
        w.writeheader()
        for i, (pid, gt, ps) in enumerate(zip(pids_test, y_test, y_score)):
            label_time = ""
            if times_test is not None and i < len(times_test):
                t = times_test[i]
                label_time = _norm_time(t) if t is not None and t != "" else ""
            w.writerow({"patient_id": int(pid), "label_time": label_time,
                        "ground_truth": int(gt),
                        "probability_score": float(ps),
                        "target_task": task})

    logger.info(f"  {task}: AUROC={auroc:.4f} [{auroc_lo:.4f}, {auroc_hi:.4f}]  "
                f"AUPRC={auprc:.4f} [{auprc_lo:.4f}, {auprc_hi:.4f}]")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embeddings_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tasks", nargs="+", default=ALL_TASK_NAMES)
    p.add_argument("--n_train", type=int, default=None)
    p.add_argument("--cohort_file", default=None)
    p.add_argument("--cohort_dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cohort_keys = None
    if args.n_train is not None:
        if args.cohort_file is None and args.cohort_dir is None:
            raise ValueError("--cohort_file or --cohort_dir required when --n_train is set")
        if args.cohort_file is not None and args.cohort_dir is not None:
            raise ValueError("Cannot specify both --cohort_file and --cohort_dir")
        if args.cohort_file is not None:
            cohort_keys = _load_cohort_keys(args.cohort_file)

    for task in args.tasks:
        logger.info(f"\nEvaluating embeddings for: {task}")
        evaluate_task(args.embeddings_dir, task, args.output_dir,
                      args.n_train, cohort_keys, args.cohort_dir)

    logger.success("All embedding evaluations complete.")


if __name__ == "__main__":
    main()
