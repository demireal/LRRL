#!/usr/bin/env python3
"""
Evaluate count-GBM features for all 15 tasks using logistic regression head.

Same train/val/test splits as CLMBR: data/sft/naivetext/{split}/{task}.json.
Supports n=full and n=40 (cohort filter).
"""

import os
import sys
import json
import csv
import argparse
import pickle
import numpy as np
import pandas as pd
import collections
from typing import Dict, List, Tuple, Optional
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import scipy
from scipy.sparse import issparse
import sklearn.utils

_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(_script_dir))
parent_root = os.path.dirname(project_root)
for p in (project_root, parent_root):
    if p not in sys.path:
        sys.path.insert(0, p)

os.chdir(project_root)

from config.tasks import ALL_TASK_NAMES


def _align_features_labels_exact_then_closest(label_pids, label_dates, feature_pids, feature_dates):
    """Match labels to features: exact (patient_id, label_time) when available,
    else use closest feature (same patient)."""
    feat_pids = np.asarray(feature_pids).astype(np.int64)
    feat_times = np.asarray(feature_dates).astype(np.int64)
    lookup = {}
    for i in range(len(feat_pids)):
        key = (int(feat_pids[i]), int(feat_times[i]))
        if key not in lookup:
            lookup[key] = i

    feature_indices = []
    label_indices = []
    n_exact, n_fallback = 0, 0
    for i in range(len(label_pids)):
        pid, ld = int(label_pids[i]), int(np.int64(label_dates[i]))
        key = (pid, ld)
        if key in lookup:
            feature_indices.append(lookup[key])
            label_indices.append(i)
            n_exact += 1
        else:
            mask = feat_pids == pid
            if not np.any(mask):
                raise RuntimeError(f"No features for patient {pid}")
            feats_idx = np.where(mask)[0]
            deltas = np.abs(feat_times[mask] - ld)
            j = feats_idx[np.argmin(deltas)]
            feature_indices.append(j)
            label_indices.append(i)
            n_fallback += 1
    if n_fallback > 0:
        logger.info(f"Match: {n_exact} exact, {n_fallback} fallback (closest same-patient)")
    return np.array(feature_indices, dtype=np.uint32), np.array(label_indices, dtype=np.uint32)


# Hyperparam search only over C (same grid as evaluate_multi_task_count.py)
LR_PARAMS = {'C': [0.01, 0.1, 1.0, 10.0, 100.0]}


def load_count_features(features_file: str) -> Tuple:
    """Load count-GBM features from pickle. Returns (feature_matrix, patient_ids, feature_times)."""
    logger.info(f"Loading count-GBM features from: {features_file}")
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"Features file not found: {features_file}")

    with open(features_file, 'rb') as f:
        raw_feats = pickle.load(f)

    if isinstance(raw_feats, dict):
        feature_matrix = raw_feats['data_matrix']
        feature_patient_ids = raw_feats['patient_ids']
        feature_times = raw_feats.get('labeling_time', raw_feats.get('prediction_time'))
        if feature_times is None:
            raise KeyError("Count pkl must have 'labeling_time' or 'prediction_time'")
    elif len(raw_feats) >= 4:
        feature_matrix, feature_patient_ids, _, feature_times = raw_feats[:4]
    else:
        raise ValueError(f"Unexpected features format: {type(raw_feats)}")

    if isinstance(feature_matrix, np.ndarray) and feature_matrix.ndim == 0 and feature_matrix.dtype == object:
        feature_matrix = feature_matrix.item()

    logger.info(f"Loaded: matrix={feature_matrix.shape}, pids={feature_patient_ids.shape}, times={feature_times.shape}")
    return feature_matrix, feature_patient_ids, feature_times


def load_task_labels_from_json(task_name: str, data_dir: str) -> Tuple[Dict, Dict, Dict]:
    """Load labels from data/sft/naivetext/{split}/{task}.json."""
    splits = ['train', 'val', 'test']
    patient_ids_by_split = {}
    label_values_by_split = {}
    label_times_by_split = {}

    for split in splits:
        json_path = os.path.join(data_dir, 'data', 'sft', 'naivetext', split, f'{task_name}.json')
        if not os.path.exists(json_path):
            json_path_alt = os.path.join(data_dir, 'data', 'serialized', 'naivetext', task_name, f'{split}.json')
            json_path = json_path_alt if os.path.exists(json_path_alt) else json_path

        if not os.path.exists(json_path):
            patient_ids_by_split[split] = np.array([], dtype=np.int64)
            label_values_by_split[split] = np.array([], dtype=bool)
            label_times_by_split[split] = np.array([], dtype='datetime64[us]')
            continue

        logger.info(f"Loading {task_name} {split} from: {json_path}")
        with open(json_path) as f:
            data = json.load(f)

        pids, vals, times = [], [], []
        for item in data:
            pids.append(int(item['patient_id']))
            vals.append(bool(item['label_value']))
            t = item['label_time']
            times.append(pd.to_datetime(t).asm8.astype("datetime64[us]"))

        patient_ids_by_split[split] = np.array(pids, dtype=np.int64)
        label_values_by_split[split] = np.array(vals, dtype=bool)
        label_times_by_split[split] = np.array(times, dtype="datetime64[us]")
        logger.info(f"  {split}: n={len(pids)} pos={np.sum(vals)}")

    return patient_ids_by_split, label_values_by_split, label_times_by_split


def match_features_with_labels_by_split(
    feature_matrix, feature_patient_ids, feature_times,
    label_patient_ids_by_split, label_values_by_split, label_times_by_split,
):
    """Match features with labels per split. Handles sparse feature matrix."""
    X_by_split, y_by_split = {}, {}
    out_patient_ids_by_split, out_label_times_by_split = {}, {}

    for split in ['train', 'val', 'test']:
        if split not in label_patient_ids_by_split:
            label_patient_ids_by_split[split] = np.array([], dtype=np.int64)
        if split not in label_values_by_split:
            label_values_by_split[split] = np.array([], dtype=bool)
        if split not in label_times_by_split:
            label_times_by_split[split] = np.array([], dtype='datetime64[us]')

    feature_times_dt = feature_times.astype("datetime64[us]")
    feature_sort_order = np.lexsort((feature_times_dt, feature_patient_ids))
    feature_patient_ids_sorted = feature_patient_ids[feature_sort_order]
    feature_times_dt_sorted = feature_times_dt[feature_sort_order]

    for split in ['train', 'val', 'test']:
        label_pids = np.asarray(label_patient_ids_by_split[split]).ravel()
        label_vals = np.asarray(label_values_by_split[split]).ravel()
        label_times = np.asarray(label_times_by_split[split]).ravel()
        if len(label_times) > 0 and str(label_times.dtype) != 'datetime64[us]':
            label_times = pd.to_datetime(label_times).values.astype("datetime64[us]")

        if len(label_pids) == 0:
            n_feat = feature_matrix.shape[1]
            X_by_split[split] = scipy.sparse.csr_matrix((0, n_feat)) if issparse(feature_matrix) else np.array([]).reshape(0, n_feat)
            y_by_split[split] = np.array([], dtype=bool)
            out_patient_ids_by_split[split] = np.array([], dtype=np.int64)
            out_label_times_by_split[split] = np.array([], dtype='datetime64[us]')
            continue

        label_sort = np.lexsort((label_times.astype(np.int64), label_pids))
        label_pids_sorted = label_pids[label_sort]
        label_vals_sorted = label_vals[label_sort]
        label_times_sorted = label_times[label_sort]

        feat_idx, lbl_idx = _align_features_labels_exact_then_closest(
            label_pids_sorted, label_times_sorted.astype(np.int64),
            feature_patient_ids_sorted, feature_times_dt_sorted.astype(np.int64),
        )

        orig_idx = feature_sort_order[feat_idx]
        if issparse(feature_matrix):
            X_split = feature_matrix[orig_idx]
        else:
            X_split = feature_matrix[orig_idx]

        X_by_split[split] = X_split
        y_by_split[split] = label_vals_sorted[lbl_idx]
        out_patient_ids_by_split[split] = label_pids_sorted[lbl_idx]
        out_label_times_by_split[split] = label_times_sorted[lbl_idx]
        logger.info(f"Matched {split}: X={X_split.shape}, y={y_by_split[split].shape}")

    return X_by_split, y_by_split, out_patient_ids_by_split, out_label_times_by_split


def _norm_time(t) -> str:
    if t is None or t == "":
        return ""
    if hasattr(t, "isoformat"):
        return t.isoformat()
    s = str(t).strip()
    return s.replace(" ", "T") if " " in s and "T" not in s else s


def _load_cohort_keys(cohort_path: str) -> set:
    with open(cohort_path) as f:
        cohort = json.load(f)
    return {(r["patient_id"], _norm_time(r.get("prediction_time", r.get("label_time", "")))) for r in cohort}


def tune_logistic_regression(X_train, X_val, y_train, y_val, n_jobs=1):
    """Select C by fitting on train and maximizing validation AUROC on val."""
    scaler = MaxAbsScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)

    best_C = None
    best_val_auc = -np.inf
    for C in LR_PARAMS["C"]:
        model = LogisticRegression(
            C=C, n_jobs=1, penalty="l2", tol=0.0001, solver="lbfgs", max_iter=10000, random_state=0
        )
        model.fit(X_train_s, y_train)
        val_auc = float(roc_auc_score(y_val, model.predict_proba(X_val_s)[:, 1]))
        logger.info(f"C={C}: val AUROC={val_auc:.4f}")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_C = C
    logger.info(f"Best C={best_C}, val AUROC: {best_val_auc:.4f}")

    best = LogisticRegression(
        C=best_C, n_jobs=1, penalty="l2", tol=0.0001, solver="lbfgs", max_iter=10000, random_state=0
    )
    best.fit(X_train_s, y_train)
    return best, scaler, best_val_auc


def evaluate_model(model, scaler, X_train, X_val, X_test, y_train, y_val, y_test, test_patient_ids):
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    y_train_p = model.predict_proba(X_train_s)[:, 1]
    y_val_p = model.predict_proba(X_val_s)[:, 1]
    y_test_p = model.predict_proba(X_test_s)[:, 1]

    scores = {}
    for name, func in [('auroc', roc_auc_score), ('auprc', average_precision_score), ('brier', brier_score_loss)]:
        scores[name] = {'score': func(y_test, y_test_p)}
        logger.info(f"Test {name}: {scores[name]['score']:.4f}")

        test_set = sorted(list(set(test_patient_ids)))
        boot = []
        for i in range(1000):
            sample = sklearn.utils.resample(test_set, random_state=i)
            counts = collections.Counter(sample)
            w = np.array([counts.get(p, 0) for p in test_patient_ids])
            boot.append(func(y_test, y_test_p, sample_weight=w))
        scores[name]['lower'] = float(np.percentile(boot, 2.5))
        scores[name]['upper'] = float(np.percentile(boot, 97.5))

    return scores, y_test_p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", required=True, choices=ALL_TASK_NAMES)
    parser.add_argument("--path_to_count_features", required=True)
    parser.add_argument("--path_to_data_dir", required=True)
    parser.add_argument("--path_to_output_dir", required=True)
    parser.add_argument("--n_train", type=int, default=None)
    parser.add_argument("--cohort_dir", default=None)
    parser.add_argument("--cohort_subcohort_dir", default=None,
                        help="If set, filter all splits to subcohort: {dir}/{task}/train.json, val.json, test.json (e.g. data/cohort_7d or data/cohort_14d).")
    parser.add_argument("--num_threads", type=int, default=1)
    args = parser.parse_args()

    if not os.path.isabs(args.path_to_count_features):
        args.path_to_count_features = os.path.join(project_root, args.path_to_count_features)
    if not os.path.isabs(args.path_to_data_dir):
        args.path_to_data_dir = os.path.join(project_root, args.path_to_data_dir)
    if not os.path.isabs(args.path_to_output_dir):
        args.path_to_output_dir = os.path.join(project_root, args.path_to_output_dir)
    if args.cohort_dir and not os.path.isabs(args.cohort_dir):
        args.cohort_dir = os.path.join(project_root, args.cohort_dir)
    if args.cohort_subcohort_dir and not os.path.isabs(args.cohort_subcohort_dir):
        args.cohort_subcohort_dir = os.path.join(project_root, args.cohort_subcohort_dir)

    if args.n_train and not args.cohort_dir:
        raise ValueError("--cohort_dir required when --n_train is set")

    logger.info(f"Count-GBM Evaluation: {args.task_name}, n_train={args.n_train or 'all'}")

    feature_matrix, feature_patient_ids, feature_times = load_count_features(args.path_to_count_features)
    label_pids, label_vals, label_times = load_task_labels_from_json(args.task_name, args.path_to_data_dir)

    X_by_split, y_by_split, patient_ids_by_split, label_times_by_split = match_features_with_labels_by_split(
        feature_matrix, feature_patient_ids, feature_times,
        label_pids, label_vals, label_times,
    )

    for split in ('train', 'val', 'test'):
        n, pos = len(y_by_split[split]), int(np.sum(y_by_split[split]))
        print(f"[{args.task_name}] {split}: n={n} pos={pos}", flush=True)

    # Filter all splits to subcohort when --cohort_subcohort_dir is set (e.g. 7d or 14d)
    if args.cohort_subcohort_dir is not None:
        for split in ['train', 'val', 'test']:
            cohort_path = os.path.join(args.cohort_subcohort_dir, args.task_name, f"{split}.json")
            if not os.path.exists(cohort_path):
                raise FileNotFoundError(f"Subcohort file not found: {cohort_path}")
            cohort_keys = _load_cohort_keys(cohort_path)
            pids = patient_ids_by_split[split]
            times = label_times_by_split[split]
            mask = np.array([
                (int(pids[i]), _norm_time(str(pd.Timestamp(times[i])))) in cohort_keys
                for i in range(len(pids))
            ])
            X_by_split[split] = X_by_split[split][mask]
            y_by_split[split] = y_by_split[split][mask]
            patient_ids_by_split[split] = patient_ids_by_split[split][mask]
            label_times_by_split[split] = label_times_by_split[split][mask]
        logger.info("Filtered all splits to subcohort")
        for split in ('train', 'val', 'test'):
            n, pos = len(y_by_split[split]), int(np.sum(y_by_split[split]))
            print(f"[{args.task_name}] {split} (subcohort): n={n} pos={pos}", flush=True)

    if len(y_by_split['train']) == 0:
        raise ValueError(f"No training data for {args.task_name}")
    if len(y_by_split['val']) == 0:
        X_by_split['val'] = X_by_split['train']
        y_by_split['val'] = y_by_split['train']
    if len(y_by_split['test']) == 0:
        raise ValueError(f"No test data for {args.task_name}")

    if args.n_train is not None:
        cohort_path = os.path.join(args.cohort_dir, args.task_name, "cohort.json")
        if not os.path.exists(cohort_path):
            raise FileNotFoundError(f"Cohort not found: {cohort_path}")
        cohort_keys = _load_cohort_keys(cohort_path)
        train_pids = patient_ids_by_split['train']
        train_times = label_times_by_split['train']
        mask = np.array([
            (int(train_pids[i]), _norm_time(str(pd.Timestamp(train_times[i])))) in cohort_keys
            for i in range(len(train_pids))
        ])
        X_by_split['train'] = X_by_split['train'][mask]
        y_by_split['train'] = y_by_split['train'][mask]
        patient_ids_by_split['train'] = patient_ids_by_split['train'][mask]
        label_times_by_split['train'] = label_times_by_split['train'][mask]
        logger.info(f"Filtered train to {np.sum(mask)} (cohort)")

    model, scaler, best_val_auc = tune_logistic_regression(
        X_by_split['train'], X_by_split['val'], y_by_split['train'], y_by_split['val'], n_jobs=args.num_threads
    )
    scores, test_proba = evaluate_model(
        model, scaler,
        X_by_split['train'], X_by_split['val'], X_by_split['test'],
        y_by_split['train'], y_by_split['val'], y_by_split['test'],
        patient_ids_by_split['test']
    )

    task_out = os.path.join(args.path_to_output_dir, args.task_name)
    os.makedirs(task_out, exist_ok=True)

    metrics = {
        "task": args.task_name,
        "best_c": model.get_params().get('C', 1.0),
        "best_l1_ratio": 0.0,
        "val_auroc": best_val_auc,
        "test_auroc": scores["auroc"]["score"],
        "test_auroc_ci": [scores["auroc"]["lower"], scores["auroc"]["upper"]],
        "test_auprc": scores["auprc"]["score"],
        "test_auprc_ci": [scores["auprc"]["lower"], scores["auprc"]["upper"]],
        "n_train": int(len(y_by_split['train'])),
        "n_test": int(len(y_by_split['test'])),
    }
    with open(os.path.join(task_out, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    test_pids = patient_ids_by_split['test']
    test_times = label_times_by_split['test']
    test_gt = y_by_split['test']
    with open(os.path.join(task_out, "predictions.csv"), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "label_time", "ground_truth", "probability_score", "target_task"])
        w.writeheader()
        for i in range(len(test_pids)):
            w.writerow({
                "patient_id": int(test_pids[i]),
                "label_time": _norm_time(pd.Timestamp(test_times[i])) if i < len(test_times) else "",
                "ground_truth": int(test_gt[i]),
                "probability_score": float(test_proba[i]),
                "target_task": args.task_name,
            })

    logger.success("Done!")


if __name__ == "__main__":
    main()
