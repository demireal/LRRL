#!/usr/bin/env python3
"""
Evaluate CLMBR-T features for all 15 tasks using logistic regression head.

This script:
1. Loads CLMBR-T features from pickle file
2. Loads labels from JSON files in data/sft/naivetext/{task_name}.json (train/val/test splits)
3. Matches features with labels by (patient_id, label_time)
4. Trains a logistic regression head on the training set
5. Evaluates on train/val/test sets with confidence intervals

Each element in the JSON files is identified by (patient_id, label_time).
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
from datetime import datetime
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import scipy
from scipy.sparse import issparse
import sklearn.utils

# Add project root and parent (for ehrshot package) to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(_script_dir))  # ehrshot-v2
parent_root = os.path.dirname(project_root)  # ehrshot-benchmark (contains ehrshot/)
for p in (project_root, parent_root):
    if p not in sys.path:
        sys.path.insert(0, p)

# Change to project root directory for imports
os.chdir(project_root)

# Import from config
from config.tasks import ALL_TASK_NAMES

def _align_features_labels_exact_then_closest(label_pids, label_dates, feature_pids, feature_dates):
    """Match labels to features: exact (patient_id, label_time) when available,
    else use closest feature (same patient). Keeps all labels."""
    feat_pids = np.asarray(feature_pids).astype(np.int64)
    feat_times = np.asarray(feature_dates).astype(np.int64)
    # Build exact-match lookup
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
            # Fallback: closest feature for same patient
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

# Logistic regression hyperparameters (match evaluate_multi_task_clmbr.py)
LR_PARAMS = {
    'C': [0.01, 0.1, 1.0, 10.0, 100.0],
}


def load_clmbr_features(features_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load CLMBR-T features from pickle file.
    
    Returns:
        feature_matrix: Feature matrix (n_samples, n_features)
        feature_patient_ids: Patient IDs (n_samples,)
        feature_times: Labeling times (n_samples,)
    """
    logger.info(f"Loading CLMBR-T features from: {features_file}")
    
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"Features file not found: {features_file}")
    
    with open(features_file, 'rb') as f:
        raw_feats = pickle.load(f)
    
    # Handle different pickle formats (femr may use 'labeling_time' or 'prediction_time')
    if isinstance(raw_feats, dict):
        feature_matrix = raw_feats['data_matrix']
        feature_patient_ids = raw_feats['patient_ids']
        feature_times = raw_feats.get('labeling_time', raw_feats.get('prediction_time'))
        if feature_times is None:
            raise KeyError("CLMBR pkl must have 'labeling_time' or 'prediction_time'")
    elif len(raw_feats) == 4:
        # Format: (feature_matrix, patient_ids, label_values, times)
        feature_matrix, feature_patient_ids, _, feature_times = raw_feats
    elif len(raw_feats) == 5:
        # Format: (feature_matrix, patient_ids, label_values, times, tasks)
        feature_matrix, feature_patient_ids, _, feature_times, _ = raw_feats
    else:
        raise ValueError(f"Unexpected number of elements in features tuple: {len(raw_feats)}")
    
    logger.info(f"Loaded CLMBR-T features:")
    logger.info(f"  Feature matrix shape: {feature_matrix.shape}")
    logger.info(f"  Patient IDs shape: {feature_patient_ids.shape}")
    logger.info(f"  Times shape: {feature_times.shape}")
    
    return feature_matrix, feature_patient_ids, feature_times


def load_task_labels_from_json(task_name: str, data_dir: str) -> Tuple[
    Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]
]:
    """Load labels for a specific task from JSON files (train/val/test splits).
    
    Args:
        task_name: Task name (e.g., 'chexpert', 'guo_icu')
        data_dir: Directory containing data/sft/naivetext/{split}/{task_name}.json
    
    Returns:
        Tuple of (patient_ids_by_split, label_values_by_split, label_times_by_split)
        Each is a dict with keys 'train', 'val', 'test'
    """
    splits = ['train', 'val', 'test']
    patient_ids_by_split = {}
    label_values_by_split = {}
    label_times_by_split = {}
    
    for split in splits:
        # Try data/sft/naivetext/{split}/{task}.json (same layout as generate_embeddings, 02_create_sft)
        json_path = os.path.join(data_dir, 'data', 'sft', 'naivetext', split, f'{task_name}.json')
        if not os.path.exists(json_path):
            # Fallback: data/serialized/naivetext/{task}/{split}.json (03_globalrubric layout)
            json_path_alt = os.path.join(data_dir, 'data', 'serialized', 'naivetext', task_name, f'{split}.json')
            if os.path.exists(json_path_alt):
                json_path = json_path_alt
            else:
                logger.warning(f"Label file not found for {task_name} {split}: {json_path} (tried {json_path_alt})")
                patient_ids_by_split[split] = np.array([], dtype=np.int64)
                label_values_by_split[split] = np.array([], dtype=bool)
                label_times_by_split[split] = np.array([], dtype='datetime64[us]')
                continue
        
        logger.info(f"Loading labels for {task_name} {split} from: {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        patient_ids = []
        label_values = []
        label_times = []
        
        for item in data:
            patient_id = item['patient_id']
            label_time_str = item['label_time']
            label_value = item['label_value']
            
            # Convert label_time to datetime64[us] (must match feature times for alignment)
            try:
                label_time_dt = np.datetime64(label_time_str).astype("datetime64[us]")
            except Exception:
                label_time_dt = pd.to_datetime(label_time_str).asm8.astype("datetime64[us]")

            patient_ids.append(int(patient_id))
            label_values.append(bool(label_value))
            label_times.append(label_time_dt)

        patient_ids_by_split[split] = np.array(patient_ids, dtype=np.int64)
        label_values_by_split[split] = np.array(label_values, dtype=bool)
        # Convert each label_time to datetime64[us] (handles str, pd.Timestamp, np.datetime64)
        label_times_arr = np.array(
            [pd.to_datetime(t).asm8.astype("datetime64[us]") for t in label_times],
            dtype="datetime64[us]"
        )
        label_times_by_split[split] = label_times_arr
        
        logger.info(f"  {split}: {len(patient_ids)} labels, "
                   f"positive: {np.sum(label_values_by_split[split])}/{len(label_values_by_split[split])} "
                   f"({np.mean(label_values_by_split[split])*100:.1f}%)")
    
    return patient_ids_by_split, label_values_by_split, label_times_by_split


def load_task_labels_from_embeddings(
    task_name: str,
    embeddings_dir: str,
) -> Tuple[
    Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]
]:
    """Load labels from embeddings npz files by (patient_id, prediction_time).

    Same format as eval_embeddings.py: {embeddings_dir}/{task}/{split}.npz with keys
    embeddings, patient_ids, labels, prediction_times. Uses (patient_id, prediction_time)
    for matching to CLMBR features (labeling_time / prediction_time in clmbr representations).
    """
    splits = ['train', 'val', 'test']
    patient_ids_by_split = {}
    label_values_by_split = {}
    label_times_by_split = {}

    task_dir = os.path.join(embeddings_dir, task_name)
    for split in splits:
        npz_path = os.path.join(task_dir, f'{split}.npz')
        if not os.path.exists(npz_path):
            logger.warning(f"Embeddings npz not found for {task_name} {split}: {npz_path}")
            patient_ids_by_split[split] = np.array([], dtype=np.int64)
            label_values_by_split[split] = np.array([], dtype=bool)
            label_times_by_split[split] = np.array([], dtype='datetime64[us]')
            continue

        d = np.load(npz_path, allow_pickle=True)
        if 'prediction_times' not in d:
            raise ValueError(f"npz at {npz_path} lacks prediction_times (required for matching)")

        patient_ids_by_split[split] = np.array(d['patient_ids'], dtype=np.int64)
        label_values_by_split[split] = np.array(d['labels']).astype(bool)
        times_raw = d['prediction_times']
        label_times_by_split[split] = np.array(
            [pd.to_datetime(t).asm8.astype("datetime64[us]") for t in times_raw],
            dtype="datetime64[us]"
        )
        logger.info(f"Loading labels for {task_name} {split} from: {npz_path}")
        logger.info(f"  {split}: {len(patient_ids_by_split[split])} labels, "
                   f"positive: {np.sum(label_values_by_split[split])}/{len(label_values_by_split[split])} "
                   f"({np.mean(label_values_by_split[split])*100:.1f}%)")

    return patient_ids_by_split, label_values_by_split, label_times_by_split


def match_features_with_labels_by_split(
    feature_matrix: np.ndarray,
    feature_patient_ids: np.ndarray,
    feature_times: np.ndarray,
    label_patient_ids_by_split: Dict[str, np.ndarray],
    label_values_by_split: Dict[str, np.ndarray],
    label_times_by_split: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Match CLMBR features with labels for each split (train/val/test).
    
    Returns:
        X_by_split: Feature matrices by split
        y_by_split: Label values by split
        patient_ids_by_split: Patient IDs by split
        label_times_by_split: Label times by split
    """
    X_by_split = {}
    y_by_split = {}
    out_patient_ids_by_split = {}
    out_label_times_by_split = {}

    # Ensure all splits exist (defensive: load may skip a split if file missing)
    for split in ['train', 'val', 'test']:
        if split not in label_patient_ids_by_split:
            label_patient_ids_by_split[split] = np.array([], dtype=np.int64)
        if split not in label_values_by_split:
            label_values_by_split[split] = np.array([], dtype=bool)
        if split not in label_times_by_split:
            label_times_by_split[split] = np.array([], dtype='datetime64[us]')

    # Convert feature times to datetime64
    feature_times_dt = feature_times.astype("datetime64[us]")
    
    # Sort feature arrays by (1) patient ID and (2) time
    feature_sort_order = np.lexsort((feature_times_dt, feature_patient_ids))
    feature_patient_ids_sorted = feature_patient_ids[feature_sort_order]
    feature_times_dt_sorted = feature_times_dt[feature_sort_order]
    
    for split in ['train', 'val', 'test']:
        label_patient_ids = label_patient_ids_by_split[split]
        label_values = label_values_by_split[split]
        label_times_dt = label_times_by_split[split]
        
        if len(label_patient_ids) == 0:
            logger.warning(f"No labels found for {split} split, skipping")
            X_by_split[split] = np.array([]).reshape(0, feature_matrix.shape[1])
            y_by_split[split] = np.array([], dtype=bool)
            out_patient_ids_by_split[split] = np.array([], dtype=np.int64)
            out_label_times_by_split[split] = np.array([], dtype='datetime64[us]')
            continue

        # Ensure 1D arrays with same length for lexsort
        label_patient_ids = np.asarray(label_patient_ids).ravel()
        label_values = np.asarray(label_values).ravel()
        label_times_dt = np.asarray(label_times_dt).ravel()
        if hasattr(label_times_dt, 'astype') and str(label_times_dt.dtype) != 'datetime64[us]':
            label_times_dt = pd.to_datetime(label_times_dt).values.astype("datetime64[us]")
        n_labels = len(label_patient_ids)
        if len(label_values) != n_labels or len(label_times_dt) != n_labels:
            logger.warning(f"Shape mismatch for {split}: ids={len(label_patient_ids)}, vals={len(label_values)}, times={len(label_times_dt)}, skipping")
            X_by_split[split] = np.array([]).reshape(0, feature_matrix.shape[1])
            y_by_split[split] = np.array([], dtype=bool)
            out_patient_ids_by_split[split] = np.array([], dtype=np.int64)
            out_label_times_by_split[split] = np.array([], dtype='datetime64[us]')
            continue

        # Sort label arrays by (1) patient ID and (2) time
        label_sort_order = np.lexsort((label_times_dt, label_patient_ids))
        label_patient_ids_sorted = label_patient_ids[label_sort_order]
        label_values_sorted = label_values[label_sort_order]
        label_times_dt_sorted = label_times_dt[label_sort_order]
        
        # Align: exact (patient_id, label_time) when available, else closest same-patient feature.
        feature_indices, label_indices = _align_features_labels_exact_then_closest(
            label_patient_ids_sorted,
            label_times_dt_sorted.astype(np.int64),
            feature_patient_ids_sorted,
            feature_times_dt_sorted.astype(np.int64),
        )
        
        # Get aligned features and filtered labels (only those with exact match)
        feature_matrix_aligned = feature_matrix[feature_sort_order[feature_indices]]
        label_values_matched = label_values_sorted[label_indices]
        label_pids_matched = label_patient_ids_sorted[label_indices]
        label_times_matched = label_times_dt_sorted[label_indices]
        
        X_by_split[split] = feature_matrix_aligned
        y_by_split[split] = label_values_matched
        out_patient_ids_by_split[split] = label_pids_matched
        out_label_times_by_split[split] = label_times_matched
        
        logger.info(f"Matched {split} split:")
        logger.info(f"  X={X_by_split[split].shape}, y={y_by_split[split].shape}, "
                   f"prevalence={np.mean(y_by_split[split]):.4f}")
    
    return X_by_split, y_by_split, out_patient_ids_by_split, out_label_times_by_split


def _norm_time(t) -> str:
    """Normalize prediction_time for matching (same as eval_embeddings.py)."""
    if t is None or t == "":
        return ""
    if hasattr(t, "isoformat"):
        return t.isoformat()
    s = str(t).strip()
    return s.replace(" ", "T") if " " in s and "T" not in s else s


def _load_cohort_keys(cohort_path: str) -> set:
    """Load cohort.json and return set of (patient_id, prediction_time) tuples (same as eval_embeddings.py)."""
    with open(cohort_path) as f:
        cohort = json.load(f)
    return {
        (r["patient_id"], _norm_time(r.get("prediction_time", r.get("label_time", ""))))
        for r in cohort
    }


def tune_logistic_regression(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    n_jobs: int = 1
) -> Tuple[LogisticRegression, MaxAbsScaler, float]:
    """Select C by fitting on train and maximizing validation AUROC on val."""
    logger.info("Tuning logistic regression (val AUROC)...")
    scaler = MaxAbsScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    best_C = None
    best_val_auc = -np.inf
    for C in LR_PARAMS["C"]:
        model = LogisticRegression(
            C=C, n_jobs=1, penalty="l2", tol=0.0001, solver="lbfgs",
            max_iter=10000, random_state=0
        )
        model.fit(X_train_scaled, y_train)
        val_auc = float(roc_auc_score(y_val, model.predict_proba(X_val_scaled)[:, 1]))
        logger.info(f"C={C}: val AUROC={val_auc:.4f}")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_C = C
    logger.info(f"Best C={best_C}, val AUROC: {best_val_auc:.4f}")

    best_model = LogisticRegression(
        C=best_C, n_jobs=1, penalty="l2", tol=0.0001, solver="lbfgs",
        max_iter=10000, random_state=0
    )
    best_model.fit(X_train_scaled, y_train)
    return best_model, scaler, best_val_auc


def evaluate_model(
    model: LogisticRegression,
    scaler: MaxAbsScaler,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    test_patient_ids: np.ndarray
) -> Tuple[Dict[str, Dict[str, float]], np.ndarray]:
    """Evaluate model and compute metrics with confidence intervals.
    
    Returns:
        scores: Dictionary of metric scores
        test_proba: Probability scores for test set
    """
    logger.info("Evaluating model...")
    
    # Scale features
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Get predictions
    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
    y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Compute metrics
    metrics = ['auroc', 'auprc', 'brier']
    metric_funcs = {
        'auroc': roc_auc_score,
        'auprc': average_precision_score,
        'brier': brier_score_loss,
    }
    
    scores = {}
    for metric in metrics:
        func = metric_funcs[metric]
        train_score = func(y_train, y_train_proba)
        val_score = func(y_val, y_val_proba)
        test_score = func(y_test, y_test_proba)
        
        logger.info(f"Train {metric}: {train_score:.4f}")
        logger.info(f"Val {metric}: {val_score:.4f}")
        logger.info(f"Test {metric}: {test_score:.4f}")
        
        # Bootstrap confidence intervals for test set
        test_set = sorted(list(set(test_patient_ids)))
        score_list = []
        for i in range(1000):  # 1k bootstrap replicates
            sample = sklearn.utils.resample(test_set, random_state=i)
            counts = collections.Counter(sample)
            weights = np.zeros_like(test_patient_ids)
            for j, p in enumerate(test_patient_ids):
                weights[j] = counts[p]
            score_val = func(y_test, y_test_proba, sample_weight=weights)
            score_list.append(score_val)
        
        # 95% CI
        lower = np.percentile(score_list, 2.5)
        upper = np.percentile(score_list, 97.5)
        std = np.std(score_list, ddof=1)
        mean = np.mean(score_list)
        
        scores[metric] = {
            'score': test_score,
            'std': std,
            'lower': lower,
            'mean': mean,
            'upper': upper,
        }
    
    return scores, y_test_proba


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CLMBR-T features for all tasks using logistic regression"
    )
    parser.add_argument(
        "--task_name",
        required=True,
        type=str,
        choices=ALL_TASK_NAMES,
        help=f"Task name: one of {ALL_TASK_NAMES}"
    )
    parser.add_argument(
        "--path_to_clmbr_features",
        required=True,
        type=str,
        help="Path to CLMBR-T features pickle file (e.g., clmbr_t_features.pkl)"
    )
    parser.add_argument(
        "--path_to_data_dir",
        required=True,
        type=str,
        help="Path to data directory (should contain data/sft/naivetext/{split}/{task_name}.json)"
    )
    parser.add_argument(
        "--path_to_output_dir",
        required=True,
        type=str,
        help="Path to directory where results will be saved"
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=None,
        help="If set, filter training to n=40 cohort. Requires --cohort_dir."
    )
    parser.add_argument(
        "--cohort_dir",
        type=str,
        default=None,
        help="Directory with {task}/cohort.json files (e.g. data/rubric). Required when --n_train is set."
    )
    parser.add_argument(
        "--cohort_14d_dir",
        type=str,
        default=None,
        help="If set, filter all splits to 14d subcohort: expects {cohort_14d_dir}/{task}/train.json, val.json, test.json."
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default=None,
        help="Load labels from embeddings npz instead of JSON. Path to dir with {task}/{split}.npz "
             "(e.g. data/embeddings/naivetext or data/embeddings_0.6b/naivetext). Uses (patient_id, "
             "prediction_time) for matching to CLMBR labeling_time."
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads for hyperparameter tuning (default: 1)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Convert relative paths to absolute
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if not os.path.isabs(args.path_to_clmbr_features):
        args.path_to_clmbr_features = os.path.join(project_root, args.path_to_clmbr_features)
    
    if not os.path.isabs(args.path_to_data_dir):
        args.path_to_data_dir = os.path.join(project_root, args.path_to_data_dir)
    
    if not os.path.isabs(args.path_to_output_dir):
        args.path_to_output_dir = os.path.join(project_root, args.path_to_output_dir)
    
    if args.cohort_dir and not os.path.isabs(args.cohort_dir):
        args.cohort_dir = os.path.join(project_root, args.cohort_dir)
    if args.cohort_14d_dir and not os.path.isabs(args.cohort_14d_dir):
        args.cohort_14d_dir = os.path.join(project_root, args.cohort_14d_dir)
    
    if args.n_train is not None and args.cohort_dir is None:
        raise ValueError("--cohort_dir required when --n_train is set (e.g. --n_train 40 --cohort_dir data/rubric)")

    if args.embeddings_dir and not os.path.isabs(args.embeddings_dir):
        args.embeddings_dir = os.path.join(project_root, args.embeddings_dir)
    
    logger.info(f"CLMBR-T Evaluation: {args.task_name}")
    logger.info(f"CLMBR-T features path: {args.path_to_clmbr_features}")
    logger.info(f"Data directory: {args.path_to_data_dir}")
    logger.info(f"Output directory: {args.path_to_output_dir}")
    logger.info(f"Number of training examples: {args.n_train if args.n_train else 'all'}")
    if args.n_train:
        logger.info(f"Cohort directory: {args.cohort_dir}")
    if args.cohort_14d_dir:
        logger.info(f"14d cohort directory (filter all splits): {args.cohort_14d_dir}")
    
    # Load CLMBR-T features
    feature_matrix, feature_patient_ids, feature_times = load_clmbr_features(args.path_to_clmbr_features)
    
    # Load task labels: from embeddings npz (if --embeddings_dir) or JSON
    if args.embeddings_dir:
        logger.info(f"Loading labels from embeddings: {args.embeddings_dir}")
        label_patient_ids_by_split, label_values_by_split, label_times_by_split = \
            load_task_labels_from_embeddings(args.task_name, args.embeddings_dir)
    else:
        label_patient_ids_by_split, label_values_by_split, label_times_by_split = \
            load_task_labels_from_json(args.task_name, args.path_to_data_dir)

    # Match features with labels and split
    X_by_split, y_by_split, patient_ids_by_split, label_times_by_split = \
        match_features_with_labels_by_split(
            feature_matrix,
            feature_patient_ids,
            feature_times,
            label_patient_ids_by_split,
            label_values_by_split,
            label_times_by_split,
        )
    
    # Print (task, split): n=samples pos=positives for each
    def _print_split_summary(task: str, X_by_split: dict, y_by_split: dict):
        for split in ('train', 'val', 'test'):
            n = len(y_by_split[split])
            pos = int(np.sum(y_by_split[split]))
            print(f"[{task}] {split}: n={n} pos={pos}", flush=True)

    _print_split_summary(args.task_name, X_by_split, y_by_split)

    # Filter all splits to 14d subcohort when --cohort_14d_dir is set
    if args.cohort_14d_dir is not None:
        for split in ['train', 'val', 'test']:
            cohort_path = os.path.join(args.cohort_14d_dir, args.task_name, f"{split}.json")
            if not os.path.exists(cohort_path):
                raise FileNotFoundError(f"14d cohort file not found: {cohort_path}")
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
        logger.info("Filtered all splits to 14d cohort")
        _print_split_summary(args.task_name + " (14d)", X_by_split, y_by_split)

    # Check if we have train/val/test data
    if len(y_by_split['train']) == 0:
        raise ValueError(f"No training data found for task {args.task_name}")
    if len(y_by_split['val']) == 0:
        logger.warning(f"No validation data found for task {args.task_name}, using train for validation")
        X_by_split['val'] = X_by_split['train']
        y_by_split['val'] = y_by_split['train']
    if len(y_by_split['test']) == 0:
        raise ValueError(f"No test data found for task {args.task_name}")
    
    # Filter training data to cohort when n_train is specified (e.g. n=40 from data/rubric/{task}/cohort.json)
    if args.n_train is not None:
        cohort_path = os.path.join(args.cohort_dir, args.task_name, "cohort.json")
        if not os.path.exists(cohort_path):
            raise FileNotFoundError(f"Cohort file not found: {cohort_path}")
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
        logger.info(f"Filtered train to {len(y_by_split['train'])} examples (cohort keys={len(cohort_keys)})")
        _print_split_summary(args.task_name + " (after cohort)", X_by_split, y_by_split)

    # Train logistic regression (best C on val, fit on train, eval on test)
    model, scaler, best_val_auc = tune_logistic_regression(
        X_by_split['train'], 
        X_by_split['val'], 
        y_by_split['train'], 
        y_by_split['val'],
        n_jobs=args.num_threads
    )
    
    # Evaluate
    scores, test_proba = evaluate_model(
        model, scaler,
        X_by_split['train'],
        X_by_split['val'],
        X_by_split['test'],
        y_by_split['train'],
        y_by_split['val'],
        y_by_split['test'],
        patient_ids_by_split['test']
    )
    
    # Save results in same format as 05_eval/eval_embeddings.py: {output_dir}/{task}/metrics.json, predictions.csv
    task_out_dir = os.path.join(args.path_to_output_dir, args.task_name)
    os.makedirs(task_out_dir, exist_ok=True)
    
    # best_c from model selected by val NLL
    best_c = model.get_params().get('C', 1.0)
    
    metrics = {
        "task": args.task_name,
        "best_c": best_c,
        "best_l1_ratio": 0.0,  # L2 only, for compatibility with eval_embeddings format
        "val_auroc": best_val_auc,
        "test_auroc": scores["auroc"]["score"],
        "test_auroc_ci": [scores["auroc"]["lower"], scores["auroc"]["upper"]],
        "test_auprc": scores["auprc"]["score"],
        "test_auprc_ci": [scores["auprc"]["lower"], scores["auprc"]["upper"]],
        "n_train": int(len(y_by_split['train'])),
        "n_test": int(len(y_by_split['test'])),
    }
    
    metrics_file = os.path.join(task_out_dir, "metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to: {metrics_file}")
    
    # predictions.csv (same schema as eval_embeddings)
    test_patient_ids = patient_ids_by_split['test']
    test_label_times = label_times_by_split['test']
    test_ground_truth = y_by_split['test']
    
    predictions_file = os.path.join(task_out_dir, "predictions.csv")
    with open(predictions_file, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            "patient_id", "label_time", "ground_truth",
            "probability_score", "target_task"])
        w.writeheader()
        for i, (pid, gt, ps) in enumerate(zip(test_patient_ids, test_ground_truth, test_proba)):
            label_time = _norm_time(pd.Timestamp(test_label_times[i])) if i < len(test_label_times) else ""
            w.writerow({
                "patient_id": int(pid),
                "label_time": label_time,
                "ground_truth": int(gt),
                "probability_score": float(ps),
                "target_task": args.task_name
            })
    logger.info(f"Predictions saved to: {predictions_file}")
    
    logger.success("Done!")


if __name__ == "__main__":
    main()
