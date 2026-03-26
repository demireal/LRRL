#!/usr/bin/env python3
"""
Generate count-GBM features for all tasks (acute MI, hyperlipidemia, hypertension, pancreatic cancer).

This script:
1. Loads labels for all 4 tasks
2. Computes count features using AgeFeaturizer and CountFeaturizer
3. Combines features from all tasks into a single pickle file
4. Saves features in the format expected by evaluate_multi_task_count.py

The output pickle file will contain a dict with:
- 'data_matrix': Feature matrix (n_samples, n_features)
- 'patient_ids': Patient IDs (n_samples,)
- 'labeling_time': Labeling times (n_samples,)
"""

import os
import sys
import json
import argparse
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from loguru import logger
import scipy.sparse
from scipy.sparse import issparse

# Add project root and parent (ehrshot) to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parent_root = os.path.dirname(project_root)
for p in (project_root, parent_root):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(project_root)

# Import from config and femr
from config.tasks import ALL_TASK_NAMES
from femr.labelers import load_labeled_patients, LabeledPatients
from femr.featurizers import AgeFeaturizer, CountFeaturizer, FeaturizerList
from ehrshot.utils import check_file_existence_and_handle_force_refresh

# Task name mappings (internal task names to dataset names)
TASK_MAPPINGS = {
    'acute_mi': 'new_acutemi',
    'hyperlipidemia': 'new_hyperlipidemia',
    'hypertension': 'new_hypertension',
    'pancreatic_cancer': 'new_pancan'
}

# Default list of tasks (legacy labeled_patients dir)
DEFAULT_TASKS = ['acute_mi', 'hyperlipidemia', 'hypertension', 'pancreatic_cancer']


def load_labels_from_naivetext_splits(
    path_to_data_dir: str,
    sft_subdir: str = "data/sft/naivetext",
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load labels from data/sft/naivetext/{split}/{task}.json (same as CLMBR).

    Returns dict with single key 'naivetext' -> (patient_ids, label_values, label_times).
    """
    from pathlib import Path
    naivetext_root = Path(path_to_data_dir) / sft_subdir
    patient_ids: list = []
    label_values: list = []
    label_times: list = []

    for split in ["train", "val", "test"]:
        split_dir = naivetext_root / split
        if not split_dir.exists():
            continue
        for task in ALL_TASK_NAMES:
            path = split_dir / f"{task}.json"
            if not path.exists():
                continue
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
        "naivetext": (
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
    
    Format matches what FEMR featurizers expect:
    - Columns: patient_id, prediction_time, value, label_type
    
    Returns:
        Path to the created CSV file
    """
    logger.info(f"Creating combined labeled_patients.csv from all tasks...")
    
    # Combine all labels into a single DataFrame
    # Format: patient_id, prediction_time, value, label_type
    all_rows = []
    for task_name, (patient_ids, label_values, label_times) in all_labels.items():
        for i in range(len(patient_ids)):
            # Convert time to datetime if needed
            time_val = label_times[i]
            if isinstance(time_val, np.datetime64):
                # Convert to datetime and ensure minute-level resolution
                dt = pd.Timestamp(time_val).to_pydatetime()
                dt = dt.replace(second=0, microsecond=0)
                prediction_time = dt.isoformat()
            elif hasattr(time_val, 'isoformat'):
                dt = time_val.replace(second=0, microsecond=0) if hasattr(time_val, 'replace') else time_val
                prediction_time = dt.isoformat()
            else:
                prediction_time = str(time_val)
            
            all_rows.append({
                'patient_id': int(patient_ids[i]),
                'prediction_time': prediction_time,
                'value': bool(label_values[i]),
                'label_type': 'boolean'
            })
    
    df = pd.DataFrame(all_rows, columns=['patient_id', 'prediction_time', 'value', 'label_type'])
    df.to_csv(output_path, index=False)
    logger.info(f"Created combined labeled_patients.csv with {len(df)} labels at: {output_path}")
    return output_path


def compute_count_features(
    path_to_database: str,
    all_labels: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    path_to_features_dir: str,
    num_threads: int = None,
    is_force_refresh: bool = False
) -> Dict[str, np.ndarray]:
    """Compute count features for all tasks using FEMR featurizers.
    
    Returns:
        Dictionary with keys: 'data_matrix', 'patient_ids', 'labeling_time'
    """
    # Create temporary directory for combined labels
    temp_dir = os.path.join(path_to_features_dir, "count_gbm_temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create combined labeled_patients.csv
    combined_labels_path = os.path.join(temp_dir, "all_labels.csv")
    create_combined_labeled_patients_csv(all_labels, combined_labels_path)
    
    # Load consolidated labels
    logger.info(f"Loading combined LabeledPatients from `{combined_labels_path}`")
    labeled_patients: LabeledPatients = load_labeled_patients(combined_labels_path)
    
    # Combine two featurizations: age and count of codes
    age = AgeFeaturizer()
    count = CountFeaturizer(is_ontology_expansion=True)
    featurizer_age_count = FeaturizerList([age, count])
    
    # Preprocessing the featurizers
    logger.info("Start | Preprocess featurizers")
    featurizer_age_count.preprocess_featurizers(path_to_database, labeled_patients, num_threads)
    logger.info("Finish | Preprocess featurizers")
    
    # Run actual featurization
    logger.info("Start | Featurize patients")
    results = featurizer_age_count.featurize(path_to_database, labeled_patients, num_threads)
    
    # Debug: log the type and structure of results
    logger.debug(f"Results type: {type(results)}")
    if isinstance(results, dict):
        logger.debug(f"Results keys: {list(results.keys())}")
        for key, value in results.items():
            if hasattr(value, 'shape'):
                logger.debug(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                logger.debug(f"  {key}: type={type(value)}")
    elif isinstance(results, (tuple, list)):
        logger.debug(f"Results length: {len(results)}")
        for i, item in enumerate(results):
            if hasattr(item, 'shape'):
                logger.debug(f"  results[{i}]: shape={item.shape}, dtype={item.dtype}")
            else:
                logger.debug(f"  results[{i}]: type={type(item)}")
    
    # Handle different return formats
    if isinstance(results, dict):
        # Dict format: {'patient_ids': ..., 'feature_times': ..., 'features': ...}
        # Try different possible key names
        feature_matrix = results.get('features') or results.get('data_matrix') or results.get('feature_matrix')
        patient_ids = results.get('patient_ids')
        label_values = results.get('label_values', np.array([]))
        label_times = results.get('feature_times') or results.get('labeling_time') or results.get('label_times')
        
        if feature_matrix is None:
            raise ValueError(f"Could not find feature matrix in results dict. Available keys: {list(results.keys())}")
    elif isinstance(results, (tuple, list)) and len(results) >= 4:
        # Tuple format: (feature_matrix, patient_ids, label_values, label_times)
        feature_matrix, patient_ids, label_values, label_times = (
            results[0],
            results[1],
            results[2],
            results[3],
        )
    else:
        raise ValueError(f"Unexpected results format: {type(results)}, length: {len(results) if hasattr(results, '__len__') else 'N/A'}")
    
    logger.info("Finish | Featurize patients")
    
    # Handle sparse matrices - featurize may return sparse matrices
    # Check if it's wrapped in an object array (bug from previous version)
    if isinstance(feature_matrix, np.ndarray) and feature_matrix.ndim == 0 and feature_matrix.dtype == object:
        # Extract the actual matrix from the object array
        feature_matrix = feature_matrix.item()
        logger.info("Extracted matrix from object array wrapper")
    
    if issparse(feature_matrix):
        logger.info(f"Feature matrix is sparse: shape={feature_matrix.shape}, format={feature_matrix.format}")
        # Keep as sparse matrix - scipy sparse matrices work with sklearn
        # Don't convert to numpy array - that would wrap it in an object array
    elif isinstance(feature_matrix, np.ndarray):
        # Handle numpy array
        if feature_matrix.ndim == 0:
            raise ValueError(f"Feature matrix is 0-dimensional (scalar). This indicates an error in feature extraction. Results type: {type(results)}, feature_matrix value: {feature_matrix}")
        if feature_matrix.ndim == 1:
            # If 1D, reshape to (n_samples, n_features) where n_features=1
            feature_matrix = feature_matrix.reshape(-1, 1)
    else:
        # Try to convert to numpy array (only if not sparse)
        if not issparse(feature_matrix):
            feature_matrix = np.array(feature_matrix)
            if feature_matrix.ndim == 0:
                raise ValueError(f"Feature matrix is 0-dimensional after conversion. Original type: {type(feature_matrix)}")
    
    # Convert patient_ids and label_times to numpy arrays if needed
    if not isinstance(patient_ids, np.ndarray):
        patient_ids = np.array(patient_ids)
    if not isinstance(label_times, np.ndarray):
        label_times = np.array(label_times)
    
    # Ensure feature_times are datetime-like for consistency
    if not np.issubdtype(label_times.dtype, np.datetime64) if hasattr(label_times, 'dtype') else True:
        try:
            label_times = pd.to_datetime(label_times).values.astype('datetime64[us]')
        except Exception as e:
            logger.warning(f"Could not convert times to datetime64: {e}, keeping original format")
    
    logger.info(f"Generated count features:")
    logger.info(f"  Feature matrix shape: {feature_matrix.shape}")
    logger.info(f"  Patient IDs shape: {patient_ids.shape}")
    logger.info(f"  Times shape: {label_times.shape}")
    
    return {
        'data_matrix': feature_matrix,
        'patient_ids': patient_ids,
        'labeling_time': label_times
    }


def save_features(features: Dict[str, np.ndarray], output_path: str):
    """Save features to pickle file."""
    logger.info(f"Saving features to: {output_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(features, f)
    
    logger.success(f"Saved features to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate count-GBM features for all tasks"
    )
    parser.add_argument(
        "--path_to_database",
        required=True,
        type=str,
        help="Path to FEMR patient database"
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
        help="Path to ehrshot-v2 data root. When set, loads from {sft_subdir}/{split}/{task}.json (same as CLMBR)"
    )
    parser.add_argument(
        "--path_to_features_dir",
        required=True,
        type=str,
        help="Path to directory where features will be saved"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="count_gbm_features.pkl",
        help="Output filename (default: count_gbm_features.pkl)"
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=None,
        help="Number of threads to use for featurization"
    )
    parser.add_argument(
        "--is_force_refresh",
        action='store_true',
        default=False,
        help="If set, then overwrite existing output file"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
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
    
    output_path = os.path.join(args.path_to_features_dir, args.output_filename)
    
    # Force refresh check
    check_file_existence_and_handle_force_refresh(output_path, args.is_force_refresh)
    
    logger.info("Count-GBM Feature Generation")
    logger.info(f"Database: {args.path_to_database}")
    logger.info(f"Features directory: {args.path_to_features_dir}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Num threads: {args.num_threads}")
    
    # Load labels: from naivetext JSON (path_to_data_dir) or legacy labeled_patients dir
    if args.path_to_data_dir:
        logger.info(f"Loading labels from naivetext: {args.path_to_data_dir}")
        all_labels = load_labels_from_naivetext_splits(args.path_to_data_dir)
    elif args.path_to_labels_dir:
        all_labels = load_all_task_labels(args.path_to_labels_dir)
    else:
        raise ValueError("Must provide --path_to_data_dir or --path_to_labels_dir")
    
    if len(all_labels) == 0:
        raise ValueError("No labels loaded! Check that label files exist in the labels directory.")
    
    num_threads = args.num_threads if args.num_threads is not None else 1
    features = compute_count_features(
        args.path_to_database,
        all_labels,
        args.path_to_features_dir,
        num_threads=num_threads,
        is_force_refresh=args.is_force_refresh
    )
    
    # Save features
    save_features(features, output_path)
    
    logger.success("Done!")


if __name__ == "__main__":
    main()
