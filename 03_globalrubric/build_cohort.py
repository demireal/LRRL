#!/usr/bin/env python3
"""
Build diverse 40-patient cohorts per task via label-stratified k-means.

For each task, this script:
  1. Loads training-split serialized records (from 01_serialize naivetext).
  2. Loads precomputed train embeddings from generate_embeddings.py output.
  3. Runs k-means (k=20) separately on positives and negatives.
  4. Selects the medoid of each cluster -> 20 pos + 20 neg = 40 patients.
  5. Saves the cohort JSON (40 records identified by patient_id + prediction_time).

Requires precomputed embeddings from 05_eval/generate_embeddings.py. Run
generate_embeddings first.
Raises an error if embeddings are missing.

Inputs:
  --input_dir     : naivetext serialized directory (data/serialized/naivetext).
  --output_dir    : where to write cohort files.
  --embeddings_dir: directory with {task}/train.npz (default: derived from input_dir
                    as {parent}/embeddings/naivetext when input_dir ends with serialized/naivetext)
  --tasks         : space-separated list of tasks (default: all 15).

Outputs:
  {output_dir}/{task}/cohort.json     -- full cohort records (40 items, identified by patient_id + prediction_time)

Connects to:
  - Upstream : 01_serialize (naivetext), 05_eval/generate_embeddings.py (embeddings)
  - Downstream : create_rubric.py, apply_rubric.py, 05_eval (n=40)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
from loguru import logger
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.tasks import TASKS, ALL_TASK_NAMES, SEED


# ---------------------------------------------------------------------------
# K-means + medoid selection
# ---------------------------------------------------------------------------

def _select_medoids(X: np.ndarray, km: KMeans,
                    indices: np.ndarray) -> List[int]:
    selected = []
    for cid in range(km.n_clusters):
        mask = km.labels_ == cid
        cidx = np.where(mask)[0]
        if len(cidx) == 0:
            continue
        dists = np.linalg.norm(X[cidx] - km.cluster_centers_[cid], axis=1)
        selected.append(int(indices[cidx[np.argmin(dists)]]))
    return selected


def build_cohort_for_task(
    records: List[dict],
    embeddings: np.ndarray,
    n_per_class: int = 20,
) -> List[dict]:
    pos = [r for r in records if r["label"] is True]
    neg = [r for r in records if r["label"] is False]
    logger.info(f"  train: {len(pos)} pos, {len(neg)} neg")

    pos_mask = np.array([r["label"] is True for r in records])
    selected_records: List[dict] = []

    for subset, mask_val, label in [(pos, True, "pos"), (neg, False, "neg")]:
        n = min(n_per_class, len(subset))
        if n == 0:
            continue
        mask = pos_mask == mask_val
        sub_emb = embeddings[mask]
        sub_idx = np.arange(len(records))[mask]
        km = KMeans(n_clusters=n, random_state=SEED, n_init=10)
        km.fit(sub_emb)
        medoid_idx = _select_medoids(sub_emb, km, sub_idx)
        for i in medoid_idx:
            selected_records.append(records[i])
        logger.info(f"  selected {len(medoid_idx)} {label} medoids")

    return selected_records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_precomputed_embeddings(
    embeddings_path: Path,
    records: List[dict],
) -> np.ndarray:
    """Load embeddings from npz and align to records by (patient_id, prediction_time).
    Raises FileNotFoundError if path does not exist.
    Raises ValueError if loading fails or records cannot be matched.
    """
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {embeddings_path}. "
            "Run 05_eval/generate_embeddings.py first."
        )
    try:
        data = np.load(embeddings_path, allow_pickle=True)
        emb = data["embeddings"]
        pids = data["patient_ids"]
        if "prediction_times" in data:
            times = data["prediction_times"]
        else:
            times = np.array([""] * len(pids), dtype=object)
        key_to_emb = {}
        for i in range(len(pids)):
            pt = str(times[i]) if i < len(times) else ""
            key_to_emb[(int(pids[i]), pt)] = emb[i]
        out = []
        missing = []
        for r in records:
            pid = r["patient_id"]
            pt = r.get("prediction_time", r.get("label_time", ""))
            if not isinstance(pt, str):
                pt = pt.isoformat() if hasattr(pt, "isoformat") else str(pt)
            k = (pid, pt)
            if k not in key_to_emb:
                missing.append(k)
                continue
            out.append(key_to_emb[k])
        if missing:
            raise ValueError(
                f"Cannot match {len(missing)} records to embeddings in {embeddings_path}. "
                f"First missing key: {missing[0]}. Ensure embeddings were generated from the same "
                "serialized data (01_serialize -> 02_create_sft -> generate_embeddings)."
            )
        return np.array(out)
    except (FileNotFoundError, ValueError):
        raise
    except Exception as e:
        raise ValueError(f"Could not load embeddings from {embeddings_path}: {e}") from e


def _default_embeddings_dir(input_dir: str) -> str:
    """Derive embeddings dir from input_dir: .../serialized/naivetext -> .../embeddings/naivetext."""
    p = Path(input_dir).resolve()
    parts = list(p.parts)
    try:
        idx = parts.index("serialized")
        parts[idx] = "embeddings"
        return str(Path(*parts))
    except ValueError:
        return str(p.parent / "embeddings" / "naivetext")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_dir", required=True,
                   help="Naivetext serialized dir (data/serialized/naivetext)")
    p.add_argument("--output_dir", required=True,
                   help="Where to write cohort JSONs")
    p.add_argument("--embeddings_dir", default=None,
                   help="Dir with {task}/train.npz from generate_embeddings.py (default: derived from input_dir)")
    p.add_argument("--tasks", nargs="+", default=ALL_TASK_NAMES)
    p.add_argument("--n_per_class", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    if args.embeddings_dir is None:
        args.embeddings_dir = _default_embeddings_dir(args.input_dir)
    embeddings_root = Path(args.embeddings_dir)

    np.random.seed(SEED)

    for task in args.tasks:
        logger.info(f"\n{'='*60}\nBuilding cohort for: {task}\n{'='*60}")
        train_path = Path(args.input_dir) / task / "train.json"
        if not train_path.exists():
            logger.warning(f"  no train.json at {train_path}, skipping")
            continue
        with open(train_path) as f:
            records = json.load(f)

        npz_path = embeddings_root / task / "train.npz"
        embeddings = _load_precomputed_embeddings(npz_path, records)
        logger.info(f"  loaded embeddings from {npz_path} (shape={embeddings.shape})")

        cohort = build_cohort_for_task(
            records, embeddings, args.n_per_class,
        )

        out_dir = Path(args.output_dir) / task
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "cohort.json", "w") as f:
            json.dump(cohort, f, indent=2)
        logger.info(f"  saved {len(cohort)} records -> {out_dir}")

    logger.success("All cohorts built.")


if __name__ == "__main__":
    main()
