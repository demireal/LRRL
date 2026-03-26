#!/usr/bin/env python3
"""
Adaptad from https://github.com/stefanhgm/ehrshot-benchmark


Serialize patient EHRs into text for all 15 clinical prediction tasks.

Inputs:
  --path_to_database   : Path to the FEMR patient database.
  --path_to_labels_dir : Directory containing per-task subdirectories,
                         each with a labeled_patients.csv file.
                         (e.g. EHRSHOT_ASSETS/benchmark/{task}/labeled_patients.csv)
  --path_to_splits     : CSV mapping patient IDs to train/val/test splits.
  --output_dir         : Where to write per-task per-split JSON files.
  --mode               : "naivetext" (no vitals sections).
  --skip_existing      : Skip tasks that already have train/val/test JSONs.

Outputs:
  {output_dir}/{task}/{split}.json   -- one file per task per split.

Each JSON file is a list of dicts:
  { patient_id, prediction_time, task, split, label, serialization,
    original_tokens, was_clipped }

Cohort balancing (applied per task per split, BEFORE serialization):
  val  : min(50, smallest-class-count) per class  (balanced pos/neg)
  train/test : if total > 3000 =>
      positives < 1000 : match negatives to positives  (balanced)
      positives >= 1000: 1000 each = 2000 total        (balanced)
  Otherwise: keep all.
  Only the capped records are serialized to avoid redundant work.

Token clipping:
  Serializations are clipped to 8192 tokens using the
  Qwen/Qwen3-Embedding-8B tokenizer.

Connects to: 02_create_sft (consumes the JSON files produced here).
"""

import argparse
import os
import sys
import json
import csv
import random
import collections
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

from loguru import logger
from transformers import AutoTokenizer
from femr.extension import datasets as extension_datasets
from femr import Patient, Event
from femr.featurizers.featurizers import get_patient_birthdate

# Resolve imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.tasks import TASKS, ALL_TASK_NAMES, TOKENIZER_NAME, MAX_SERIALIZATION_TOKENS, SEED
from ehr_serializer import (
    EHRSerializer,
    AGGREGATED_EVENTS_CODES_LOINC,
)

PatientDatabase = extension_datasets.PatientDatabase
Ontology = extension_datasets.Ontology

RANDOM_SEED = SEED
AGE_IDENTIFIER = "Patient age"

# Excluded ontologies (same as the "no_labs" preset in the original code)
EXCLUDED_ONTOLOGIES = ["LOINC", "Domain", "CARE_SITE", "ICDO3"]

# ---------------------------------------------------------------------------
# Cohort balancing
# ---------------------------------------------------------------------------

def balance_split(records: List[dict], split: str) -> List[dict]:
    """Return a (possibly subsampled) balanced list of records for *split*."""
    rng = random.Random(RANDOM_SEED)
    pos = [r for r in records if r["label"] is True]
    neg = [r for r in records if r["label"] is False]

    if split == "val":
        n = min(50, len(pos), len(neg))
        pos = rng.sample(pos, n)
        neg = rng.sample(neg, n)
    elif split in ("train", "test"):
        total = len(pos) + len(neg)
        if total > 3000:
            if len(pos) < 1000:
                # match negatives to positives
                n = len(pos)
            else:
                n = 1000
            pos = rng.sample(pos, min(n, len(pos)))
            neg = rng.sample(neg, min(n, len(neg)))
    # else: keep all

    result = pos + neg
    rng.shuffle(result)
    return result


# ---------------------------------------------------------------------------
# Token clipping
# ---------------------------------------------------------------------------

def clip_text(text: str, tokenizer, max_tokens: int) -> Tuple[str, int, bool]:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    n = len(tokens)
    if n <= max_tokens:
        return text, n, False
    clipped = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
    return clipped, n, True


# ---------------------------------------------------------------------------
# Patient loading
# ---------------------------------------------------------------------------

def load_splits(path: str) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            mapping[int(row["omop_person_id"])] = row["split"]
    return mapping


def _parse_label_value(task: str, raw_value: str, label_type: str) -> bool:
    """Convert raw value string to bool based on task and label_type.

    Task-specific rules are applied first so lab/chexpert are never misinterpreted.

    - lab_*: value is 0/1/2/3 (categorical). Only 1 = abnormal (True); 0=normal, 2/3=other (False).
    - chexpert: value is int bitmask. 8192 = no finding (False); any other value = at least one finding (True).
      Matches ehrshot/9_make_cohort_plots.py neg_values = ['8192'].
    - guo_*, new_*: value is "True"/"False" (boolean) or label_type "boolean".
    """
    if task.startswith("lab_"):
        return int(raw_value) == 1
    if task == "chexpert":
        return int(raw_value) != 8192
    if label_type == "boolean" or raw_value in ("True", "False"):
        return raw_value == "True"
    return raw_value == "True"


def load_labels(labels_dir: str, task_names: set):
    """Load labels from per-task labeled_patients.csv files.

    Reads {labels_dir}/{task}/labeled_patients.csv for each task in *task_names*.
    Each CSV has columns: patient_id, prediction_time, value, label_type.
    One record per CSV row (no dedup); label is taken from that row.
    Supports boolean (guo/new), categorical 0/1/2/3 with 1=positive (lab_), and bitmask (chexpert) formats.

    Returns:
        patients  -- Dict[patient_id, List[(prediction_time, task, label)]]
    """
    patients: Dict[int, List[Tuple[datetime, str, bool]]] = collections.defaultdict(list)
    for task in sorted(task_names):
        label_file = os.path.join(labels_dir, task, "labeled_patients.csv")
        if not os.path.exists(label_file):
            logger.warning(f"Labels file not found, skipping task: {label_file}")
            continue
        count = 0
        with open(label_file) as f:
            for row in csv.DictReader(f):
                pid = int(row["patient_id"])
                pt_str = row["prediction_time"].replace(" ", "T")
                t = datetime.fromisoformat(pt_str)
                if t.second != 0:
                    t = t.replace(second=0)
                val = _parse_label_value(task, row["value"], row.get("label_type", ""))
                patients[pid].append((t, task, val))
                count += 1
        logger.info(f"  Loaded {count} labels for task={task}")
    return patients


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def serialize_patient(
    database: PatientDatabase,
    ontology: Ontology,
    patient_id: int,
    label_time: datetime,
    num_aggregated: int,
) -> str:
    patient: Patient = database[patient_id]

    def is_visit(event: Event) -> bool:
        return event.code.startswith("Visit/")

    def resolve_code(code: str, included_ontologies: List[str] = []) -> Optional[str]:
        ont = code.split("/")[0].strip()
        if (ont in EXCLUDED_ONTOLOGIES
                and code not in AGGREGATED_EVENTS_CODES_LOINC
                and ont not in included_ontologies):
            return None
        if code.startswith(f"{AGE_IDENTIFIER}: "):
            return code
        if ont == "Cancer Modifier" and "OMOP" not in code:
            return code.split("/", 1)[1].replace("_", " ").replace("-", " ").replace("/", " ").strip()
        desc = ontology.get_text_description(code)
        if desc.startswith("Birth"):
            return None
        return desc.strip()

    events = [e for e in patient.events if e.start <= label_time]

    # Replace birth event with age
    birth = get_patient_birthdate(patient)
    age = int((label_time - birth).days / 365)
    if events:
        b = events[0]
        events[0] = Event(b.start, f"{AGE_IDENTIFIER}: {age}", b.value)

    serializer = EHRSerializer()
    serializer.load_from_femr_events(events, resolve_code, is_visit,
                                     filter_aggregated_events=(num_aggregated > 0))
    return serializer.serialize(num_aggregated, label_time=label_time)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path_to_database", required=True)
    p.add_argument("--path_to_labels_dir", required=True)
    p.add_argument("--path_to_splits", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mode", required=True, choices=["naivetext"])
    p.add_argument("--tasks", default="", help="Comma-separated task subset (default: all)")
    p.add_argument("--max_tokens", type=int, default=MAX_SERIALIZATION_TOKENS)
    p.add_argument("--no_clip", action="store_true",
                   help="Do not clip serializations to max_tokens; use full text as-is")
    p.add_argument("--no_balance", action="store_true",
                   help="Skip cohort balancing; serialize all available records.")
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip tasks that already have train.json, val.json, test.json")
    return p.parse_args()


def main():
    args = parse_args()
    num_aggregated = 0 if args.mode == "naivetext" else 3

    if os.path.exists(args.output_dir) and not args.force and not args.skip_existing:
        raise FileExistsError(f"{args.output_dir} exists. Use --force to overwrite.")
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which tasks to process
    task_set = set(args.tasks.split(",")) if args.tasks else set(ALL_TASK_NAMES)
    if args.skip_existing:
        skipped = []
        for task in list(task_set):
            task_dir = os.path.join(args.output_dir, task)
            if all(os.path.exists(os.path.join(task_dir, f"{s}.json")) for s in ("train", "val", "test")):
                task_set.discard(task)
                skipped.append(task)
        if skipped:
            logger.info(f"Skipping {len(skipped)} tasks (already have JSONs): {skipped}")
    if not task_set:
        logger.success("All tasks already serialized. Nothing to do.")
        return
    logger.info(f"Mode={args.mode}  num_aggregated={num_aggregated}  no_clip={args.no_clip}  no_balance={args.no_balance}  tasks={sorted(task_set)}")
    logger.info("Label rules: lab_* value==1 only; chexpert value!=8192; guo/new boolean")

    # Load tokenizer (skip when no_clip to save time and memory)
    tokenizer = None
    if not args.no_clip:
        logger.info(f"Loading tokenizer: {TOKENIZER_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)

    # Load splits & labels (from per-task labeled_patients.csv files)
    patient_to_split = load_splits(args.path_to_splits)
    logger.info(f"Loading labels from {args.path_to_labels_dir} ...")
    patients_to_labels = load_labels(args.path_to_labels_dir, task_set)
    logger.info(f"Loaded {len(patient_to_split)} splits, "
                f"{len(patients_to_labels)} patients with labels")

    # Build minimal records (patient_id, prediction_time, task, split, label) - no serialization yet
    collection: Dict[str, Dict[str, List[dict]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for pid, labels in patients_to_labels.items():
        split = patient_to_split.get(pid)
        if split is None:
            continue
        for label_time, task, label_val in labels:
            collection[task][split].append({
                "patient_id": pid,
                "prediction_time": label_time,
                "task": task,
                "split": split,
                "label": label_val,
            })

    # Cap/balance BEFORE serialization
    for task in sorted(collection):
        for split in ("train", "val", "test"):
            records = collection[task].get(split, [])
            if records:
                before = len(records)
                n_pos = sum(1 for r in records if r["label"])
                if args.no_balance:
                    logger.info(f"  {task}/{split}: {before} records kept (no balancing)  [pos={n_pos}]")
                else:
                    collection[task][split] = balance_split(records, split)
                    logger.info(f"  {task}/{split}: {before} -> {len(collection[task][split])} "
                                f"(capped before serialization)  [pos={n_pos}]")

    # Load database (only after we know what to serialize)
    logger.info(f"Loading FEMR database: {args.path_to_database}")
    database = PatientDatabase(args.path_to_database)
    ontology = database.get_ontology()

    # Serialize only the capped records
    to_serialize = sum(
        len(records)
        for task_records in collection.values()
        for records in task_records.values()
    )
    logger.info(f"Serializing {to_serialize} records (after capping)")

    done = 0
    clipped_n = 0
    for task in sorted(collection):
        task_dir = os.path.join(args.output_dir, task)
        os.makedirs(task_dir, exist_ok=True)
        for split in ("train", "val", "test"):
            records = collection[task].get(split, [])
            if not records:
                continue
            for r in records:
                label_time = r["prediction_time"]
                text = serialize_patient(
                    database, ontology, r["patient_id"], label_time, num_aggregated
                )
                if args.no_clip:
                    # Skip slow tokenizer.encode on long texts; use -1 to indicate not computed
                    orig_tokens = -1
                    clipped_text, was_clipped = text, False
                else:
                    clipped_text, orig_tokens, was_clipped = clip_text(
                        text, tokenizer, args.max_tokens
                    )  # tokenizer is not None when not no_clip
                if was_clipped:
                    clipped_n += 1
                r["prediction_time"] = label_time.isoformat()
                r["serialization"] = clipped_text
                r["original_tokens"] = orig_tokens
                r["was_clipped"] = was_clipped
                done += 1
                if done % 500 == 0:
                    logger.info(f"  {done}/{to_serialize} serialized (clipped so far: {clipped_n})")

            out_path = os.path.join(task_dir, f"{split}.json")
            with open(out_path, "w") as f:
                json.dump(records, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            pos = sum(1 for r in records if r["label"])
            neg = len(records) - pos
            logger.info(f"  {task}/{split}: {len(records)} saved (pos={pos}, neg={neg}) => {out_path}")

        logger.info(f"  Task {task} complete - outputs in {task_dir} (ready for inspection)")

    logger.success(f"Done. Output in {args.output_dir}")


if __name__ == "__main__":
    main()
