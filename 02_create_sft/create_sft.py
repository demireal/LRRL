#!/usr/bin/env python3
"""
Create SFT (Supervised Fine-Tuning) datasets from serialized patient records.

Inputs:
  --input_dir  : Directory with serialized JSON files produced by 01_serialize.
                 Expected layout: {input_dir}/{task}/{split}.json
  --output_dir : Where to write per-task per-split SFT JSON files.

Outputs:
  {output_dir}/{split}/{task}.json  -- one file per task per split.

Each output file is a JSON list of conversation dicts:
  {
    "conversations": [
      {"role": "system", "content": "..."},
      {"role": "user",   "content": "..."},
      {"role": "assistant", "content": "Yes" | "No"}
    ],
    "patient_id": ...,
    "label_time": ...,
    "label_value": ...,
    "task": ...,
    "task_query": ...
  }

Prompt format (sandwich style):
  The task query appears before the EHR to prime attention and again after it
  so the model has it fresh in context when generating the answer.

All tasks use Yes / No as the output label.

Connects to:
  - Upstream: 01_serialize (produces the serialized JSONs consumed here).
  - Downstream: 03_globalrubric, 04_localrubric, 05_eval
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.tasks import TASKS, ALL_TASK_NAMES, SYSTEM_MESSAGE


def build_user_prompt(task_query: str, serialization: str) -> str:
    """Build the sandwich-style user prompt."""
    return (
        f"Based on the patient's EHR below, predict: {task_query}\n\n"
        f"--- Patient EHR ---\n{serialization}\n--- End of EHR ---\n\n"
        f"Based on the above EHR, predict: {task_query}\n"
        "Respond with exactly one word: Yes or No."
    )


def make_sft_entry(record: dict, task_query: str) -> dict:
    """Convert a serialized record into an SFT conversation dict."""
    user_content = build_user_prompt(task_query, record["serialization"])
    assistant_content = "Yes" if record["label"] else "No"
    return {
        "conversations": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "patient_id": record["patient_id"],
        "label_time": record["prediction_time"],
        "label_value": record["label"],
        "task": record["task"],
        "task_query": task_query,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_dir", required=True,
                   help="Directory with serialized JSONs (from 01_serialize)")
    p.add_argument("--output_dir", required=True,
                   help="Where to write SFT datasets")
    p.add_argument("--tasks", default="",
                   help="Comma-separated task subset (default: all)")
    return p.parse_args()


def main():
    args = parse_args()
    task_set = set(args.tasks.split(",")) if args.tasks else set(ALL_TASK_NAMES)
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    for task in sorted(task_set):
        task_query = TASKS[task]
        task_dir = input_root / task
        if not task_dir.exists():
            print(f"  [skip] {task}: no serialized data at {task_dir}")
            continue

        for split in ("train", "val", "test"):
            src = task_dir / f"{split}.json"
            if not src.exists():
                continue
            with open(src) as f:
                records = json.load(f)

            sft_entries = [make_sft_entry(r, task_query) for r in records]

            dst_dir = output_root / split
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{task}.json"
            with open(dst, "w") as f:
                json.dump(sft_entries, f, indent=2)

            pos = sum(1 for e in sft_entries if e["label_value"])
            neg = len(sft_entries) - pos
            print(f"  {task}/{split}: {len(sft_entries)} entries "
                  f"(pos={pos}, neg={neg}) => {dst}")

    print(f"\nDone. SFT datasets written to {output_root}")


if __name__ == "__main__":
    main()
