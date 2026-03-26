#!/usr/bin/env python3
"""
Create SFT datasets from rubricified patient records.

Reads the outputs of apply_rubric.py and converts them into the same
conversation format used by 02_create_sft, substituting the rubricified
text for the original naivetext serialization.

Inputs:
  --input_dir  : Directory with rubricified JSONs ({task}/{split}.json).
  --output_dir : Where to write SFT datasets.
  --tasks      : Space-separated list (default: all 15).

Outputs:
  {output_dir}/{split}/{task}.json

Connects to:
  - Upstream  : apply_rubric.py
  - Downstream: 05_eval
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.tasks import TASKS, ALL_TASK_NAMES, SYSTEM_MESSAGE


def _build_user_prompt(task_query: str, rubricified_text: str) -> str:
    return (
        f"Based on the patient's EHR below, predict: {task_query}\n\n"
        f"--- Patient EHR ---\n{rubricified_text}\n--- End of EHR ---\n\n"
        f"Based on the above EHR, predict: {task_query}\n"
        "Respond with exactly one word: Yes or No."
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tasks", nargs="+", default=ALL_TASK_NAMES)
    return p.parse_args()


def main():
    args = parse_args()
    in_root = Path(args.input_dir)
    out_root = Path(args.output_dir)

    for task in args.tasks:
        query = TASKS[task]
        for split in ("train", "val", "test"):
            src = in_root / task / f"{split}.json"
            if not src.exists():
                continue
            with open(src) as f:
                records = json.load(f)

            entries = []
            for r in records:
                entries.append({
                    "conversations": [
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        {"role": "user",
                         "content": _build_user_prompt(query, r["rubricified_text"])},
                        {"role": "assistant",
                         "content": "Yes" if r["label"] else "No"},
                    ],
                    "patient_id": r["patient_id"],
                    "label_time": r["prediction_time"],
                    "label_value": r["label"],
                    "task": task,
                    "task_query": query,
                })

            dst_dir = out_root / split
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{task}.json"
            with open(dst, "w") as f:
                json.dump(entries, f, indent=2)
            pos = sum(1 for e in entries if e["label_value"])
            print(f"  {task}/{split}: {len(entries)} entries "
                  f"(pos={pos}, neg={len(entries)-pos}) => {dst}")

    print(f"\nDone. LLM-rubric SFT datasets in {out_root}")


if __name__ == "__main__":
    main()
