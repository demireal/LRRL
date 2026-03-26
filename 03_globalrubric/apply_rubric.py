#!/usr/bin/env python3
"""
Apply a task-specific rubric to every patient's naivetext EHR via GPT-5-mini.

For each task, this script:
  1. Loads the rubric instructions (from create_rubric.py).
  2. Loads serialized patient records (from 01_serialize naivetext).
  3. Sends each patient's EHR + rubric to GPT-5-mini.
  4. Saves the rubricified text per patient (incremental, resumable).

Inputs:
  --rubric_dir     : Directory with {task}/rubric.json.
  --serialized_dir : Naivetext serialized dir (data/serialized/naivetext).
  --output_dir     : Where to write rubricified JSONs.
  --tasks          : Space-separated list (default: all 15).
  --splits         : Which splits to process (default: train val test).
  --max_workers    : Parallel threads for API calls (default: 8).

Outputs:
  {output_dir}/{task}/{split}.json
  Each record has: patient_id, prediction_time, task, split, label,
                   rubricified_text (the GPT-5-mini output).

GPT-5-mini parameters: max_completion_tokens=16384, temperature=1.

Connects to:
  - Upstream  : create_rubric.py, 01_serialize (naivetext)
  - Downstream: create_globalrubric_sft.py
"""

import argparse
import json
import os
import sys
import threading
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from loguru import logger
from openai import AzureOpenAI, RateLimitError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.tasks import TASKS, ALL_TASK_NAMES
from config.azure import AzureConfig

# Project root (ehrshot-v2); used to resolve relative dir paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _build_transform_prompt(ehr_text: str, rubric_instructions: str,
                            task_query: str) -> str:
    return f"""You are a medical data extraction specialist. Your job is to read a patient's EHR and fill in a structured rubric template.

## Task
{task_query}

## Rubric Template (follow this exactly)

{rubric_instructions}

## Patient EHR

{ehr_text}

## Instructions
Fill in every field of the rubric template above using ONLY information from this patient's EHR. Rules:
- Follow the exact field order and section structure of the rubric.
- Be concise: use short phrases, numbers, and dates. Do not write paragraphs.
- If data for a field is not present in the EHR, write "No data".
- Do NOT add commentary, predictions, risk assessments, or conclusions.
- Do NOT include any information not found in the EHR above.

Rubric output:"""


def _transform_one(record: dict, rubric_text: str, task_query: str,
                   config: AzureConfig, max_retries: int = 5) -> Dict[str, Any]:
    client = AzureOpenAI(
        api_version=config.api_version,
        azure_endpoint=config.endpoint,
        api_key=config.api_key,
    )
    prompt = _build_transform_prompt(
        record["serialization"], rubric_text, task_query)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=config.deployment,
                messages=[
                    {"role": "system",
                     "content": ("You are a medical expert AI assistant specializing "
                                 "in structured clinical evaluation.")},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=config.max_completion_tokens,
                temperature=config.temperature,
            )
            text = resp.choices[0].message.content
            if text and text.strip():
                return {
                    "patient_id": record["patient_id"],
                    "prediction_time": record["prediction_time"],
                    "task": record["task"],
                    "split": record["split"],
                    "label": record["label"],
                    "rubricified_text": text.strip(),
                }
        except RateLimitError:
            delay = (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"Rate limit (patient {record['patient_id']}), "
                           f"retry in {delay:.1f}s")
            time.sleep(delay)
        except Exception as e:
            logger.error(f"Error for patient {record['patient_id']}: {e}")
            break

    # Fallback
    return {
        "patient_id": record["patient_id"],
        "prediction_time": record["prediction_time"],
        "task": record["task"],
        "split": record["split"],
        "label": record["label"],
        "rubricified_text": "[ERROR: transformation failed]",
    }


def _load_done_keys(path: Path) -> Set:
    if not path.exists():
        return set()
    try:
        with open(path) as f:
            data = json.load(f)
        return {(e["patient_id"], e["prediction_time"]) for e in data}
    except Exception:
        return set()


def _save_incremental(entry: dict, path: Path, lock: threading.Lock):
    with lock:
        data = []
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
            except Exception:
                pass
        data.append(entry)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rubric_dir", required=True)
    p.add_argument("--serialized_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tasks", nargs="+", default=ALL_TASK_NAMES)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--max_workers", type=int, default=20)
    p.add_argument("--azure_config", default=None)
    return p.parse_args()


def _resolve_dir(path_str: str) -> Path:
    """Resolve to absolute path; if relative, interpret relative to project root."""
    p = Path(path_str)
    return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def main():
    args = parse_args()
    rubric_dir = _resolve_dir(args.rubric_dir)
    serialized_dir = _resolve_dir(args.serialized_dir)
    output_dir = _resolve_dir(args.output_dir)

    config = AzureConfig.from_json(args.azure_config)
    config.max_completion_tokens = 16384
    config.temperature = 1.0
    lock = threading.Lock()

    for split in args.splits:
        logger.info(f"=== Split: {split} ===")
        for task in args.tasks:
            rubric_path = rubric_dir / task / "rubric.json"
            if not rubric_path.exists():
                logger.warning(f"No rubric for {task}, skipping")
                continue
            with open(rubric_path) as f:
                rubric_text = json.load(f)["rubric_instructions"]

            src = serialized_dir / task / f"{split}.json"
            if not src.exists():
                continue
            with open(src) as f:
                records = json.load(f)

            out_path = output_dir / task / f"{split}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            done = _load_done_keys(out_path)
            todo = [r for r in records
                    if (r["patient_id"], r["prediction_time"]) not in done]
            logger.info(f"  {task}/{split}: {len(todo)} to do "
                       f"({len(done)} already done)")
            if not todo:
                continue

            with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
                futs = {
                    pool.submit(_transform_one, r, rubric_text,
                                TASKS[task], config): r
                    for r in todo
                }
                for fut in as_completed(futs):
                    result = fut.result()
                    _save_incremental(result, out_path, lock)

            logger.info(f"  {task}/{split} done -> {out_path}")

    logger.success("All rubric transformations complete.")


if __name__ == "__main__":
    main()
