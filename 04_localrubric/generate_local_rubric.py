#!/usr/bin/env python3
"""
Generate local rubric representations using GPT-5-mini.

Local rubrics are task-conditioned summaries where the model characterizes
the patient's risk profile without knowing the ground-truth label.

The generated representations are used in two ways:
  1. As input representations for embedding-based evaluation
     (generate_embeddings.py -> eval_embeddings.py).

We generate traces for train + val + test splits:
  - train : training data for both direct FT and embedding approaches
  - val   : eval loss during direct FT / hyperparameter C selection for embeddings
  - test  : final evaluation for both approaches

Inputs:
  --sft_dir   : Naivetext SFT directory (data/sft/naivetext).
  --output_dir: Where to write CoT SFT datasets.
  --tasks     : Space-separated list (default: all 15).
  --splits    : Which splits to process (default: train val test).
  --minimal_instructions: Ablation — no prescribed output format (no <think> structure).
  --general_purpose_summaries: Ablation — general-purpose clinical summaries (no task-specific instructions).

Outputs:
  {output_dir}/{split}/{task}.json  (SFT conversation format)

GPT-5-mini parameters: max_completion_tokens=16384, temperature=1.
Incremental saving + resumability.

Connects to:
  - Upstream  : 02_create_sft (naivetext)
  - Downstream: 05_eval/generate_embeddings.py
"""

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Set

from loguru import logger
from openai import AzureOpenAI, RateLimitError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.tasks import TASKS, ALL_TASK_NAMES, SYSTEM_MESSAGE
from config.azure import AzureConfig


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _generation_prompt(ehr_text: str, task_query: str) -> str:
    return (
        "GOAL: Read the patient's text-serialized EHR and write a compact reasoning trace "
        "that characterizes the patient's risk profile for the following clinical "
        f"outcome prediction task: {task_query}\n\n"
        f"### INPUT EHR DATA\n\n"
        f"--- START OF EHR DATA ---\n{ehr_text}\n--- END OF EHR DATA ---\n\n"
        "### OUTPUT FORMAT\n"
        "Your output MUST follow this exact structure:\n"
        "<think>\n"
        "1. ### PATIENT SNAPSHOT\n"
        "2. ### MAIN RISK FACTORS\n"
        "3. ### PROTECTIVE FACTORS\n"
        "4. ### WHAT'S UNKNOWN / COULD SWING THE RISK\n"
        "5. ### WEIGHING AND AGGREGATING THE EVIDENCE\n"
        "6. ### CONCLUSION AND AN OVERALL RISK IMPRESSION\n"
        "</think>"
    )


def _generation_prompt_minimal(ehr_text: str, task_query: str) -> str:
    """Ablation: no prescribed output format — minimal instructions only."""
    return (
        f"Read the patient's EHR below and write a compact summary that is useful for the task query: {task_query}\n\n"
        f"Do not make any predictions, just create a summary in the context of the task query.\n\n"
        f"--- START OF EHR DATA ---\n{ehr_text}\n--- END OF EHR DATA ---"
    )


def _generation_prompt_general_purpose(ehr_text: str, task_query: str) -> str:
    """Ablation: general-purpose clinical summary — no task-specific instructions."""
    return (
        "Read the patient's EHR below and write a compact clinical summary that characterizes "
        "the patient's overall health status, key medical conditions, risk factors, and relevant "
        "clinical context. Focus on information that would be useful for clinical decision-making.\n\n"
        "Do not make predictions about any specific outcome. Provide a general-purpose summary "
        "that captures the essential clinical information.\n\n"
        f"--- START OF EHR DATA ---\n{ehr_text}\n--- END OF EHR DATA ---"
    )


def _sft_user_prompt(task_query: str, ehr_text: str) -> str:
    """User prompt stored in the SFT dataset (for direct FT / embedding)."""
    return (
        f"Based on the patient's EHR below, predict: {task_query}\n\n"
        f"--- Patient EHR ---\n\n{ehr_text}\n\n--- End of EHR ---\n\n"
        f"Based on the above EHR, predict: {task_query}\n"
        "Respond with exactly one word: Yes or No."
    )


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def _use_responses_api(config: AzureConfig) -> bool:
    """GPT-5.x models require the Responses API instead of Chat Completions."""
    dep = (config.deployment or "").lower()
    return "gpt-5." in dep and dep != "gpt-5-mini"


def _generate_trace(record: dict, task_query: str,
                    config: AzureConfig, use_minimal_instructions: bool = False,
                    use_general_purpose: bool = False,
                    max_retries: int = 5) -> Dict[str, Any]:
    client = AzureOpenAI(
        api_version=config.api_version,
        azure_endpoint=config.endpoint,
        api_key=config.api_key,
    )
    ehr_text = record["conversations"][1]["content"]
    start = ehr_text.find("--- Patient EHR ---\n") + len("--- Patient EHR ---\n")
    end = ehr_text.find("\n--- End of EHR ---")
    ehr_text = ehr_text[start:end]

    if use_general_purpose:
        prompt = _generation_prompt_general_purpose(ehr_text, task_query)
    elif use_minimal_instructions:
        prompt = _generation_prompt_minimal(ehr_text, task_query)
    else:
        prompt = _generation_prompt(ehr_text, task_query)

    use_responses = _use_responses_api(config)
    messages = [
        {"role": "system",
         "content": "You are a clinical assistant specializing in "
                    "risk profile inference."},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(max_retries):
        try:
            if use_responses:
                resp = client.responses.create(
                    model=config.deployment,
                    input=messages,
                    max_output_tokens=config.max_completion_tokens,
                    temperature=config.temperature,
                )
                trace = resp.output_text.strip()
            else:
                resp = client.chat.completions.create(
                    model=config.deployment,
                    messages=messages,
                    max_completion_tokens=config.max_completion_tokens,
                    temperature=config.temperature,
                )
                trace = resp.choices[0].message.content.strip()
            if trace:
                # The "serialization" is the trace itself.
                # We wrap it in the SFT format so generate_embeddings
                # can consume it uniformly.
                return {
                    "conversations": [
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        {"role": "user",
                         "content": _sft_user_prompt(task_query, trace)},
                        {"role": "assistant",
                         "content": "Yes" if record["label_value"] else "No"},
                    ],
                    "patient_id": record["patient_id"],
                    "label_time": record["label_time"],
                    "label_value": record["label_value"],
                    "task": record["task"],
                    "task_query": task_query,
                    "unsupervised_trace": trace,
                }
        except RateLimitError:
            delay = (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"Rate limit (patient {record['patient_id']}), "
                           f"retry in {delay:.1f}s")
            time.sleep(delay)
        except Exception as e:
            logger.error(f"Error for patient {record['patient_id']}: {e}")
            break

    return None


# ---------------------------------------------------------------------------
# Incremental save / resume
# ---------------------------------------------------------------------------

def _load_done(path: Path) -> Set:
    if not path.exists():
        return set()
    try:
        with open(path) as f:
            data = json.load(f)
        return {(e["patient_id"], e["label_time"]) for e in data}
    except Exception:
        return set()


def _save_inc(entry: dict, path: Path, lock: threading.Lock):
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
    p.add_argument("--sft_dir", required=True,
                   help="Naivetext SFT dir (data/sft/naivetext)")
    p.add_argument("--output_dir", required=True,
                   help="Where to write local rubric SFT datasets")
    p.add_argument("--tasks", nargs="+", default=ALL_TASK_NAMES)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--max_workers", type=int, default=40)
    p.add_argument("--azure_config", default=None)
    p.add_argument("--minimal_instructions", action="store_true",
                   help="Ablation: no prescribed output format; only minimal task instruction.")
    p.add_argument("--general_purpose_summaries", action="store_true",
                   help="Ablation: general-purpose clinical summaries (no task-specific instructions).")
    return p.parse_args()


def main():
    args = parse_args()
    if args.general_purpose_summaries:
        logger.info("Ablation: general-purpose clinical summaries (no task-specific instructions)")
    elif args.minimal_instructions:
        logger.info("Ablation: minimal instructions (no prescribed output format)")
    config = AzureConfig.from_json(args.azure_config)
    config.max_completion_tokens = 16384
    config.temperature = 1.0
    lock = threading.Lock()

    for task in args.tasks:
        for split in args.splits:
            src = Path(args.sft_dir) / split / f"{task}.json"
            if not src.exists():
                logger.info(f"  {task}/{split}: no data, skipping")
                continue
            with open(src) as f:
                records = json.load(f)

            out_dir = Path(args.output_dir) / split
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{task}.json"
            done = _load_done(out_path)
            todo = [r for r in records
                    if (r["patient_id"], r["label_time"]) not in done]
            logger.info(f"{task}/{split}: {len(todo)} to generate "
                        f"({len(done)} already done)")
            if not todo:
                continue

            with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
                futs = {
                    pool.submit(_generate_trace, r, TASKS[task], config,
                                args.minimal_instructions, args.general_purpose_summaries): r
                    for r in todo
                }
                for fut in as_completed(futs):
                    result = fut.result()
                    if result is not None:
                        _save_inc(result, out_path, lock)

            logger.info(f"  {task}/{split} done -> {out_path}")

    logger.success("Unsupervised CoT generation complete.")


if __name__ == "__main__":
    main()
