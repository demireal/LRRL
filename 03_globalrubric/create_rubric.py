#!/usr/bin/env python3
"""
Generate task-specific rubric instructions using GPT-5-mini.

For each task, this script:
  1. Loads the 40-patient cohort (from build_cohort.py).
  2. Formats examples with their ground-truth labels.
  3. Prompts GPT-5-mini to produce a step-by-step rubric template.
  4. Saves the rubric JSON.

Inputs:
  --cohort_dir  : Directory with {task}/cohort.json files.
  --output_dir  : Where to write rubric JSONs.
  --tasks       : Space-separated list (default: all 15).

Outputs:
  {output_dir}/{task}/rubric.json

GPT-5-mini parameters: max_completion_tokens=16384, temperature=1.

Connects to:
  - Upstream  : build_cohort.py
  - Downstream: apply_rubric.py
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger
from openai import AzureOpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.tasks import TASKS, ALL_TASK_NAMES, SEED
from config.azure import AzureConfig


def _format_example(example: dict, task_query: str) -> str:
    label = "Yes" if example.get("label", False) else "No"
    ctx = example.get("serialization", "")
    pid = example.get("patient_id", "unknown")
    return f"[Example {pid}] Ground Truth: {label}\n{ctx}"


def _build_prompt(task: str, examples: List[dict], task_query: str) -> str:
    random.seed(SEED)

    pos_examples = [e for e in examples if e.get("label", False)]
    neg_examples = [e for e in examples if not e.get("label", False)]

    formatted_pos = [_format_example(e, task_query) for e in pos_examples]
    formatted_neg = [_format_example(e, task_query) for e in neg_examples]
    random.shuffle(formatted_pos)
    random.shuffle(formatted_neg)

    pos_text = "\n\n---NEW POSITIVE EXAMPLE---\n\n".join(formatted_pos)
    neg_text = "\n\n---NEW NEGATIVE EXAMPLE---\n\n".join(formatted_neg)

    return f"""You are a medical expert designing a structured rubric for a clinical prediction task.

## Task
- Name: {task.replace('_', ' ').title()}
- Query: {task_query}

## Context

You will be given {len(examples)} labeled patient EHR examples ({len(pos_examples)} positive, {len(neg_examples)} negative). Another model will later use your rubric to transform new patient EHRs into structured summaries, which will then serve as input to a supervised classifier.

## What You Must Do

Study the examples below. Combine what you observe in them with your medical knowledge to design a rubric template -- a set of named fields that, when filled in for any patient, produce a structured summary optimized for this prediction task.

The rubric should:
1. **Be data-driven and discriminative.** Identify which features, patterns, and interactions actually separate the positive and negative cases. The rubric should capture not just obvious indicators but also subtler or compound features you notice. At the same time, do not overfit to these 40 cases -- use your clinical knowledge to include factors that are generally relevant even if not prominent in this sample.
2. **Be structured and consistent.** Every rubricified output must follow the same field names and order. For each field, specify what to extract from the EHR and how to format it. Specify what to write when data is absent.
3. **Extract facts only.** The evaluator filling in the rubric must extract and organize information from the EHR. It must NOT make predictions, assign risk levels, or draw conclusions.
4. **Be concise.** The rubric should focus on extracting information that is relevant to the task. It should not ask the evaluator to reproduce the entire EHR.

## Examples

### Positive Examples (Ground Truth: Yes)

---NEW POSITIVE EXAMPLE---

{pos_text}

### Negative Examples (Ground Truth: No)

---NEW NEGATIVE EXAMPLE---

{neg_text}

## Output

Output ONLY the rubric template itself -- the instructions another model will follow to transform a patient EHR. No preamble, no explanation of your reasoning. The template must be self-contained and directly usable."""


def generate_rubric(task: str, examples: List[dict],
                    task_query: str, config: AzureConfig) -> Dict[str, Any]:
    client = AzureOpenAI(
        api_version=config.api_version,
        azure_endpoint=config.endpoint,
        api_key=config.api_key,
    )
    prompt = _build_prompt(task, examples, task_query)
    logger.info(f"  prompt length: {len(prompt)} chars")

    resp = client.chat.completions.create(
        model=config.deployment,
        messages=[
            {"role": "system",
             "content": ("You are a medical expert AI assistant specializing in "
                         "creating structured clinical evaluation rubrics.")},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=config.max_completion_tokens,
        temperature=config.temperature,
    )
    text = resp.choices[0].message.content.strip()
    usage = {
        "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
        "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
    }
    logger.info(f"  rubric generated ({usage.get('completion_tokens', '?')} completion tokens)")
    return {
        "task": task,
        "task_query": task_query,
        "rubric_instructions": text,
        "num_examples": len(examples),
        "usage": usage,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohort_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--azure_config", default=None)
    p.add_argument("--tasks", nargs="+", default=ALL_TASK_NAMES)
    return p.parse_args()


def main():
    args = parse_args()
    config = AzureConfig.from_json(args.azure_config)
    # Rubric creation benefits from lower temperature than application:
    # we want a structured, deterministic template, not creative variation.
    config.max_completion_tokens = 16384
    config.temperature = 1.0

    for task in args.tasks:
        logger.info(f"\n{'='*60}\nCreating rubric for: {task}\n{'='*60}")
        cohort_path = Path(args.cohort_dir) / task / "cohort.json"
        if not cohort_path.exists():
            logger.warning(f"  cohort not found at {cohort_path}, skipping")
            continue
        with open(cohort_path) as f:
            examples = json.load(f)

        result = generate_rubric(task, examples, TASKS[task], config)

        out_dir = Path(args.output_dir) / task
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "rubric.json", "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"  saved -> {out_dir / 'rubric.json'}")

    logger.success("All rubrics created.")


if __name__ == "__main__":
    main()
