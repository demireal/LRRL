#!/usr/bin/env python3
"""
Transform each task's rubric_instructions into a single dictionary schema (keys to fill
per patient) using GPT-5-mini. The schema is designed so that once filled for each
patient, the result can be converted to a numerical feature vector.

For each task:
  1. Loads data/rubric/{task}/rubric.json (rubric_instructions).
  2. Asks GPT-5-mini to derive a JSON schema: one key per extractable field, with
     type/encoding hints (numeric, categorical, boolean, date, text) for downstream
     vectorization.
  3. Saves data/rubric/{task}/rubric_schema.json.

Inputs:
  --rubric_dir   : Directory containing {task}/rubric.json (default: data/rubric).
  --output_dir    : Where to write rubric_schema.json per task (default: same as rubric_dir).
  --azure_config  : Path to Azure OpenAI config JSON (optional).
  --tasks         : Space-separated list of tasks (default: all with existing rubric).

Outputs:
  {output_dir}/{task}/rubric_schema.json
  Each file is a JSON object: keys = field names to fill per patient;
  values = { "type": "numeric"|"categorical"|"boolean"|"date"|"text",
             "allowed_values": [...] (optional, for categorical) }.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from loguru import logger
from openai import AzureOpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config.azure import AzureConfig
from config.tasks import ALL_TASK_NAMES


def _build_schema_prompt(task: str, task_query: str, rubric_instructions: str) -> str:
    return f"""You are a medical data schema designer. Given a clinical extraction rubric, produce a **single JSON dictionary** that will be used as a template: each key is a field name to be filled for each patient from the EHR; once filled, the dictionary will be converted into a numerical feature vector for machine learning.

## Task
- Name: {task}
- Query: {task_query}

## Rubric instructions (extraction template)

{rubric_instructions}

## Your task

Derive from the rubric above a JSON object with one key per extractable field. The rubric may use sections, numbered items, or inline "Format:" specs—collapse these into a flat set of keys that cover the same information. Use short, machine-friendly key names (snake_case).

For each key, provide a value that is itself a small object with:
- **type**: one of "numeric", "categorical", "boolean", "date", "text"
  - numeric: a number (for scalars, counts, or one-hot index after binning)
  - categorical: one of a fixed set of strings; include "allowed_values" list
  - boolean: true/false (or Yes/No mapped to bool)
  - date: ISO date string (will be encoded as numeric or time delta)
  - text: free text (will be embedded or tokenized later)
- **allowed_values**: (optional) list of allowed strings for type "categorical"
- **description**: (optional) one short line describing the field

Requirements:
- Every distinct rubric field that can be filled from the EHR must become exactly one key.
- Prefer categorical with explicit allowed_values where the rubric specifies a format (e.g. Yes/No/NA, or a fixed set of encounter types).
- Use numeric for counts, lab values, ages, etc.; boolean for binary flags.
- Output ONLY valid JSON. No markdown, no code fence, no explanation. The response must parse as a single JSON object."""


def _extract_json(text: str) -> dict:
    """Parse JSON from model output, stripping markdown code blocks if present."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    return json.loads(text)


def create_schema_for_task(
    task: str,
    task_query: str,
    rubric_instructions: str,
    config: AzureConfig,
) -> dict:
    client = AzureOpenAI(
        api_version=config.api_version,
        azure_endpoint=config.endpoint,
        api_key=config.api_key,
    )
    prompt = _build_schema_prompt(task, task_query, rubric_instructions)
    logger.info(f"  prompt length: {len(prompt)} chars")

    resp = client.chat.completions.create(
        model=config.deployment,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical data schema expert. Output only valid JSON "
                    "with no surrounding text or markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=config.max_completion_tokens,
        temperature=config.temperature,
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise ValueError("Empty model response")

    schema = _extract_json(text)
    if not isinstance(schema, dict):
        raise ValueError("Model did not return a JSON object")

    usage = {}
    if resp.usage:
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
        }
    logger.info(f"  schema has {len(schema)} keys ({usage.get('completion_tokens', '?')} completion tokens)")
    return {
        "task": task,
        "task_query": task_query,
        "schema": schema,
        "usage": usage,
    }


def discover_tasks(rubric_dir: Path) -> list[str]:
    """Return task names that have rubric.json under rubric_dir."""
    out = []
    for p in rubric_dir.iterdir():
        if p.is_dir() and (p / "rubric.json").exists():
            out.append(p.name)
    return sorted(out)


def parse_args():
    rubric_default = PROJECT_ROOT / "data" / "rubric"
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--rubric_dir", type=Path, default=rubric_default,
                   help=f"Directory with {{task}}/rubric.json (default: {rubric_default})")
    p.add_argument("--output_dir", type=Path, default=None,
                   help="Where to write rubric_schema.json (default: same as rubric_dir)")
    p.add_argument("--azure_config", default=None)
    p.add_argument("--tasks", nargs="+", default=None,
                   help="Tasks to process (default: all with rubric.json under rubric_dir)")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or args.rubric_dir
    if not args.rubric_dir.exists():
        logger.error(f"Rubric dir not found: {args.rubric_dir}")
        sys.exit(1)

    tasks = args.tasks or discover_tasks(args.rubric_dir)
    if not tasks:
        logger.warning("No tasks with rubric.json found")
        sys.exit(0)

    config = AzureConfig.from_json(args.azure_config)
    config.max_completion_tokens = 8192

    for task in tasks:
        rubric_path = args.rubric_dir / task / "rubric.json"
        if not rubric_path.exists():
            logger.warning(f"Skip {task}: no {rubric_path}")
            continue
        logger.info(f"\n{'='*60}\nSchema for: {task}\n{'='*60}")
        with open(rubric_path) as f:
            rubric_data = json.load(f)
        task_query = rubric_data.get("task_query", "")
        rubric_instructions = rubric_data.get("rubric_instructions", "")
        if not rubric_instructions:
            logger.warning(f"  no rubric_instructions in {rubric_path}, skip")
            continue

        try:
            result = create_schema_for_task(
                task, task_query, rubric_instructions, config
            )
        except Exception as e:
            logger.exception(f"  failed: {e}")
            continue

        out_path = output_dir / task / "rubric_schema.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"  saved -> {out_path}")

    logger.success("Done.")


if __name__ == "__main__":
    main()
