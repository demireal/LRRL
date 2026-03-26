#!/usr/bin/env python3
"""
Use GPT-5.2 to generate a task-specific Python parser script for rubric filling.

This is the "plus" variant of create_rubric_auto.py. The key difference is that
each training cohort example shown to the model is augmented with a rubric version
of that same patient's serialization, already produced by an LLM. This gives the
model both the raw EHR text AND the gold-standard rubric fill, helping it learn
the extraction mapping more precisely.

For each task, this script:
  1. Loads the 40-patient cohort (from build_cohort.py / create_rubric.py).
  2. Loads the rubric instructions (from create_rubric.py).
  3. Loads the LLM-produced rubricified train split from
       {rubricified_dir}/{task}/train.json
     and matches each cohort patient by (patient_id, prediction_time).
  4. Prompts GPT-5.2 to write a self-contained Python script that parses any
     patient's plaintext EHR serialization and fills in the rubric template
     using only deterministic string/regex extraction (no LLM calls per patient).
  5. Saves the generated Python script to {output_dir}/{task}_parser.py.

Inputs:
  --cohort_dir       : Directory with {task}/cohort.json files (e.g. data/rubric).
  --rubric_dir       : Directory with {task}/rubric.json files (e.g. data/rubric).
  --rubricified_dir  : Root dir of LLM-rubricified data ({task}/train.json).
  --output_dir       : Where to write the generated parser scripts.
  --tasks            : Space-separated list (default: all 15).
  --azure_config     : Path to azure_config_gpt52.json.

Outputs:
  {output_dir}/{task}_parser.py   — generated Python parser (one per task)

The generated parser script interface:
  python {task}_parser.py \\
    --input_dir  data/serialized/naivetext \\
    --output_dir data/llmrubric-auto-plus \\
    --task       {task} \\
    --splits     train val test

Connects to:
  - Upstream  : build_cohort.py, create_rubric.py, apply_rubric.py
  - Downstream: the generated {task}_parser.py → data/llmrubric-auto-plus/
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import AzureOpenAI, BadRequestError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.tasks import TASKS, ALL_TASK_NAMES, SEED
from config.azure import AzureConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _build_rubricified_index(rubricified_train_path: Path) -> Dict[tuple, str]:
    """Load rubricified train.json and index by (patient_id, prediction_time)."""
    if not rubricified_train_path.exists():
        logger.warning(f"  rubricified train not found at {rubricified_train_path}")
        return {}
    with open(rubricified_train_path) as f:
        records = json.load(f)
    index = {}
    for r in records:
        key = (r["patient_id"], r["prediction_time"])
        index[key] = r.get("rubricified_text", "")
    return index


def _format_cohort_examples(
    examples: List[dict],
    rubricified_index: Dict[tuple, str],
    max_per_class: int = 20,
) -> str:
    """Format cohort patients as annotated examples, augmented with their LLM rubric."""
    pos = [e for e in examples if e.get("label", False)][:max_per_class]
    neg = [e for e in examples if not e.get("label", False)][:max_per_class]

    lines = []
    lines.append(
        f"Total examples: {len(examples)} "
        f"({len(pos)} positive / {len(neg)} negative)"
    )
    lines.append("")

    for group_label, group in [("POSITIVE (label=True)", pos),
                                ("NEGATIVE (label=False)", neg)]:
        lines.append(f"{'='*60}")
        lines.append(f"  {group_label} EXAMPLES")
        lines.append(f"{'='*60}")
        for i, e in enumerate(group, 1):
            pid = e.get("patient_id", "unknown")
            pred_time = e.get("prediction_time", "")
            lines.append(f"\n--- {group_label} Example {i} | patient_id={pid} ---")
            lines.append("")
            lines.append("[RAW EHR SERIALIZATION]")
            lines.append(e.get("serialization", ""))

            rubric_text = rubricified_index.get((pid, pred_time), "")
            if rubric_text:
                lines.append("")
                lines.append("[LLM-PRODUCED RUBRIC FILL FOR THIS PATIENT]")
                lines.append(rubric_text)
            else:
                lines.append("")
                lines.append("[LLM-PRODUCED RUBRIC FILL FOR THIS PATIENT]")
                lines.append("(not available for this example)")

    return "\n".join(lines)


def _build_prompt(
    task: str,
    task_query: str,
    rubric_instructions: str,
    cohort_examples: List[dict],
    rubricified_index: Dict[tuple, str],
    max_per_class: int = 20,
) -> str:
    examples_text = _format_cohort_examples(cohort_examples, rubricified_index,
                                            max_per_class=max_per_class)

    return f"""You are an expert Python developer and medical informaticist.

## Your Task

Write a complete, self-contained Python script that reads patient EHR serializations and fills in a structured clinical rubric template using **deterministic string/regex parsing only** — no LLM API calls, no network requests.

## Clinical Task Context

- Task name: {task}
- Prediction query: {task_query}

## Rubric Template to Fill

The script must fill in every field defined in the following rubric instructions:

{rubric_instructions}

## EHR Serialization Format

Below are 40 example patient EHR serializations from the training cohort, labeled by ground-truth outcome.
For each patient you are shown BOTH:
  1. The raw plaintext EHR serialization.
  2. The LLM-produced rubric fill for that exact patient — showing you how the fields should be extracted from the raw text.

Use these paired examples to understand the extraction mapping precisely.

{examples_text}

## Required Script Interface

The generated script must:

1. Accept the following command-line arguments via argparse:
   - `--input_dir`  : root directory of plaintext serializations (e.g. `data/serialized/naivetext`)
   - `--output_dir` : root directory for llmrubric-auto-plus outputs (e.g. `data/llmrubric-auto-plus`)
   - `--task`       : task name (default: `{task}`)
   - `--splits`     : one or more of `train val test` (default: all three)

2. For each split, read `{{input_dir}}/{{task}}/{{split}}.json` — a JSON array where each element has:
   - `patient_id` (int)
   - `prediction_time` (ISO datetime string)
   - `task` (str)
   - `split` (str)
   - `label` (bool)
   - `serialization` (str) ← the EHR text to parse

3. For each patient call `fill_rubric(serialization: str) -> str`, which:
   - Extracts all rubric fields from the EHR text using regex and string operations
   - Returns a filled-in rubric string that follows the exact field names, order, and format from the rubric template above
   - Writes "NA" for any field whose data is absent from the EHR

4. Write output to `{{output_dir}}/{{task}}/{{split}}.json` — a JSON array where each element has:
   - `patient_id` (int)
   - `prediction_time` (str)
   - `task` (str)
   - `split` (str)
   - `label` (bool)
   - `rubricified_text` (str) ← output of fill_rubric()

5. Create output directories as needed (parents=True, exist_ok=True).

6. Print progress to stdout: total patients processed per split.

## Constraints

- Use only Python standard library plus `re`, `json`, `argparse`, `pathlib`, `sys`. No third-party packages.
- No LLM API calls, no network requests, no subprocess calls to external tools.
- The `fill_rubric` function must be deterministic and handle missing data gracefully (write "NA" rather than raising exceptions).
- The script must be syntactically valid Python 3.8+.
- Do NOT hardcode file paths — use the argparse arguments.

## Output

Output ONLY the Python script, with no explanation, no preamble, and no markdown fences. The output must start with `#!/usr/bin/env python3` and be directly writable to a .py file."""


def _is_context_length_error(exc: BadRequestError) -> bool:
    """Return True if the API error is a context-length / token-limit error."""
    msg = str(exc).lower()
    code = getattr(exc, "code", "") or ""
    return (
        "context_length_exceeded" in code
        or "context_length_exceeded" in msg
        or "maximum context length" in msg
        or "reduce the length" in msg
        or "token" in msg and "limit" in msg
    )


def generate_parser_script(
    task: str,
    task_query: str,
    rubric_instructions: str,
    cohort_examples: List[dict],
    rubricified_index: Dict[tuple, str],
    config: AzureConfig,
    max_per_class: int = 20,
) -> Dict[str, Any]:
    client = AzureOpenAI(
        api_version=config.api_version,
        azure_endpoint=config.endpoint,
        api_key=config.api_key,
    )
    prompt = _build_prompt(
        task, task_query, rubric_instructions, cohort_examples, rubricified_index,
        max_per_class=max_per_class,
    )
    logger.info(f"  prompt length: {len(prompt):,} chars  (max_per_class={max_per_class})")

    resp = client.chat.completions.create(
        model=config.deployment,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert Python developer and medical informaticist. "
                    "You write clean, robust Python scripts for clinical data processing."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=config.max_completion_tokens,
        temperature=config.temperature,
    )
    raw = resp.choices[0].message.content.strip()
    usage = {
        "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
        "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
    }
    logger.info(f"  script generated ({usage.get('completion_tokens', '?')} completion tokens)")
    return {"code": raw, "usage": usage}


def _strip_fences(code: str) -> str:
    """Remove markdown code fences if GPT wrapped the output in them."""
    code = code.strip()
    code = re.sub(r'^```(?:python)?\s*\n', '', code)
    code = re.sub(r'\n```\s*$', '', code)
    return code.strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cohort_dir", required=True,
                   help="Directory with {task}/cohort.json files (e.g. data/rubric)")
    p.add_argument("--rubric_dir", required=True,
                   help="Directory with {task}/rubric.json files (e.g. data/rubric)")
    p.add_argument("--rubricified_dir", required=True,
                   help="Root dir of LLM-rubricified data; expects {task}/train.json")
    p.add_argument("--output_dir", required=True,
                   help="Directory to write generated parser scripts")
    p.add_argument("--azure_config", default=None,
                   help="Path to azure_config_gpt52.json")
    p.add_argument("--tasks", nargs="+", default=ALL_TASK_NAMES)
    return p.parse_args()


def _resolve_dir(path_str: str) -> Path:
    p = Path(path_str)
    return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def main():
    args = parse_args()
    config = AzureConfig.from_json(args.azure_config)
    config.max_completion_tokens = 32768
    config.temperature = 1.0

    cohort_dir = _resolve_dir(args.cohort_dir)
    rubric_dir = _resolve_dir(args.rubric_dir)
    rubricified_dir = _resolve_dir(args.rubricified_dir)
    output_dir = _resolve_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for task in args.tasks:
        logger.info(f"\n{'='*60}\nGenerating plus-parser for: {task}\n{'='*60}")

        out_path = output_dir / f"{task}_parser.py"
        if out_path.exists():
            logger.info(f"  parser already exists at {out_path}, skipping")
            continue

        cohort_path = cohort_dir / task / "cohort.json"
        if not cohort_path.exists():
            logger.warning(f"  cohort not found at {cohort_path}, skipping")
            continue

        rubric_path = rubric_dir / task / "rubric.json"
        if not rubric_path.exists():
            logger.warning(f"  rubric not found at {rubric_path}, skipping")
            continue

        with open(cohort_path) as f:
            cohort_examples = json.load(f)
        with open(rubric_path) as f:
            rubric_data = json.load(f)

        rubric_instructions = rubric_data["rubric_instructions"]
        task_query = rubric_data.get("task_query", TASKS.get(task, ""))

        rubricified_train_path = rubricified_dir / task / "train.json"
        rubricified_index = _build_rubricified_index(rubricified_train_path)
        matched = sum(
            1 for e in cohort_examples
            if (e.get("patient_id"), e.get("prediction_time")) in rubricified_index
        )
        logger.info(
            f"  rubricified index: {len(rubricified_index)} records loaded, "
            f"{matched}/{len(cohort_examples)} cohort examples matched"
        )

        max_per_class = 20
        result = None
        while max_per_class >= 1:
            try:
                result = generate_parser_script(
                    task, task_query, rubric_instructions,
                    cohort_examples, rubricified_index, config,
                    max_per_class=max_per_class,
                )
                break
            except BadRequestError as exc:
                if _is_context_length_error(exc):
                    logger.warning(
                        f"  context length exceeded with max_per_class={max_per_class}, "
                        f"retrying with {max_per_class - 1}..."
                    )
                    max_per_class -= 1
                else:
                    raise

        if result is None:
            logger.error(f"  failed to generate parser for {task} even with 1 example per class, skipping")
            continue

        code = _strip_fences(result["code"])

        with open(out_path, "w") as f:
            f.write(code)
            if not code.endswith("\n"):
                f.write("\n")

        logger.success(
            f"  saved -> {out_path}  "
            f"(max_per_class={max_per_class}, "
            f"prompt={result['usage']['prompt_tokens']} tokens, "
            f"completion={result['usage']['completion_tokens']} tokens)"
        )

    logger.success("Done. All plus-parser scripts generated.")


if __name__ == "__main__":
    main()
