#!/usr/bin/env python3
"""
Use GPT-5.2 to generate a task-specific Python featurizer that maps rubric
text into a fixed numeric feature vector.

For each task, this script:
  1. Loads the rubric parser source from the auto_parsers directory (so GPT-5.2
     understands the rubric field names and their text formats).
  2. Loads the 40-patient cohort and matches them to the SFT train split to
     extract rubric text examples.
  3. Prompts GPT-5.2 to write a self-contained Python featurizer script that
     parses rubric text into a fixed-dimension feature vector (numeric +
     binary/one-hot for categorical fields) with no LLM calls.
  4. Saves the generated script to {output_dir}/{task}_featurizer.py.

Inputs:
  --parsers_dir  : Directory with {task}_parser.py files (auto_parsers).
  --cohort_dir   : Root cohort directory (data/rubric) containing {task}/cohort.json.
  --sft_dir      : Root SFT directory (data/sft/llmrubric-auto).
  --output_dir   : Where to write generated featurizer scripts.
  --tasks        : Space-separated list (default: all 15).
  --azure_config : Path to azure_config_gpt52.json.

Outputs:
  {output_dir}/{task}_featurizer.py

The generated featurizer interface:
  python {task}_featurizer.py \\
    --input_dir  data/sft/llmrubric-auto \\
    --output_dir data/rubric_features_auto \\
    --task       {task} \\
    --splits     train val test

Connects to:
  - Upstream  : auto_parsers/{task}_parser.py, data/sft/llmrubric-auto
  - Downstream: 05_eval/eval_rubric_features.py
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import AzureOpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.tasks import TASKS, ALL_TASK_NAMES, SEED
from config.azure import AzureConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _extract_rubric_text(sft_record: dict) -> str:
    """Extract the rubric text from an SFT conversation record."""
    user_content = sft_record["conversations"][1]["content"]
    m = re.search(
        r"---\s*Patient EHR\s*---\s*\n(.*?)\n---\s*End of EHR\s*---",
        user_content,
        re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    return user_content


def _load_cohort_examples(cohort_path: Path, sft_train_path: Path) -> List[dict]:
    """Load 40 cohort patients and match them to SFT train records.

    cohort.json uses 'patient_id' + 'prediction_time'.
    SFT train records use 'patient_id' + 'label_time'.
    """
    with open(cohort_path) as f:
        cohort = json.load(f)

    with open(sft_train_path) as f:
        sft_records = json.load(f)

    sft_index: dict = {}
    for r in sft_records:
        key = (int(r["patient_id"]), str(r["label_time"]))
        sft_index[key] = r

    matched: List[dict] = []
    unmatched = 0
    for c in cohort:
        key = (int(c["patient_id"]), str(c["prediction_time"]))
        if key in sft_index:
            matched.append(sft_index[key])
        else:
            unmatched += 1

    if unmatched:
        logger.warning(f"  {unmatched}/{len(cohort)} cohort patients not found in SFT train split")

    return matched


def _format_examples(examples: List[dict]) -> str:
    lines = []
    for i, e in enumerate(examples, 1):
        label = "POSITIVE" if e.get("label_value", False) else "NEGATIVE"
        pid = e.get("patient_id", "unknown")
        rubric_text = _extract_rubric_text(e)
        lines.append(f"--- Example {i}/{len(examples)} | label={label} | patient_id={pid} ---")
        lines.append(rubric_text)
        lines.append("")
    return "\n".join(lines)


def _build_prompt(task: str, task_query: str, parser_source: str,
                  examples: List[dict]) -> str:
    examples_text = _format_examples(examples)
    n_pos = sum(1 for e in examples if e.get("label_value", False))
    n_neg = len(examples) - n_pos

    return f"""You are an expert Python developer and medical informaticist.

## Your Task

Write a complete, self-contained Python featurizer script that reads rubric-formatted patient EHR texts and converts each one into a **fixed-dimension numeric feature vector** using deterministic string/regex parsing — no LLM calls, no network requests.

## Clinical Task Context

- Task name: {task}
- Prediction query: {task_query}

## Rubric Parser Source (shows all rubric field names and their text formats)

The following is the parser that generates the rubric text. Study it to understand which fields exist and how their values are formatted in the text. This is the **ground truth** for what fields can appear in a rubric text and how their values are formatted.

```python
{parser_source}
```

## Reference Rubric Texts ({n_pos} positive, {n_neg} negative)

**Important context:** These {n_pos + n_neg} patients are the cohort that was used to *design* the rubric itself. They are provided as examples so you can calibrate your regex patterns against actual data.

**However**, the featurizer you write will be applied to a **much larger dataset** (thousands of patients). Your feature extraction logic must therefore be:
- **General**: handle any value the rubric parser could plausibly produce, not just the values seen in these 40 patients
- **Robust**: gracefully handle missing, NA, or unexpected values for every field
- **Comprehensive**: derive features from every field in the rubric, even if that field happens to be NA for all 40 examples shown here

Use the parser source above as the authoritative specification of fields and value formats; use the examples below to validate and calibrate your regex patterns.

{examples_text}

## Required Script Interface

The generated script must:

1. Accept CLI arguments via argparse:
   - `--input_dir`  : root of SFT datasets (e.g. `data/sft/llmrubric-auto`)
   - `--output_dir` : root for feature output (e.g. `data/rubric_features_auto`)
   - `--task`       : task name (default: `{task}`)
   - `--splits`     : one or more of `train val test` (default: all three)

2. For each split, read `{{input_dir}}/{{split}}/{{task}}.json` — a JSON array where each element has:
   - `patient_id` (int)
   - `label_time` (ISO datetime string)
   - `label_value` (bool)
   - `conversations` (list) — rubric text is in `conversations[1]["content"]` between `--- Patient EHR ---` and `--- End of EHR ---`

3. Implement `def extract_features(rubric_text: str) -> dict[str, float]`:
   - Parse every rubric field from the text
   - Return a flat dict mapping feature name → float value
   - For **numeric fields**: extract the number; if missing/NA write `0.0` and set `{{field}}_missing = 1.0`
   - For **categorical / Yes/No fields**: one-hot encode all known values; unknown/NA → all zeros plus a `{{field}}_missing = 1.0` indicator
   - All returned values must be float (0.0 or 1.0 for binary, numeric otherwise)
   - The dict must have the **same keys in the same order** for every call (fixed schema)

4. Define `SCHEMA: list[dict]` at module level — one entry per feature with keys:
   - `"name"`: feature name (matches key in extract_features output)
   - `"type"`: `"numeric"`, `"binary"`, or `"categorical"`
   - `"description"`: short human-readable description
   - `"possible_values"`: list of string values for categorical/binary fields, omit for numeric

5. For each split, build an N×F float32 matrix from `extract_features`, save as:
   - `{{output_dir}}/{{task}}/{{split}}.npz` with numpy keys:
     - `embeddings`: shape (N, F) float32
     - `labels`: shape (N,) int32
     - `patient_ids`: shape (N,) int64
     - `prediction_times`: shape (N,) object (strings)

6. Save `{{output_dir}}/{{task}}/feature_schema.json` once (after processing the first split):
   ```json
   {{
     "task": "{task}",
     "task_query": "{task_query}",
     "num_features": <F>,
     "features": <SCHEMA list>
   }}
   ```

7. Create output directories as needed. Print progress to stdout.

## Constraints

- Use only Python standard library plus `re`, `json`, `numpy`, `argparse`, `pathlib`, `sys`. No third-party packages beyond numpy.
- No LLM API calls, no network requests.
- `extract_features` must be deterministic and never raise exceptions on any input (catch all errors, default to 0.0).
- The script must be syntactically valid Python 3.8+.
- Do NOT hardcode file paths — use the argparse arguments.
- Aim for **at least 30 features** to capture the richness of the rubric. Include all numeric fields, all categorical fields (one-hot), and Yes/No procedure/comorbidity flags.

## Output

Output ONLY the Python script, with no explanation, no preamble, and no markdown fences. Start with `#!/usr/bin/env python3`."""


def generate_featurizer(task: str, task_query: str, parser_source: str,
                        examples: List[dict],
                        config: AzureConfig) -> Dict[str, Any]:
    client = AzureOpenAI(
        api_version=config.api_version,
        azure_endpoint=config.endpoint,
        api_key=config.api_key,
    )
    prompt = _build_prompt(task, task_query, parser_source, examples)
    logger.info(f"  prompt length: {len(prompt):,} chars")

    resp = client.chat.completions.create(
        model=config.deployment,
        messages=[
            {"role": "system",
             "content": ("You are an expert Python developer and medical informaticist. "
                         "You write clean, robust Python scripts for clinical data processing.")},
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
    code = code.strip()
    code = re.sub(r'^```(?:python)?\s*\n', '', code)
    code = re.sub(r'\n```\s*$', '', code)
    return code.strip()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parsers_dir", required=True,
                   help="Directory with {task}_parser.py files")
    p.add_argument("--cohort_dir", required=True,
                   help="Root cohort directory (data/rubric) containing {task}/cohort.json")
    p.add_argument("--sft_dir", required=True,
                   help="Root SFT directory (data/sft/llmrubric-auto)")
    p.add_argument("--output_dir", required=True,
                   help="Where to write generated featurizer scripts")
    p.add_argument("--azure_config", default=None)
    p.add_argument("--tasks", nargs="+", default=ALL_TASK_NAMES)
    return p.parse_args()


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def main():
    args = parse_args()
    config = AzureConfig.from_json(args.azure_config)
    config.max_completion_tokens = 32768
    config.temperature = 1.0

    parsers_dir = _resolve(args.parsers_dir)
    cohort_dir  = _resolve(args.cohort_dir)
    sft_dir     = _resolve(args.sft_dir)
    output_dir  = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for task in args.tasks:
        logger.info(f"\n{'='*60}\nGenerating featurizer for: {task}\n{'='*60}")

        parser_path = parsers_dir / f"{task}_parser.py"
        if not parser_path.exists():
            logger.warning(f"  parser not found at {parser_path}, skipping")
            continue

        cohort_path = cohort_dir / task / "cohort.json"
        if not cohort_path.exists():
            logger.warning(f"  cohort not found at {cohort_path}, skipping")
            continue

        sft_train_path = sft_dir / "train" / f"{task}.json"
        if not sft_train_path.exists():
            logger.warning(f"  SFT train split not found at {sft_train_path}, skipping")
            continue

        parser_source = parser_path.read_text()
        examples = _load_cohort_examples(cohort_path, sft_train_path)
        task_query = TASKS.get(task, "")
        if not task_query and examples:
            task_query = examples[0].get("task_query", "")

        logger.info(f"  loaded {len(examples)} cohort examples "
                    f"({sum(1 for e in examples if e.get('label_value'))} pos, "
                    f"{sum(1 for e in examples if not e.get('label_value'))} neg)")

        result = generate_featurizer(
            task, task_query, parser_source, examples, config)

        code = _strip_fences(result["code"])

        out_path = output_dir / f"{task}_featurizer.py"
        with open(out_path, "w") as f:
            f.write(code)
            if not code.endswith("\n"):
                f.write("\n")

        logger.success(
            f"  saved -> {out_path}  "
            f"(prompt={result['usage']['prompt_tokens']} tokens, "
            f"completion={result['usage']['completion_tokens']} tokens)"
        )

    logger.success("Done. All featurizer scripts generated.")


if __name__ == "__main__":
    main()
