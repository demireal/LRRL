# 04_localrubric/

Generate local rubric representations using GPT-5-mini.

## Scripts

| Script | Purpose | Splits |
|--------|---------|--------|
| `generate_local_rubric.py` | Task-conditioned summary generation | train + val + test |

## GPT-5-mini Parameters

- `max_completion_tokens=16384`
- `temperature=1`

## Task Configuration

Accepts `--tasks` to select which tasks to generate for (default: all 15).

## Inputs

- Naivetext SFT datasets (`data/sft/naivetext`)

## Outputs

```
data/sft/local-rubric/{split}/{task}.json
```

## Usage Downstream

- `05_eval/generate_embeddings.py` (embed for LogReg)
