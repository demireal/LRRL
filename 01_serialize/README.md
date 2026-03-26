Adaptad from https://github.com/stefanhgm/ehrshot-benchmark

# 01_serialize/

Serialize patient EHRs from the FEMR database into structured Markdown text.

## Scripts

| Script | Purpose |
|--------|---------|
| `ehr_serializer.py` | Core EHR-to-Markdown converter. Modified from original: **"Past Medical Visits" summary section removed** to save tokens. |
| `serialize.py` | CLI driver: loads labels, applies cohort balancing, serializes, clips at 8192 tokens. |
| `run.sh` | Runs `naivetext` mode for all 15 tasks. Uses `--skip_existing` to avoid re-serializing tasks that already have JSONs. |

## Cohort Balancing

| Split | Rule |
|-------|------|
| val | `min(50, smallest_class)` per class (pos/neg) |
| train/test | If total > 3000: match to min(pos, 1000) balanced; otherwise keep all |

## Token Clipping

All serializations clipped to **8192 tokens** using `Qwen/Qwen3-Embedding-8B` tokenizer.

## Inputs

- FEMR patient database
- Labels CSV (`all_labels_tasks_out.csv`)
- Splits CSV (`person_id_map.csv`)

## Outputs

```
data/serialized/naivetext/{task}/{split}.json
```

Each JSON is a list of `{patient_id, prediction_time, task, split, label, serialization, original_tokens, was_clipped}`.

## Next Step

`02_create_sft/` consumes these JSON files.
