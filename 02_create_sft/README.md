# 02_create_sft/

Convert serialized patient records into SFT (Supervised Fine-Tuning) conversation format.

## Scripts

| Script | Purpose |
|--------|---------|
| `create_sft.py` | Reads serialized JSONs, wraps each in a sandwich-style prompt, outputs SFT conversations. |
| `run.sh` | Creates SFT datasets for `naivetext`. |

## Prompt Format (Sandwich Style)

```
System: You are a medical expert specializing in clinical risk prediction.
User: Based on the patient's EHR below, predict: {task_query}

      --- Patient EHR ---
      {serialization}
      --- End of EHR ---

      Based on the above EHR, predict: {task_query}
      Respond with exactly one word: Yes or No.
Assistant: Yes/No
```

The task query appears **before** and **after** the EHR to keep it fresh in context.

## Inputs

```
data/serialized/naivetext/{task}/{split}.json
```

## Outputs

```
data/sft/naivetext/{split}/{task}.json
```
