# 03_globalrubric/

Build LLM-generated rubrics for structured EHR evaluation.

## Pipeline

```
build_cohort.py -> create_rubric.py -> apply_rubric.py -> create_globalrubric_sft.py
```

1. **build_cohort.py**: Embed training patients with Qwen3-Embedding-8B, run label-stratified k-means (k=20 per class), select medoids -> 40 patients per task.
2. **create_rubric.py**: Prompt GPT-5-mini with the 40 cohort examples to generate a step-by-step rubric template.
3. **apply_rubric.py**: Apply the rubric to all patients' naivetext EHRs via GPT-5-mini (parallel, resumable).
4. **create_globalrubric_sft.py**: Wrap rubricified text into SFT conversation format.

## GPT-5-mini Parameters

- `max_completion_tokens=16384`
- `temperature=1`

## Inputs

- Naivetext serialized data (`data/serialized/naivetext`)

## Outputs

```
data/rubric/{task}/cohort.json          # 40 cohort records (patient_id + prediction_time)
data/rubric/{task}/rubric.json          # rubric instructions
data/rubric/rubricified/{task}/{split}.json  # rubricified patient records
data/sft/global-rubric/{split}/{task}.json  # SFT datasets
```

## Ablation: No Examples

To run a rubric ablation (no 40-patient examples shown to GPT):

```bash
bash 03_globalrubric/run_ablation.sh
```

This uses `create_rubric_ablation.py` to generate rubrics from medical knowledge alone (task name + query, no cohort examples), then applies them identically. Outputs:

- `data/global-rubric-ablation/{task}/rubric.json`
- `data/global-rubric-ablation/rubricified/{task}/{split}.json`
- `data/sft/global-rubric-ablation/{split}/{task}.json`

## Auto Rubric Pipeline (Deterministic Parsers)

In addition to the LLM-based rubric application above, this directory also supports **auto rubric** generation — using GPT-5.2 to generate deterministic Python parser scripts that fill rubrics without per-patient LLM calls.

### Scripts

| Script | Purpose |
|--------|---------|
| `create_rubric_auto.py` | Use GPT-5.2 to generate `{task}_parser.py` scripts that parse EHR text with regex (no per-patient LLM calls) |
| `create_rubric_auto_plus.py` | Enhanced variant — shows GPT-5.2 both raw EHR + LLM-produced rubric fills as training signal |
| `create_rubric_schema.py` | Use GPT-5-mini to derive a typed JSON schema from rubric instructions (for downstream vectorization) |
| `create_feature_extractor.py` | Use GPT-5.2 to generate `{task}_featurizer.py` scripts that convert rubric text → fixed-dimension numeric vectors |

### Auto Rubric Flow

```
build_cohort.py -> create_rubric.py -> create_rubric_auto.py -> auto_parsers/{task}_parser.py
                                                                       |
                                                               create_globalrubric_sft.py
                                                                       |
                                                        create_feature_extractor.py -> feature_extractors/{task}_featurizer.py
                                                                                              |
                                                                              05_eval/eval_rubric_features.py
```

### Running the Auto Pipeline

```bash
# Generate auto parsers, apply them, and create SFT datasets
bash 03_globalrubric/run_auto.sh

# For the "plus" variant (augmented with LLM rubric fills):
VARIANT=auto_plus bash 03_globalrubric/run_auto.sh

# Run featurizers and evaluate with LR + XGBoost
bash 05_eval/run_rubric_features.sh
```

### Outputs

```
03_globalrubric/auto_parsers/{task}_parser.py          # Generated deterministic parsers
03_globalrubric/auto_parsers_plus/{task}_parser.py     # Plus-variant parsers
03_globalrubric/feature_extractors/{task}_featurizer.py # Generated featurizers
data/llmrubric-auto/rubricified/{task}/{split}.json    # Auto-rubricified patient records
data/sft/llmrubric-auto/{split}/{task}.json            # SFT datasets from auto rubrics
data/rubric_features_auto/{task}/{split}.npz           # Tabular feature vectors
data/rubric/{task}/rubric_schema.json                  # Typed rubric schema
```

## Next Step

- `05_eval/` evaluates embeddings from `data/sft/global-rubric/`.
- `05_eval/run_rubric_features.sh` evaluates tabular rubric features with LR + XGBoost.
- `cohort.json` is used by `--n_train 40` experiments (filtering by patient_id + prediction_time tuples).
