# LLM Rubric Representation Learning (LRRL)

This is the repository for the paper [LLMs can construct powerful representations and streamline sample-efficient supervised learning](https://arxiv.org/abs/2603.11679).

If you find this codebase useful, please cite:

```bibtex
@article{demirel2026llms,
  title={LLMs can construct powerful representations and streamline sample-efficient supervised learning},
  author={Demirel, Ilker and Shi, Lawrence and Hussain, Zeshan and Sontag, David},
  journal={arXiv preprint arXiv:2603.11679},
  year={2026}
}
```

## Pipeline Overview

```
01_serialize  ->  02_create_sft  ->  05_eval
                       |                |
               03_globalrubric    generate_embeddings
               (GPT-5-mini)       + eval_embeddings
                       |
               04_localrubric     Auto rubric path:
               (GPT-5-mini)       create_rubric_auto -> auto_parsers
                                  create_feature_extractor -> feature_extractors
                                         |
                                  eval_rubric_features (LR + XGBoost)
```

### Execution Order

```bash
# Step 1: Serialize EHRs
bash 01_serialize/run.sh

# Step 2: Create SFT datasets
bash 02_create_sft/run.sh

# Step 3: Build global rubrics + rubricified representations (requires GPT-5-mini + GPU for embedding generation)
bash 03_globalrubric/run.sh

# Step 3b (optional): Generate auto rubric parsers + apply deterministically (requires GPT-5.2)
bash 03_globalrubric/run_auto.sh

# Step 4: Generate local rubric representations (requires GPT-5-mini)
bash 04_localrubric/run.sh

# Step 5: Evaluate everything (embeddings + LogReg)
bash 05_eval/run.sh

# Step 5b (optional): Evaluate rubric tabular features with LR + XGBoost
bash 05_eval/run_rubric_features.sh
```

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU(s)
- Access to Azure OpenAI (for steps 3-4)
- EHRSHOT dataset (FEMR extract + benchmark labels + splits) -- acquire access at https://stanford.redivis.com/datasets/53gc-8rhx41kgt

### Environment Variables

Set these before running `01_serialize/run.sh`:

```bash
export EHRSHOT_DB=/path/to/EHRSHOT_ASSETS/femr/extract
export EHRSHOT_LABELS=/path/to/EHRSHOT_ASSETS/benchmark
export EHRSHOT_SPLITS=/path/to/EHRSHOT_ASSETS/splits/person_id_map.csv
```

### Azure OpenAI

Copy the template and fill in your credentials:

```bash
cp config/azure_config.json.template config/azure_config.json
# Edit config/azure_config.json with your endpoint and API key
```

## Directory Structure

```
lrrl/
├── README.md
├── .gitignore
├── config/
│   ├── tasks.py                      # 15 task definitions + model names
│   ├── azure.py                      # Azure OpenAI config loader
│   └── azure_config.json.template    # Credentials template (not committed)
│
├── 01_serialize/
│   ├── serialize.py                  # EHR -> Markdown text
│   ├── ehr_serializer.py             # Core serializer
│   └── run.sh
│
├── 02_create_sft/
│   ├── create_sft.py                 # Serialized data -> SFT conversations
│   └── run.sh
│
├── 03_globalrubric/
│   ├── build_cohort.py               # K-means + medoid selection (40 patients)
│   ├── create_rubric.py              # GPT-5-mini rubric generation
│   ├── apply_rubric.py               # Apply rubric to all patients
│   ├── create_globalrubric_sft.py    # Rubricified -> SFT format
│   ├── create_rubric_auto.py         # GPT-5.2 generates deterministic parsers
│   ├── create_rubric_auto_plus.py    # Enhanced: parsers with LLM rubric examples
│   ├── create_rubric_schema.py       # GPT-5-mini derives typed rubric schema
│   ├── create_feature_extractor.py   # GPT-5.2 generates tabular featurizers
│   ├── run.sh                        # LLM rubric pipeline
│   ├── run_auto.sh                   # Auto rubric pipeline
│   ├── auto_parsers/                 # Generated deterministic parsers (gitignored)
│   ├── auto_parsers_plus/            # Plus-variant parsers (gitignored)
│   └── feature_extractors/           # Generated featurizers (gitignored)
│
├── 04_localrubric/
│   ├── generate_local_rubric.py      # Local rubric generation (train+val+test)
│   └── run.sh
│
├── 05_eval/
│   ├── generate_embeddings.py        # Qwen3-Embedding-8B embeddings
│   ├── eval_embeddings.py            # LogReg with val-based C selection
│   ├── eval_rubric_features.py       # LR + XGBoost on tabular rubric features
│   ├── compute_metrics.py            # AUROC/AUPRC + bootstrap CIs
│   ├── run.sh                        # Orchestrator
│   ├── run_embeddings.sh             # Embedding generation + LogReg evaluation
│   └── run_rubric_features.sh        # Featurizer + LR/XGBoost evaluation
│
├── 06_baselines/
│   ├── clmbrt/                       # CLMBR-T baseline
│   └── count-gbm/                    # Count-GBM baseline
│
└── data/                             # All generated outputs (gitignored)
```

## Tasks (15)

| Category | Task | Query |
|----------|------|-------|
| Operational | `guo_icu` | Will the patient be transferred to the ICU? |
| Operational | `guo_los` | Will the patient stay > 7 days? |
| Operational | `guo_readmission` | Will the patient be readmitted within 30 days? |
| Lab | `lab_thrombocytopenia` | Will the thrombocytopenia lab come back as abnormal? |
| Lab | `lab_hyperkalemia` | Will the hyperkalemia lab come back as abnormal? |
| Lab | `lab_hypoglycemia` | Will the hypoglycemia lab come back as abnormal? |
| Lab | `lab_hyponatremia` | Will the hyponatremia lab come back as abnormal? |
| Lab | `lab_anemia` | Will the anemia lab come back as abnormal? |
| Diagnosis | `new_hypertension` | Will the patient develop hypertension in the next year? |
| Diagnosis | `new_hyperlipidemia` | Will the patient develop hyperlipidemia in the next year? |
| Diagnosis | `new_pancan` | Will the patient develop pancreatic cancer in the next year? |
| Diagnosis | `new_celiac` | Will the patient develop celiac disease in the next year? |
| Diagnosis | `new_lupus` | Will the patient develop lupus in the next year? |
| Diagnosis | `new_acutemi` | Will the patient develop an acute MI in the next year? |
| Imaging | `chexpert` | Does the patient have abnormal chest X-ray findings? |