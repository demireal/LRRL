"""
Central task registry for all 15 clinical prediction tasks.

Each task maps to a natural-language query used in prompts.
All tasks use Yes/No output format.

Naming conventions:
  - guo_*   : Operational outcome tasks (ICU, length-of-stay, readmission)
  - lab_*   : Lab result abnormality tasks
  - new_*   : New diagnosis tasks (will the patient develop X in the next year?)
  - chexpert: Chest X-ray findings (binary abnormal / normal)
"""

# Task name -> prediction query
TASKS = {
    # Guo operational tasks
    "guo_icu": "Will the patient be transferred to the intensive care unit?",
    "guo_los": "Will the patient stay in the hospital for more than 7 days?",
    "guo_readmission": "Will the patient be readmitted to the hospital within 30 days?",
    # Lab result tasks
    "lab_thrombocytopenia": "Will the patient's thrombocytopenia lab come back as abnormal?",
    "lab_hyperkalemia": "Will the patient's hyperkalemia lab come back as abnormal?",
    "lab_hypoglycemia": "Will the patient's hypoglycemia lab come back as abnormal?",
    "lab_hyponatremia": "Will the patient's hyponatremia lab come back as abnormal?",
    "lab_anemia": "Will the patient's anemia lab come back as abnormal?",
    # New diagnosis tasks
    "new_hypertension": "Will the patient develop hypertension in the next year?",
    "new_hyperlipidemia": "Will the patient develop hyperlipidemia in the next year?",
    "new_pancan": "Will the patient develop pancreatic cancer in the next year?",
    "new_celiac": "Will the patient develop celiac disease in the next year?",
    "new_lupus": "Will the patient develop lupus in the next year?",
    "new_acutemi": "Will the patient develop an acute myocardial infarction in the next year?",
    # Imaging task
    "chexpert": "Will the patient's chest X-ray come back as abnormal?",
}

ALL_TASK_NAMES = list(TASKS.keys())

# System message used across all SFT datasets
SYSTEM_MESSAGE = "You are a medical expert specializing in clinical risk prediction."

# Tokenizer used for clipping serializations
TOKENIZER_NAME = "Qwen/Qwen3-Embedding-8B"

# Max tokens for EHR serialization clipping
MAX_SERIALIZATION_TOKENS = 8192

# Embedding model
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"

# Fine-tuning base model
FINETUNE_MODEL = "Qwen/Qwen3-8B"

# Global random seed for reproducibility
SEED = 42
