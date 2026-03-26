# config/

Central configuration shared by all pipeline stages.

## Files

| File | Purpose |
|------|---------|
| `tasks.py` | 15 task definitions (name -> query), system message, tokenizer/model names |
| `azure.py` | Azure OpenAI config loader (`AzureConfig` dataclass) |
| `azure_config.json` | Credentials file (**not committed**) |

## Usage

```python
from config.tasks import TASKS, ALL_TASK_NAMES, SYSTEM_MESSAGE
from config.azure import AzureConfig

config = AzureConfig.from_json()  # loads azure_config.json
```
