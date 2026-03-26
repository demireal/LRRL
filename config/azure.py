"""
Azure OpenAI configuration loader.

Reads credentials from azure_config.json (expected at ehrshot-v2/config/azure_config.json
or the path passed via --azure_config CLI flag).  The JSON file should NOT be committed
to version control.

Expected JSON format:
{
    "endpoint": "https://...",
    "api_key": "...",
    "api_version": "2024-12-01-preview",
    "deployment": "gpt-5-mini"
}
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AzureConfig:
    """Configuration for Azure OpenAI API calls."""

    endpoint: str
    api_key: str
    api_version: str = "2024-12-01-preview"
    deployment: str = "gpt-5-mini"
    max_completion_tokens: int = 16384
    temperature: float = 1.0
    top_p: float = 1.0

    @classmethod
    def from_json(cls, config_path: Optional[str] = None) -> "AzureConfig":
        """Load config from a JSON file.

        If *config_path* is None the loader looks for
        ``ehrshot-v2/config/azure_config.json`` relative to this file.
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "azure_config.json"
            )

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Azure config not found at {config_path}. "
                "Create it from the template or pass --azure_config."
            )

        with open(config_path, "r") as f:
            data = json.load(f)

        return cls(
            endpoint=data["endpoint"],
            api_key=data["api_key"],
            api_version=data.get("api_version", "2024-12-01-preview"),
            deployment=data.get("deployment", "gpt-5-mini"),
        )
