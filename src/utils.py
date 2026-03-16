"""
UTILS: Shared helpers used by all 4 modules
============================================
Rule: if 2+ modules need the same function, it lives here.
"""

import json
import os
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load config.yaml and return as a Python dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_json(data, output_path: str) -> None:
    """Save any data to a JSON file (creates dirs automatically)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(input_path: str):
    """Load a JSON file and return parsed data."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)
