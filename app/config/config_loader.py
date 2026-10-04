"""
app/config/config_loader.py
---------------------------
Utility to read YAML configuration files and overlay values from ``.env``
environment variables. Uses ``ruamel.yaml`` for preserving order (already in
requirements). Returns a simple dict‑like object.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

# Helper to recursively merge two dicts (src into dst)
def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    for key, val in src.items():
        if isinstance(val, dict) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], val)
        else:
            dst[key] = val


def load_config(config_dir: str | Path = None) -> Dict[str, Any]:
    """Load all ``*.yaml`` files from *config_dir* (or the default
    ``D:\\gaming_video_agent\\config``) and merge them.

    Environment variables override any matching top‑level key. For nested
    overrides you can use ``APP__LOG_LEVEL`` style – the loader expands ``__``
    into dictionary nesting.
    """
    base_dir = Path(config_dir) if config_dir else Path(__file__).parents[2] / "config"
    config: Dict[str, Any] = {}
    # Load each yaml file alphabetically so that later files can override earlier
    for yaml_path in sorted(base_dir.glob("*.yaml")):
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            _deep_merge(config, data)

    # Apply environment overrides (simple flat keys first)
    for env_key, env_val in os.environ.items():
        if env_key.startswith("APP__"):
            # Nested path e.g. APP__LOG_LEVEL => config["app"]["log_level"]
            parts = env_key.split("__")[1:]
            cur = config
            for part in parts[:-1]:
                cur = cur.setdefault(part.lower(), {})
            cur[parts[-1].lower()] = env_val
        elif env_key.lower() in config:
            config[env_key.lower()] = env_val
    return config
