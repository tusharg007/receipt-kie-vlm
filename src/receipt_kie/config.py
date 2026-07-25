"""YAML configuration loading with inheritance and deterministic path handling."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries without mutating either input."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Load YAML and recursively resolve an optional ``extends`` key."""
    config_path = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if config_path in seen:
        raise ValueError(f"Circular configuration inheritance detected at {config_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    seen.add(config_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Configuration root must be a mapping: {config_path}")
    parent_name = payload.pop("extends", None)
    if parent_name is None:
        return payload
    parent_path = config_path.parent / str(parent_name)
    return deep_merge(load_config(parent_path, seen), payload)


def get_required(config: dict[str, Any], dotted_key: str) -> Any:
    """Read a required nested configuration value."""
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Missing required configuration key: {dotted_key}")
        value = value[part]
    return value
