"""Configuration loading and merging."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (in-place) and return *base*."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> dict:
    """Load a YAML config file and optionally merge CLI overrides.

    Parameters
    ----------
    path : str | Path
        Path to the YAML config file.
    overrides : dict, optional
        Flat dict of dot-separated keys, e.g. ``{"es.sigma": 0.02}``.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    # Load and merge defaults first
    defaults_path = path.parent / "default.yaml"
    if defaults_path.exists() and defaults_path.resolve() != path.resolve():
        with defaults_path.open("r", encoding="utf-8") as fh:
            defaults = yaml.safe_load(fh) or {}
        cfg = _deep_merge(copy.deepcopy(defaults), cfg)

    # Apply CLI overrides (dot-notation)
    if overrides:
        for dotkey, value in overrides.items():
            keys = dotkey.split(".")
            d = cfg
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value

    return cfg
