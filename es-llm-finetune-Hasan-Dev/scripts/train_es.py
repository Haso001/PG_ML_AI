#!/usr/bin/env python3
"""Entry point for ES layer-wise fine-tuning.

Usage
─────
    python scripts/train_es.py --config configs/gsm8k_last_layer.yaml

    # Override individual settings via CLI:
    python scripts/train_es.py \
        --config configs/default.yaml \
        --set es.sigma=0.02 \
        --set es.num_generations=200 \
        --set fitness.num_samples=30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from es_llm.utils.config import load_config
from es_llm.training.es_trainer import train


def parse_args():
    ap = argparse.ArgumentParser(
        description="Fine-tune individual LLM layers with Evolutionary Strategies"
    )
    ap.add_argument(
        "--config", "-c",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    ap.add_argument(
        "--set", "-s",
        nargs="*",
        default=[],
        dest="overrides",
        help="Override config values, e.g. --set es.sigma=0.02 fitness.num_samples=30",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    # Parse --set overrides into a dict
    overrides = {}
    for item in args.overrides:
        if "=" not in item:
            print(f"Invalid override (missing '='): {item}", file=sys.stderr)
            sys.exit(1)
        key, val = item.split("=", 1)
        # Auto-cast types
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                if val.lower() in ("true", "false"):
                    val = val.lower() == "true"
        overrides[key] = val

    cfg = load_config(args.config, overrides=overrides if overrides else None)
    train(cfg)


if __name__ == "__main__":
    main()
