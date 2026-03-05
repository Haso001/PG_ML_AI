#!/usr/bin/env python3
"""Inspect the architecture and layers of a HuggingFace model.

Useful for deciding which layers/parameters to target with ES.

Usage
─────
    python scripts/inspect_layers.py --model Qwen/Qwen2.5-0.5B-Instruct
    python scripts/inspect_layers.py --model Qwen/Qwen2.5-0.5B-Instruct --filter "layers.23"
    python scripts/inspect_layers.py --model Qwen/Qwen2.5-0.5B-Instruct --top 20
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from es_llm.model.loader import load_model_and_tokenizer, inspect_model_layers


def main():
    ap = argparse.ArgumentParser(description="Inspect model layers")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--filter", default=None, help="Regex to filter parameter names")
    ap.add_argument("--top", type=int, default=None, help="Show top N largest parameters")
    ap.add_argument("--summary", action="store_true", help="Only print per-layer summary")
    args = ap.parse_args()

    model, _ = load_model_and_tokenizer(args.model, dtype=args.dtype, device=args.device)
    layers = inspect_model_layers(model)

    # Filter
    if args.filter:
        pat = re.compile(args.filter)
        layers = [l for l in layers if pat.search(l["name"])]

    # Sort by size (largest first) if --top is used
    if args.top:
        layers = sorted(layers, key=lambda x: x["numel"], reverse=True)[:args.top]

    # ── Summary mode ──────────────────────────────────────────
    if args.summary:
        # Group by decoder-layer index
        from collections import defaultdict
        per_layer: dict[str, int] = defaultdict(int)
        for l in layers:
            parts = l["name"].split(".")
            # Find "layers.N" prefix
            key = "other"
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    key = f"layers.{parts[i + 1]}"
                    break
            per_layer[key] += l["numel"]

        print(f"\n{'Layer':<20s} {'Parameters':>15s}")
        print("─" * 37)
        total = 0
        for key in sorted(per_layer, key=lambda k: (not k.startswith("layers"), k)):
            count = per_layer[key]
            total += count
            print(f"{key:<20s} {count:>15,}")
        print("─" * 37)
        print(f"{'TOTAL':<20s} {total:>15,}")
        return

    # ── Full listing ──────────────────────────────────────────
    print(f"\n{'Parameter Name':<60s} {'Shape':<25s} {'Elements':>12s} {'Dtype'}")
    print("─" * 110)
    total = 0
    for l in layers:
        total += l["numel"]
        shape_str = str(l["shape"])
        print(f"{l['name']:<60s} {shape_str:<25s} {l['numel']:>12,}  {l['dtype']}")
    print("─" * 110)
    print(f"{'TOTAL':<60s} {'':25s} {total:>12,}")
    print(f"\nTotal parameters: {total:,}")


if __name__ == "__main__":
    main()
