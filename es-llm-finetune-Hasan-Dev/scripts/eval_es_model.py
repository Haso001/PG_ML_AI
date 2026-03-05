#!/usr/bin/env python3
"""Load an ES checkpoint and evaluate the fine-tuned model.

Usage
─────
    python scripts/eval_es_model.py \
        --run experiments/runs/gsm8k_last_layer_es \
        --split test \
        --num_samples 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from es_llm.model.loader import load_model_and_tokenizer
from es_llm.model.layer_selector import LayerSelector
from es_llm.fitness.gsm8k import GSM8KFitness
from es_llm.utils.logging import setup_logger


def main():
    ap = argparse.ArgumentParser(description="Evaluate an ES-fine-tuned model")
    ap.add_argument("--run", required=True, help="Path to the run directory")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()

    setup_logger()
    run_dir = Path(args.run)

    # ── Option A: Load saved full model ──────────────────────
    model_dir = run_dir / "model"
    if model_dir.exists():
        print(f"Loading full model from {model_dir}")
        model, tokenizer = load_model_and_tokenizer(
            model_name=str(model_dir),
            dtype=args.dtype,
            device=args.device,
        )
    else:
        # ── Option B: Load base model + apply layer weights ──
        config_path = run_dir / "config.json"
        if not config_path.exists():
            print(f"No config.json found in {run_dir}", file=sys.stderr)
            sys.exit(1)

        cfg = json.loads(config_path.read_text())
        model_cfg = cfg["model"]
        model, tokenizer = load_model_and_tokenizer(
            model_name=model_cfg["name"],
            revision=model_cfg.get("revision", "main"),
            dtype=args.dtype,
            device=args.device,
        )

        # Apply saved layer weights
        weights_path = run_dir / "best_layer_weights.pt"
        if weights_path.exists():
            state = torch.load(weights_path, map_location=args.device, weights_only=True)
            model_dict = dict(model.named_parameters())
            for name, tensor in state.items():
                if name in model_dict:
                    model_dict[name].data.copy_(tensor)
            print(f"Loaded layer weights from {weights_path} ({len(state)} parameters)")

    # ── Evaluate ─────────────────────────────────────────────
    fitness = GSM8KFitness(
        num_samples=args.num_samples,
        split=args.split,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"\nEvaluating on GSM8K ({args.split}, n={args.num_samples})...")
    score = fitness.evaluate(model, tokenizer)

    result = {
        "run": str(run_dir),
        "task": "gsm8k",
        "split": args.split,
        "num_samples": args.num_samples,
        "accuracy": score,
    }
    print(json.dumps(result, indent=2))

    # Save result
    out_path = run_dir / f"eval_{args.split}_{args.num_samples}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
