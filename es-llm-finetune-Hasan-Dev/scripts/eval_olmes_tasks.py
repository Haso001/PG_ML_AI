#!/usr/bin/env python3
"""Run OLMES evaluation for one or more tasks.

Primary use-case:
  evaluate a fine-tuned checkpoint (run_dir/model) on tasks such as
  ``minerva_math_algebra`` and ``hellaswag``.

Examples
--------
python scripts/eval_olmes_tasks.py \
  --run experiments/runs/gsm8k_last_layer_es \
  --tasks minerva_math_algebra hellaswag \
  --limit 200 \
  --num_shots 0

python scripts/eval_olmes_tasks.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --tasks hellaswag \
  --output_dir out/hellaswag_baseline
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


def _resolve_olmes_cmd() -> Tuple[List[str], str]:
    """Resolve the evaluator command.

    Priority:
    1) OLMES_BIN env var (explicit override)
    2) `olmes` on PATH
    3) `oe-eval` on PATH
    """
    explicit = (os.environ.get("OLMES_BIN") or "").strip()
    if explicit:
        return [explicit], explicit

    for candidate in ["olmes", "oe-eval"]:
        found = shutil.which(candidate)
        if found:
            return [candidate], candidate

    return [], ""


def _list_olmes_tasks() -> Tuple[List[str], str]:
    """Return available OLMES tasks and raw command output."""
    cmd_prefix, cmd_name = _resolve_olmes_cmd()
    if not cmd_prefix:
        return [], "No evaluator CLI found. Tried: olmes, oe-eval"

    result = subprocess.run(
        cmd_prefix + ["--list-tasks"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    combined = "\n".join([result.stdout or "", result.stderr or ""]).strip()

    tasks: List[str] = []
    for line in combined.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[") or stripped.startswith("All "):
            continue
        if stripped[:4].isdigit() and "-" in stripped:
            # Ignore timestamp-like logging lines.
            continue
        task_name = stripped.split(" ", 1)[0].strip()
        if task_name:
            tasks.append(task_name)

    seen = set()
    deduped = []
    for t in tasks:
        if t not in seen:
            deduped.append(t)
            seen.add(t)
    return deduped, combined


def _resolve_tasks(requested: List[str], available: List[str]) -> Tuple[List[str], List[str]]:
    """Resolve requested task names with light fuzzy matching.

    Returns
    -------
    resolved : list[str]
        Resolved task names to pass to OLMES.
    missing : list[str]
        Requested tasks that could not be resolved.
    """
    resolved: List[str] = []
    missing: List[str] = []

    lowered = {t.lower(): t for t in available}
    for raw in requested:
        q = raw.strip()
        if not q:
            continue
        if q in available:
            resolved.append(q)
            continue
        if q.lower() in lowered:
            resolved.append(lowered[q.lower()])
            continue

        # Fuzzy fallback: unique contains-match
        candidates = [t for t in available if q.lower() in t.lower()]
        if len(candidates) == 1:
            resolved.append(candidates[0])
        else:
            missing.append(q)

    # Preserve order but remove duplicates.
    seen = set()
    unique_resolved = []
    for t in resolved:
        if t not in seen:
            unique_resolved.append(t)
            seen.add(t)
    return unique_resolved, missing


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate model checkpoints with OLMES on multiple tasks")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", type=str, help="Run directory containing a saved model/ folder")
    group.add_argument("--model", type=str, help="Model id/path passed directly to OLMES")

    ap.add_argument(
        "--tasks",
        nargs="+",
        required=True,
        help="Tasks to evaluate, e.g. minerva_math_algebra hellaswag",
    )
    ap.add_argument("--model_type", default="hf", choices=["hf", "vllm"])
    ap.add_argument("--limit", default="200", help="Instance limit (int) or fraction (e.g. 0.1)")
    ap.add_argument("--num_shots", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--cache_dir", type=str, default="cache")
    ap.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    ap.add_argument("--extra_args", type=str, default="", help="Extra raw args appended to OLMES")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    if args.run:
        run_dir = Path(args.run)
        model_path = run_dir / "model"
        if not model_path.exists():
            print(f"No model directory found: {model_path}", file=sys.stderr)
            return 1
        model = str(model_path)
        run_name = run_dir.name
    else:
        model = args.model
        run_name = Path(model).name.replace("/", "_")

    out_dir = Path(args.output_dir) if args.output_dir else Path("out") / f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    available_tasks, raw_list_output = _list_olmes_tasks()
    if not available_tasks:
        print("Failed to read task list from OLMES. Output:", file=sys.stderr)
        print(raw_list_output or "<empty>", file=sys.stderr)
        return 1

    cmd_prefix, cmd_name = _resolve_olmes_cmd()
    if not cmd_prefix:
        print("No evaluator CLI available (expected `olmes` or `oe-eval`).", file=sys.stderr)
        return 1

    resolved_tasks, missing = _resolve_tasks(args.tasks, available_tasks)
    if missing:
        print("Some tasks could not be resolved:", ", ".join(missing), file=sys.stderr)
        print("Tip: run `olmes --list-tasks` and use exact names.", file=sys.stderr)
        return 1

    cmd = cmd_prefix + [
        "--model", model,
        "--model-type", args.model_type,
        "--limit", str(args.limit),
        "--num-shots", str(args.num_shots),
        "--batch-size", str(args.batch_size),
        "--output-dir", str(out_dir),
        "--cached-output-dir", args.cache_dir,
    ]
    # OLMES expects one --task flag per task; a comma-joined value is treated as a single unknown task.
    for task_name in resolved_tasks:
        cmd.extend(["--task", task_name])
    if args.extra_args.strip():
        cmd.extend(shlex.split(args.extra_args.strip()))

    print("Running:")
    print(f"Evaluator CLI: {cmd_name}")
    print(" ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(cmd, check=False)

    summary = {
        "model": model,
        "tasks_requested": args.tasks,
        "tasks_resolved": resolved_tasks,
        "limit": args.limit,
        "num_shots": args.num_shots,
        "batch_size": args.batch_size,
        "output_dir": str(out_dir),
        "returncode": proc.returncode,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
