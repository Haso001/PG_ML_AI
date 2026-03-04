"""Logging helpers for ES training."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def setup_logger(name: str = "es_llm", level: int = logging.INFO) -> logging.Logger:
    """Create a console + file logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured
    logger.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def log_generation(
    logger: logging.Logger,
    generation: int,
    best_fitness: float,
    mean_fitness: float,
    std_fitness: float,
    extra: Dict[str, Any] | None = None,
) -> None:
    """Log one generation summary."""
    msg = (
        f"Gen {generation:>4d} | "
        f"best={best_fitness:+.4f}  mean={mean_fitness:+.4f}  std={std_fitness:.4f}"
    )
    if extra:
        msg += " | " + " ".join(f"{k}={v}" for k, v in extra.items())
    logger.info(msg)


def save_run_log(log_dir: Path, history: list[dict]) -> Path:
    """Persist generation-level metrics as JSONL."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "training_log.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for entry in history:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path
