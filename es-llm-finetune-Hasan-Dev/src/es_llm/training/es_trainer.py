"""Main ES training loop.

Orchestrates:
1. Model loading
2. Layer selection
3. ES ask/tell loop with fitness evaluation
4. Checkpointing and logging
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

from tqdm.auto import tqdm

from ..es.base import BaseES
from ..es.openai_es import OpenAIES
from ..es.cma_es import CMA_ES
from ..fitness.base import BaseFitness
from ..fitness.gsm8k import GSM8KFitness
from ..fitness.gsm8k_loglikelihood import GSM8KLogLikelihoodFitness
from ..model.layer_selector import LayerSelector
from ..model.loader import load_model_and_tokenizer
from ..utils.logging import log_generation, save_run_log, setup_logger

logger = logging.getLogger("es_llm")


def _build_es(cfg: dict) -> BaseES:
    """Instantiate the ES algorithm from config."""
    es_cfg = cfg["es"]
    algo = es_cfg["algorithm"]

    if algo == "openai_es":
        return OpenAIES(
            population_size=es_cfg["population_size"],
            sigma=es_cfg["sigma"],
            learning_rate=es_cfg["learning_rate"],
            antithetic=es_cfg.get("antithetic", True),
            fitness_shaping=es_cfg.get("fitness_shaping", "centered_rank"),
            weight_decay=es_cfg.get("weight_decay", 0.0),
            seed=es_cfg.get("seed", 42),
        )
    elif algo == "cma_es":
        return CMA_ES(
            population_size=es_cfg["population_size"],
            sigma=es_cfg["sigma"],
            seed=es_cfg.get("seed", 42),
        )
    else:
        raise ValueError(f"Unknown ES algorithm: {algo}")


def _build_fitness(cfg: dict) -> BaseFitness:
    """Instantiate the fitness evaluator from config."""
    fit_cfg = cfg["fitness"]
    task = fit_cfg["task"]

    if task == "gsm8k":
        return GSM8KFitness(
            num_samples=fit_cfg.get("num_samples", 50),
            split=fit_cfg.get("split", "train"),
            max_new_tokens=fit_cfg.get("max_new_tokens", 256),
            seed=cfg["es"].get("seed", 42),
        )
    elif task == "gsm8k_loglikelihood":
        return GSM8KLogLikelihoodFitness(
            num_samples=fit_cfg.get("num_samples", 50),
            split=fit_cfg.get("split", "train"),
            target_mode=fit_cfg.get("target_mode", "short"),
            seed=cfg["es"].get("seed", 42),
        )
    else:
        raise ValueError(f"Unknown fitness task: {task}")


def _make_run_dir(cfg: dict) -> Path:
    """Create the run output directory."""
    base = Path(cfg["output"]["dir"])
    run_name = cfg["output"].get("run_name")
    if not run_name:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def train(cfg: dict) -> Path:
    """Run the full ES training loop.

    Parameters
    ----------
    cfg : dict
        Merged configuration (see configs/default.yaml).

    Returns
    -------
    Path
        Path to the run directory containing checkpoints and logs.
    """
    setup_logger()
    run_dir = _make_run_dir(cfg)
    logger.info("Run directory: %s", run_dir)

    # Save config
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    # ── 1) Load model ────────────────────────────────────────────
    model_cfg = cfg["model"]
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_cfg["name"],
        revision=model_cfg.get("revision", "main"),
        dtype=model_cfg.get("dtype", "float16"),
        device=model_cfg.get("device", "cuda"),
    )

    # ── 2) Select layers ─────────────────────────────────────────
    layer_cfg = cfg["layer_selection"]
    selector = LayerSelector(
        model=model,
        strategy=layer_cfg["strategy"],
        layer_indices=layer_cfg.get("layer_indices"),
        layer_names=layer_cfg.get("layer_names"),
        layer_regex=layer_cfg.get("layer_regex"),
        components=layer_cfg.get("components", "all"),
    )

    # Log selection summary
    summary = selector.summary()
    logger.info("Layer selection:\n%s", summary)
    (run_dir / "layer_selection.txt").write_text(summary)

    # ── 3) Build ES algorithm ────────────────────────────────────
    es = _build_es(cfg)
    es_cfg = cfg["es"]

    # ── 4) Build fitness evaluator ───────────────────────────────
    fitness = _build_fitness(cfg)

    # ── 5) Baseline evaluation ───────────────────────────────────
    logger.info("Evaluating baseline fitness...")
    baseline_fitness = fitness.evaluate(model, tokenizer)
    logger.info("Baseline fitness (%s): %.4f", fitness.name(), baseline_fitness)

    # ── 6) ES training loop ──────────────────────────────────────
    current_params = selector.get_flat_params().clone()
    best_ever_fitness = baseline_fitness
    best_ever_params = current_params.clone()
    history = []
    save_every = cfg["output"].get("save_every", 10)
    log_every = cfg["output"].get("log_every", 1)

    num_generations = es_cfg["num_generations"]
    device = model_cfg.get("device", "cuda")
    is_cuda = device.startswith("cuda")

    if is_cuda:
        logger.info(
            "GPU: %s | VRAM: %.1f GB (%.1f GB used)",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
            torch.cuda.memory_allocated() / 1e9,
        )

    logger.info(
        "Starting ES training: %d generations, pop=%d, σ=%.4f, lr=%.5f",
        num_generations,
        es_cfg["population_size"],
        es_cfg["sigma"],
        es_cfg.get("learning_rate", 0.001),
    )

    for gen in tqdm(range(1, num_generations + 1), desc="Generationen", unit="gen"):
        t0 = time.time()

        # Ask: generate candidates
        candidates = es.ask(current_params)
        print(f"\n── Generation {gen}/{num_generations} ──", flush=True)

        # Evaluate each candidate
        fitnesses = []
        cand_bar = tqdm(enumerate(candidates), total=len(candidates),
                        desc=f"  Gen {gen:3d} Kandidaten", unit="cand", leave=False)
        for i, cand in cand_bar:
            selector.set_flat_params(cand)
            fit = fitness.evaluate(model, tokenizer)
            fitnesses.append(fit)
            cand_bar.set_postfix(fit=f"{fit:.3f}")
            print(f"  Kandidat {i+1}/{len(candidates)} → fitness={fit:.3f}", flush=True)
            # Free intermediate GPU cache between candidates
            if is_cuda:
                torch.cuda.empty_cache()

        # Tell: update parameters
        result = es.tell(current_params, candidates, fitnesses)
        current_params = result.new_params.clone()
        selector.set_flat_params(current_params)

        elapsed = time.time() - t0

        # Track best
        if result.best_fitness > best_ever_fitness:
            best_ever_fitness = result.best_fitness
            best_ever_params = current_params.clone()

        # History
        entry = {
            "generation": gen,
            "best_fitness": result.best_fitness,
            "mean_fitness": result.mean_fitness,
            "std_fitness": result.std_fitness,
            "best_ever_fitness": best_ever_fitness,
            "elapsed_sec": elapsed,
        }
        history.append(entry)

        # Log
        if gen % log_every == 0:
            log_generation(
                logger,
                gen,
                result.best_fitness,
                result.mean_fitness,
                result.std_fitness,
                extra={"best_ever": f"{best_ever_fitness:.4f}", "time": f"{elapsed:.1f}s"},
            )

        # Checkpoint
        if gen % save_every == 0 or gen == num_generations:
            ckpt_path = run_dir / f"checkpoint_gen{gen:04d}.pt"
            torch.save(
                {
                    "generation": gen,
                    "flat_params": current_params,
                    "best_ever_params": best_ever_params,
                    "best_ever_fitness": best_ever_fitness,
                    "target_names": selector.target_names,
                },
                ckpt_path,
            )
            logger.info("Saved checkpoint: %s", ckpt_path)

    # ── 7) Final: apply best-ever parameters ─────────────────────
    selector.set_flat_params(best_ever_params)
    logger.info("Applied best-ever parameters (fitness=%.4f)", best_ever_fitness)

    # Save final model state dict (only modified layers)
    final_state = {name: param.data.clone() for name, param in selector.get_target_params()}
    torch.save(final_state, run_dir / "best_layer_weights.pt")

    # Save full model
    model.save_pretrained(run_dir / "model")
    tokenizer.save_pretrained(run_dir / "model")
    logger.info("Saved final model to %s/model", run_dir)

    # Save training log
    save_run_log(run_dir, history)

    if device.startswith("cuda"):
        logger.info("Final VRAM: %.2f GB", torch.cuda.memory_allocated() / 1e9)

    logger.info("Training complete. Run dir: %s", run_dir)

    return run_dir
