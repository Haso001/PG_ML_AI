"""CMA-ES wrapper using the ``cmaes`` or ``cma`` Python package.

CMA-ES (Covariance Matrix Adaptation) is a state-of-the-art derivative-free
optimizer that maintains a full covariance model. It is well-suited for
*small* parameter counts (< ~10k elements). For larger layers, prefer
``OpenAIES`` with low-rank perturbations.

This module provides a thin adapter so it plugs into the same ask/tell
interface.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from .base import BaseES, ESResult

logger = logging.getLogger("es_llm")


class CMA_ES(BaseES):
    """CMA-ES adapter (requires ``pip install cmaes``).

    Parameters
    ----------
    population_size : int
        λ – number of candidates per generation.
    sigma : float
        Initial step-size σ₀.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        population_size: int = 20,
        sigma: float = 0.01,
        seed: int = 42,
        **kwargs,
    ):
        self.population_size = population_size
        self.sigma0 = sigma
        self.seed = seed
        self._optimizer = None      # lazily created on first ask()
        self._solutions: list[np.ndarray] = []

    def _ensure_optimizer(self, dim: int):
        if self._optimizer is not None:
            return
        try:
            from cmaes import CMA
        except ImportError:
            raise ImportError(
                "CMA-ES requires the 'cmaes' package. Install via: pip install cmaes"
            )
        self._optimizer = CMA(
            mean=np.zeros(dim),
            sigma=self.sigma0,
            population_size=self.population_size,
            seed=self.seed,
        )
        logger.info("CMA-ES initialized (dim=%d, σ₀=%.4f, λ=%d)", dim, self.sigma0, self.population_size)

    # ── ask ────────────────────────────────────────────────────────
    def ask(self, current_params: torch.Tensor) -> list[torch.Tensor]:
        dim = current_params.numel()
        device = current_params.device
        self._ensure_optimizer(dim)

        self._solutions = []
        candidates = []
        for _ in range(self.population_size):
            x = self._optimizer.ask()
            self._solutions.append(x)
            # CMA-ES solutions are offsets from the internal mean,
            # but we add them to current_params for evaluation
            cand = current_params + torch.from_numpy(x).float().to(device)
            candidates.append(cand)
        return candidates

    # ── tell ───────────────────────────────────────────────────────
    def tell(
        self,
        current_params: torch.Tensor,
        candidates: list[torch.Tensor],
        fitnesses: list[float],
    ) -> ESResult:
        # CMA-ES minimizes, so we negate fitness (we want to maximize)
        solutions_with_values = [
            (sol, -fit) for sol, fit in zip(self._solutions, fitnesses)
        ]
        self._optimizer.tell(solutions_with_values)

        # Reconstruct new params from CMA-ES internal mean
        new_mean = torch.from_numpy(self._optimizer._mean).float().to(current_params.device)
        new_params = current_params + new_mean

        best_idx = int(np.argmax(fitnesses))
        return ESResult(
            new_params=new_params,
            best_fitness=fitnesses[best_idx],
            mean_fitness=float(np.mean(fitnesses)),
            std_fitness=float(np.std(fitnesses)),
            best_perturbation=candidates[best_idx] - current_params,
        )
