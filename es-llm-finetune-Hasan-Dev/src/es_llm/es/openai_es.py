"""OpenAI-style Evolutionary Strategy (Salimans et al., 2017).

Core idea
─────────
Given current parameters θ and fitness function F:

1.  Sample N noise vectors:  ε_i ~ N(0, σ²I)
2.  Evaluate:                F_i = F(θ + ε_i)
3.  Update:                  θ ← θ + α / (Nσ) · Σ F_i · ε_i

With antithetic sampling we generate (ε, −ε) pairs to reduce variance.
Fitness shaping (centered rank) improves robustness to outliers.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from .base import BaseES, ESResult
from .noise import antithetic_pairs, gaussian_noise

logger = logging.getLogger("es_llm")


def _centered_rank_transform(fitnesses: list[float]) -> np.ndarray:
    """Rank-based fitness shaping (from OpenAI ES paper).

    Maps fitness values to utilities in [-0.5, 0.5] based on rank.
    """
    n = len(fitnesses)
    ranks = np.zeros(n)
    order = np.argsort(fitnesses)
    for rank_idx, idx in enumerate(order):
        ranks[idx] = rank_idx
    # Normalize to [-0.5, 0.5]
    utilities = (ranks / (n - 1)) - 0.5 if n > 1 else np.zeros(n)
    return utilities


def _normalize(fitnesses: list[float]) -> np.ndarray:
    """Zero-mean, unit-variance normalization."""
    arr = np.array(fitnesses, dtype=np.float64)
    std = arr.std()
    if std < 1e-8:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


class OpenAIES(BaseES):
    """OpenAI-style Evolution Strategy.

    Parameters
    ----------
    population_size : int
        Number of candidates per generation (N).
    sigma : float
        Noise standard deviation.
    learning_rate : float
        Step-size α.
    antithetic : bool
        If True, use mirrored/antithetic sampling.
    fitness_shaping : str
        ``"centered_rank"`` | ``"normalized"`` | ``"raw"``.
    weight_decay : float
        L2 weight decay applied to parameters (0 = off).
    seed : int
        Random seed.
    """

    def __init__(
        self,
        population_size: int = 20,
        sigma: float = 0.01,
        learning_rate: float = 0.001,
        antithetic: bool = True,
        fitness_shaping: str = "centered_rank",
        weight_decay: float = 0.0,
        seed: int = 42,
    ):
        self.population_size = population_size
        self.sigma = sigma
        self.lr = learning_rate
        self.antithetic = antithetic
        self.fitness_shaping = fitness_shaping
        self.weight_decay = weight_decay

        self._generator = torch.Generator()
        self._generator.manual_seed(seed)

        # Store noise vectors between ask()/tell() so we don't have to re-generate
        self._noise_cache: list[torch.Tensor] = []

    # ── ask ────────────────────────────────────────────────────────
    def ask(self, current_params: torch.Tensor) -> list[torch.Tensor]:
        dim = current_params.numel()
        device = current_params.device

        if self.antithetic:
            n_pairs = self.population_size // 2
            noises = antithetic_pairs(
                dim, n_pairs, sigma=self.sigma, generator=self._generator, device=device
            )
            # If population_size is odd, add one extra
            if self.population_size % 2 == 1:
                noises.append(gaussian_noise(dim, sigma=self.sigma, generator=self._generator, device=device))
        else:
            noises = [
                gaussian_noise(dim, sigma=self.sigma, generator=self._generator, device=device)
                for _ in range(self.population_size)
            ]

        self._noise_cache = noises
        candidates = [current_params + eps for eps in noises]
        return candidates

    # ── tell ───────────────────────────────────────────────────────
    def tell(
        self,
        current_params: torch.Tensor,
        candidates: list[torch.Tensor],
        fitnesses: list[float],
    ) -> ESResult:
        assert len(fitnesses) == len(self._noise_cache), (
            f"Got {len(fitnesses)} fitnesses but {len(self._noise_cache)} noise vectors"
        )

        # Fitness shaping
        if self.fitness_shaping == "centered_rank":
            shaped = _centered_rank_transform(fitnesses)
        elif self.fitness_shaping == "normalized":
            shaped = _normalize(fitnesses)
        else:
            shaped = np.array(fitnesses, dtype=np.float64)

        # Weighted combination of noise vectors
        #   grad_est = (1 / Nσ) Σ shaped_i · ε_i
        n = len(self._noise_cache)
        grad = torch.zeros_like(current_params)
        for i, eps in enumerate(self._noise_cache):
            grad += float(shaped[i]) * eps
        grad /= n * self.sigma

        # Weight decay
        if self.weight_decay > 0:
            grad -= self.weight_decay * current_params

        new_params = current_params + self.lr * grad

        # Stats
        best_idx = int(np.argmax(fitnesses))
        return ESResult(
            new_params=new_params,
            best_fitness=fitnesses[best_idx],
            mean_fitness=float(np.mean(fitnesses)),
            std_fitness=float(np.std(fitnesses)),
            best_perturbation=self._noise_cache[best_idx],
        )
