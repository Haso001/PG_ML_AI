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
import math
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
        optimizer: str = "sgd",
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        sigma_decay: str = "constant",
        sigma_final: float | None = None,
        max_generations: int | None = None,
    ):
        self.population_size = population_size
        self.sigma = sigma
        self.lr = learning_rate
        self.antithetic = antithetic
        self.fitness_shaping = fitness_shaping
        self.weight_decay = weight_decay

        self._seed = seed
        self._generator: torch.Generator | None = None
        self._generator_device: str = ""

        # Store noise vectors between ask()/tell() so we don't have to re-generate
        self._noise_cache: list[torch.Tensor] = []

        # Adam optimizer state (Salimans et al., 2017 – practical improvement)
        self.optimizer_type = optimizer
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self._adam_m: torch.Tensor | None = None
        self._adam_v: torch.Tensor | None = None
        self._adam_t: int = 0

        # Sigma schedule (exploration → exploitation)
        self.sigma_decay = sigma_decay
        self.sigma_final = sigma_final if sigma_final is not None else sigma
        self.max_generations = max_generations
        self._generation: int = 0
        self._cached_sigma: float = sigma

    def _get_sigma(self) -> float:
        """Return current sigma, potentially decayed over generations."""
        if self.sigma_decay == "constant" or self.max_generations is None or self.max_generations <= 0:
            return self.sigma
        progress = min(self._generation / self.max_generations, 1.0)
        if self.sigma_decay == "cosine":
            return self.sigma_final + 0.5 * (self.sigma - self.sigma_final) * (1 + math.cos(math.pi * progress))
        elif self.sigma_decay == "linear":
            return self.sigma + (self.sigma_final - self.sigma) * progress
        return self.sigma

    @property
    def current_sigma(self) -> float:
        """Current noise standard deviation (accounts for schedule)."""
        return self._get_sigma()

    def _get_generator(self, device: torch.device | str) -> torch.Generator:
        """Return a seeded generator on the correct device (lazy init)."""
        dev_str = str(device)
        if self._generator is None or self._generator_device != dev_str:
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(self._seed)
            self._generator_device = dev_str
        return self._generator

    # ── ask ────────────────────────────────────────────────────────
    def ask(self, current_params: torch.Tensor) -> list[torch.Tensor]:
        dim = current_params.numel()
        device = current_params.device
        gen = self._get_generator(device)
        sigma = self._get_sigma()
        self._cached_sigma = sigma

        if self.antithetic:
            n_pairs = self.population_size // 2
            noises = antithetic_pairs(
                dim, n_pairs, sigma=sigma, generator=gen, device=device
            )
            # If population_size is odd, add one extra
            if self.population_size % 2 == 1:
                noises.append(gaussian_noise(dim, sigma=sigma, generator=gen, device=device))
        else:
            noises = [
                gaussian_noise(dim, sigma=sigma, generator=gen, device=device)
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
        grad /= n * self._cached_sigma

        # Weight decay
        if self.weight_decay > 0:
            grad -= self.weight_decay * current_params

        # Parameter update (SGD or Adam)
        if self.optimizer_type == "adam":
            self._adam_t += 1
            if self._adam_m is None:
                self._adam_m = torch.zeros_like(grad)
                self._adam_v = torch.zeros_like(grad)
            self._adam_m = self.adam_beta1 * self._adam_m + (1 - self.adam_beta1) * grad
            self._adam_v = self.adam_beta2 * self._adam_v + (1 - self.adam_beta2) * grad ** 2
            m_hat = self._adam_m / (1 - self.adam_beta1 ** self._adam_t)
            v_hat = self._adam_v / (1 - self.adam_beta2 ** self._adam_t)
            new_params = current_params + self.lr * m_hat / (torch.sqrt(v_hat) + self.adam_epsilon)
        else:
            new_params = current_params + self.lr * grad

        self._generation += 1

        # Stats
        best_idx = int(np.argmax(fitnesses))
        return ESResult(
            new_params=new_params,
            best_fitness=fitnesses[best_idx],
            mean_fitness=float(np.mean(fitnesses)),
            std_fitness=float(np.std(fitnesses)),
            best_perturbation=self._noise_cache[best_idx],
        )
