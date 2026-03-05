"""Noise generation utilities for ES."""

from __future__ import annotations

import torch


def gaussian_noise(
    shape: int | tuple[int, ...],
    sigma: float = 1.0,
    generator: torch.Generator | None = None,
    device: str = "cuda",
) -> torch.Tensor:
    """Sample Gaussian noise N(0, σ²I) directly on *device*."""
    if isinstance(shape, int):
        shape = (shape,)
    return torch.randn(*shape, generator=generator, device=device) * sigma


def antithetic_pairs(
    dim: int,
    n_pairs: int,
    sigma: float = 1.0,
    generator: torch.Generator | None = None,
    device: str = "cuda",
) -> list[torch.Tensor]:
    """Generate *n_pairs* mirrored (antithetic) noise pairs → 2*n_pairs vectors.

    For each ε ~ N(0, σ²I) we also emit -ε, which halves gradient variance.
    """
    noises: list[torch.Tensor] = []
    for _ in range(n_pairs):
        eps = gaussian_noise(dim, sigma=sigma, generator=generator, device=device)
        noises.append(eps)
        noises.append(-eps)
    return noises


def low_rank_noise(
    rows: int,
    cols: int,
    rank: int,
    sigma: float = 1.0,
    generator: torch.Generator | None = None,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate a low-rank perturbation matrix A·B^T of shape (rows, cols).

    Parameters
    ----------
    rows, cols : int
        Desired perturbation shape.
    rank : int
        Rank of the factorized perturbation.
    """
    A = torch.randn(rows, rank, generator=generator, device=device) * (sigma / rank**0.5)
    B = torch.randn(cols, rank, generator=generator, device=device) * (sigma / rank**0.5)
    return A @ B.T  # shape (rows, cols)
