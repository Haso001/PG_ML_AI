"""Abstract base class for Evolutionary Strategy algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ESResult:
    """Result of one ES generation step."""
    new_params: torch.Tensor        # updated flat parameter vector
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    best_perturbation: torch.Tensor  # best ε from this generation


class BaseES(ABC):
    """Interface that every ES algorithm must implement."""

    @abstractmethod
    def ask(self, current_params: torch.Tensor) -> list[torch.Tensor]:
        """Generate a population of candidate parameter vectors.

        Parameters
        ----------
        current_params : Tensor
            Current flat parameter vector (1-D).

        Returns
        -------
        list[Tensor]
            List of candidate vectors, length = ``population_size``.
        """

    @abstractmethod
    def tell(
        self,
        current_params: torch.Tensor,
        candidates: list[torch.Tensor],
        fitnesses: list[float],
    ) -> ESResult:
        """Update parameters based on fitness evaluations.

        Parameters
        ----------
        current_params : Tensor
            Parameter vector *before* this generation.
        candidates : list[Tensor]
            Same list returned by :meth:`ask`.
        fitnesses : list[float]
            Scalar fitness for each candidate (higher = better).

        Returns
        -------
        ESResult
        """
