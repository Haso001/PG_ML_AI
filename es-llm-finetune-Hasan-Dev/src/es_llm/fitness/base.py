"""Abstract base class for fitness evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase


class BaseFitness(ABC):
    """Interface for fitness evaluation of a model on a task.

    All fitness functions should return a scalar that is **higher = better**.
    """

    @abstractmethod
    def evaluate(self, model: nn.Module, tokenizer: PreTrainedTokenizerBase) -> float:
        """Evaluate the model and return a scalar fitness value."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this fitness function."""

    def reshuffle_data(self) -> None:
        """Signal a new generation — subclasses may resample data."""
