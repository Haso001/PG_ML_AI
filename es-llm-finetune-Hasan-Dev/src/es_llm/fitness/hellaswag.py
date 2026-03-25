"""HellaSwag fitness evaluator for ES training (binary/multiple-choice).

Evaluates the model on a subset of HellaSwag and returns accuracy as fitness.
"""

from __future__ import annotations
import random
from typing import Optional

import torch
from datasets import load_dataset
from torch import nn
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase

from .base import BaseFitness

class HellaSwagBinaryFitness(BaseFitness):
    def __init__(
        self,
        num_samples: int = 20,
        split: str = "validation",
        max_new_tokens: int = 32,
        seed: int = 42,
        reshuffle: bool = True,
        pool_size: Optional[int] = 4000,
    ):
        self.num_samples = num_samples
        self.split = split
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.reshuffle = reshuffle
        self.pool_size = pool_size
        self._init_data()

    def _init_data(self):
        ds = load_dataset("hellaswag")[self.split]
        if self.pool_size:
            ds = ds.select(range(min(self.pool_size, len(ds))))
        self.ds = ds
        self.indices = list(range(len(ds)))
        self.reshuffle_data()

    def reshuffle_data(self) -> None:
        if self.reshuffle:
            random.seed(self.seed)
            random.shuffle(self.indices)
        self.sample_indices = self.indices[: self.num_samples]

    def name(self) -> str:
        return "hellaswag-binary"

    @torch.inference_mode()
    def evaluate(self, model: nn.Module, tokenizer: PreTrainedTokenizerBase) -> float:
        correct = 0
        for idx in self.sample_indices:
            ex = self.ds[idx]
            ctx = ex["ctx_a"] + ex["ctx_b"]
            choices = ex["endings"]
            prompt = f"{ctx}\nOptions:\n"
            for i, c in enumerate(choices):
                prompt += f"{i}: {c}\n"
            prompt += "Answer with the number of the correct option."
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            output = model.generate(**inputs, max_new_tokens=2)
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            # Extrahiere die erste Zahl als Antwort
            import re
            m = re.search(r"(\d)", decoded)
            pred = int(m.group(1)) if m else -1
            gold = int(ex["label"])
            if pred == gold:
                correct += 1
        return correct / self.num_samples if self.num_samples > 0 else 0.0
