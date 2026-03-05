"""GSM8K fitness evaluator for ES training.

Evaluates the model on a subset of GSM8K and returns accuracy as fitness.
Uses the same extraction/normalization logic as scripts/eval_gsm8k.py.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import torch
from datasets import load_dataset
from torch import nn
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase

from .base import BaseFitness

logger = logging.getLogger("es_llm")

# ── Answer extraction ─────────────────────────────────────────────

SYSTEM_PROMPT = "You are a helpful assistant."
USER_TEMPLATE = (
    "Solve the following problem step by step.\n"
    "At the end, output ONLY the final numeric answer in exactly this format:\n"
    "#### <answer>\n\n"
    "Problem:\n{q}"
)


def extract_hash_answer(text: str) -> Optional[str]:
    """Extract the number after #### in model output."""
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return m.group(1)
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def normalize_answer(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    x = x.strip().replace(",", "").replace("$", "").replace(" ", "")
    try:
        fx = float(x)
        if fx.is_integer():
            return str(int(fx))
    except (ValueError, OverflowError):
        pass
    return x


# ── Fitness class ─────────────────────────────────────────────────


class GSM8KFitness(BaseFitness):
    """Evaluate a model on a random subset of GSM8K.

    Parameters
    ----------
    num_samples : int
        Number of examples to evaluate per fitness call.
    split : str
        ``"train"`` or ``"test"``.
    max_new_tokens : int
        Maximum tokens to generate per answer.
    seed : int
        Seed for subset selection (for reproducibility across generations).
    """

    def __init__(
        self,
        num_samples: int = 50,
        split: str = "train",
        max_new_tokens: int = 256,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.split = split
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self._dataset = None

    def _load_data(self):
        if self._dataset is not None:
            return
        logger.info("Loading GSM8K dataset (split=%s, n=%d)...", self.split, self.num_samples)
        ds = load_dataset("openai/gsm8k", "main", split=self.split)
        # Fix subset for reproducibility
        ds = ds.shuffle(seed=self.seed)
        self._dataset = ds.select(range(min(self.num_samples, len(ds))))
        logger.info("GSM8K loaded: %d examples", len(self._dataset))

    def name(self) -> str:
        return f"gsm8k_{self.split}_{self.num_samples}"

    @torch.inference_mode()
    def evaluate(self, model: nn.Module, tokenizer: PreTrainedTokenizerBase) -> float:
        """Run GSM8K evaluation. Returns accuracy ∈ [0, 1] as fitness."""
        self._load_data()
        model.eval()
        correct = 0

        for ex in tqdm(self._dataset, desc="    GSM8K Eval", unit="q", leave=False):
            question = ex["question"]
            gold = normalize_answer(extract_hash_answer(ex["answer"]))

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(q=question)},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
            gen = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
            out_ids = gen[0, inputs["input_ids"].shape[1] :]
            pred_text = tokenizer.decode(out_ids, skip_special_tokens=True)
            pred = normalize_answer(extract_hash_answer(pred_text))

            if pred is not None and gold is not None and pred == gold:
                correct += 1

        accuracy = correct / len(self._dataset) if len(self._dataset) > 0 else 0.0
        return accuracy
