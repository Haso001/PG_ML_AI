"""GSM8K Log-Likelihood fitness evaluator for ES training.

Instead of generating answers and checking correctness (slow, discrete),
this evaluator measures how much probability the model assigns to the
**correct answer tokens** given the prompt — a fast, continuous fitness signal.

    Fitness = mean log P(gold_answer | prompt)

One forward pass per sample (~0.1s) instead of autoregressive generation (~2s).
"""

from __future__ import annotations

import logging
import random
import re
from typing import Optional

import torch
from datasets import load_dataset
from torch import nn
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase

from .base import BaseFitness

logger = logging.getLogger("es_llm")

SYSTEM_PROMPT = "You are a helpful assistant."
USER_TEMPLATE = (
    "Solve the following problem step by step.\n"
    "At the end, output ONLY the final numeric answer in exactly this format:\n"
    "#### <answer>\n\n"
    "Problem:\n{q}"
)


def _extract_gold_answer(answer_text: str) -> str:
    """Extract the final answer string from GSM8K 'answer' field.

    GSM8K answers contain step-by-step reasoning followed by #### <number>.
    We return the full reasoning + answer as the gold target for log-likelihood.
    """
    return answer_text.strip()


def _extract_short_answer(answer_text: str) -> str:
    """Extract just the '#### <number>' part for a shorter target."""
    m = re.search(r"####\s*(.+)", answer_text)
    if m:
        return "#### " + m.group(1).strip()
    return answer_text.strip()


class GSM8KLogLikelihoodFitness(BaseFitness):
    """Evaluate a model via log-likelihood of correct GSM8K answers.

    Parameters
    ----------
    num_samples : int
        Number of examples to evaluate per fitness call.
    split : str
        ``"train"`` or ``"test"``.
    target_mode : str
        ``"short"`` = only score the "#### <number>" part (fast, focused).
        ``"full"`` = score the entire step-by-step solution (slower, richer signal).
    seed : int
        Seed for subset selection.
    reshuffle : bool
        If True, sample a new random subset of ``num_samples`` from the full
        dataset on each ``evaluate()`` call.  Prevents overfitting to a fixed
        set when ES runs many generations.
    pool_size : int or None
        Size of the data pool to draw from.  ``None`` = use the entire split.
    """

    def __init__(
        self,
        num_samples: int = 50,
        split: str = "train",
        target_mode: str = "short",
        seed: int = 42,
        batch_size: int = 1,
        reshuffle: bool = False,
        pool_size: int | None = None,
    ):
        self.num_samples = num_samples
        self.split = split
        self.target_mode = target_mode
        self.seed = seed
        self.batch_size = batch_size
        self.reshuffle = reshuffle
        self.pool_size = pool_size
        self._dataset = None
        self._full_pool = None          # full data pool for reshuffling
        self._cached_inputs: list[dict] | None = None
        self._rng = random.Random(seed)
        self._needs_reshuffle = True     # reshuffle on first call

    def reshuffle_data(self) -> None:
        """Signal that a new generation starts — resample data on next evaluate()."""
        if self.reshuffle:
            self._needs_reshuffle = True

    def _load_data(self):
        if self._full_pool is None:
            logger.info("Loading GSM8K dataset (split=%s)...", self.split)
            ds = load_dataset("openai/gsm8k", "main", split=self.split)
            ds = ds.shuffle(seed=self.seed)
            if self.pool_size is not None:
                ds = ds.select(range(min(self.pool_size, len(ds))))
            self._full_pool = ds
            logger.info("GSM8K pool loaded: %d examples", len(self._full_pool))

        if self.reshuffle and self._needs_reshuffle:
            # Draw a fresh random subset — once per generation, not per candidate!
            indices = self._rng.sample(
                range(len(self._full_pool)),
                min(self.num_samples, len(self._full_pool)),
            )
            self._dataset = self._full_pool.select(indices)
            self._cached_inputs = None   # force re-tokenization
            self._needs_reshuffle = False
            logger.debug("Reshuffled: %d new examples for this generation", len(self._dataset))
        elif self._dataset is None:
            self._dataset = self._full_pool.select(
                range(min(self.num_samples, len(self._full_pool)))
            )

    def name(self) -> str:
        return f"gsm8k_ll_{self.split}_{self.num_samples}"

    def _prepare_inputs(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Pre-tokenize all samples and cache for repeated evaluate() calls."""
        self._cached_inputs = []
        for ex in self._dataset:
            question = ex["question"]
            if self.target_mode == "short":
                gold_text = _extract_short_answer(ex["answer"])
            else:
                gold_text = _extract_gold_answer(ex["answer"])

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(q=question)},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"][0]
            full_text = prompt_text + gold_text
            full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"][0]

            if full_ids.shape[0] > prompt_ids.shape[0]:
                self._cached_inputs.append({
                    "full_ids": full_ids,
                    "answer_start": prompt_ids.shape[0],
                })

    @torch.inference_mode()
    def evaluate(self, model: nn.Module, tokenizer: PreTrainedTokenizerBase) -> float:
        """Compute mean log-likelihood of gold answers. Higher = better.

        Supports batched forward passes via ``batch_size`` for faster evaluation
        on GPUs with sufficient VRAM (A100/H100).
        """
        self._load_data()
        model.eval()

        if self._cached_inputs is None:
            self._prepare_inputs(tokenizer)

        if not self._cached_inputs:
            return -float("inf")

        device = next(model.parameters()).device
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        total_ll = 0.0
        count = 0

        for batch_start in range(0, len(self._cached_inputs), self.batch_size):
            batch = self._cached_inputs[batch_start:batch_start + self.batch_size]
            max_len = max(item["full_ids"].shape[0] for item in batch)

            input_ids = torch.full(
                (len(batch), max_len), pad_id, dtype=torch.long, device=device
            )
            attention_mask = torch.zeros(
                (len(batch), max_len), dtype=torch.long, device=device
            )

            for i, item in enumerate(batch):
                ids = item["full_ids"]
                input_ids[i, :ids.shape[0]] = ids.to(device)
                attention_mask[i, :ids.shape[0]] = 1

            logits = model(input_ids, attention_mask=attention_mask).logits
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            for i, item in enumerate(batch):
                ans_start = item["answer_start"]
                seq_len = item["full_ids"].shape[0]

                ans_log_probs = log_probs[i, ans_start - 1 : seq_len - 1, :]
                ans_tokens = input_ids[i, ans_start:seq_len]
                token_lls = ans_log_probs.gather(1, ans_tokens.unsqueeze(1)).squeeze(1)
                total_ll += token_lls.mean().item()
                count += 1

        return total_ll / count if count > 0 else -float("inf")
