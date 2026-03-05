"""GSM8K Log-Likelihood fitness evaluator for ES training.

Instead of generating answers and checking correctness (slow, discrete),
this evaluator measures how much probability the model assigns to the
**correct answer tokens** given the prompt — a fast, continuous fitness signal.

    Fitness = mean log P(gold_answer | prompt)

One forward pass per sample (~0.1s) instead of autoregressive generation (~2s).
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
    """

    def __init__(
        self,
        num_samples: int = 50,
        split: str = "train",
        target_mode: str = "short",
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.split = split
        self.target_mode = target_mode
        self.seed = seed
        self._dataset = None

    def _load_data(self):
        if self._dataset is not None:
            return
        logger.info("Loading GSM8K dataset (split=%s, n=%d)...", self.split, self.num_samples)
        ds = load_dataset("openai/gsm8k", "main", split=self.split)
        ds = ds.shuffle(seed=self.seed)
        self._dataset = ds.select(range(min(self.num_samples, len(ds))))
        logger.info("GSM8K loaded: %d examples", len(self._dataset))

    def name(self) -> str:
        return f"gsm8k_ll_{self.split}_{self.num_samples}"

    @torch.inference_mode()
    def evaluate(self, model: nn.Module, tokenizer: PreTrainedTokenizerBase) -> float:
        """Compute mean log-likelihood of gold answers. Higher = better."""
        self._load_data()
        model.eval()
        total_ll = 0.0

        for ex in tqdm(self._dataset, desc="    GSM8K LL", unit="q", leave=False):
            question = ex["question"]

            if self.target_mode == "short":
                gold_text = _extract_short_answer(ex["answer"])
            else:
                gold_text = _extract_gold_answer(ex["answer"])

            # Build prompt via chat template
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(q=question)},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Tokenize prompt and gold answer separately to know the boundary
            prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
            full_text = prompt_text + gold_text
            full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"].to(model.device)

            # The answer tokens start after the prompt
            answer_start = prompt_ids.shape[1]

            # Skip if answer has no tokens beyond prompt
            if full_ids.shape[1] <= answer_start:
                continue

            # Forward pass — get logits for all positions
            logits = model(full_ids).logits  # (1, seq_len, vocab_size)

            # Log-prob of each answer token:
            # logits[t] predicts token[t+1], so for answer tokens [answer_start:]
            # we use logits[answer_start-1 : -1]
            answer_logits = logits[0, answer_start - 1 : -1, :]  # (answer_len, vocab)
            answer_tokens = full_ids[0, answer_start:]            # (answer_len,)

            log_probs = torch.nn.functional.log_softmax(answer_logits, dim=-1)
            token_log_probs = log_probs.gather(1, answer_tokens.unsqueeze(1)).squeeze(1)

            # Mean log-prob over answer tokens for this example
            total_ll += token_log_probs.mean().item()

        # Average over all examples
        fitness = total_ll / len(self._dataset) if len(self._dataset) > 0 else -float("inf")
        return fitness
