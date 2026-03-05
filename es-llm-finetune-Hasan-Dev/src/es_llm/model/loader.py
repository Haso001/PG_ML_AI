"""Model loading utilities for Qwen2.5 and other HuggingFace models."""

from __future__ import annotations

import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("es_llm")


def load_model_and_tokenizer(
    model_name: str,
    revision: str = "main",
    dtype: str = "float32",
    device: str = "cuda",
    model_path: Optional[str] = None,
):
    """Load a HuggingFace causal-LM model and its tokenizer.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier, e.g. ``"Qwen/Qwen2.5-0.5B-Instruct"``.
    revision : str
        Git revision / branch.
    dtype : str
        One of ``"float16"``, ``"bfloat16"``, ``"float32"``.
    device : str
        Target device (``"cpu"``, ``"cuda"``, ``"cuda:0"``).
    model_path : str, optional
        Load from a local checkpoint instead of HuggingFace Hub.

    Returns
    -------
    model, tokenizer
    """
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)
    source = model_path or model_name

    logger.info("Loading model: %s  (dtype=%s, device=%s)", source, dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        source,
        revision=revision,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded – %.1fM parameters", total_params / 1e6)

    return model, tokenizer


def inspect_model_layers(model) -> list[dict]:
    """Return a summary of all named parameters with shapes and sizes.

    Useful for deciding which layers to target with ES.
    """
    rows = []
    for name, param in model.named_parameters():
        rows.append({
            "name": name,
            "shape": list(param.shape),
            "numel": param.numel(),
            "dtype": str(param.dtype),
            "requires_grad": param.requires_grad,
        })
    return rows
