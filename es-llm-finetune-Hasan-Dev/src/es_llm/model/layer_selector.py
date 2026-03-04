"""Layer selection & freezing utilities.

Qwen2.5-0.5B architecture reference
────────────────────────────────────
model.embed_tokens                    # Embedding
model.layers.{0..23}                  # 24 decoder blocks
  ├── self_attn
  │     ├── q_proj, k_proj, v_proj    # Query / Key / Value projections
  │     └── o_proj                    # Output projection
  ├── mlp
  │     ├── gate_proj                 # Gated MLP
  │     ├── up_proj
  │     └── down_proj
  ├── input_layernorm
  └── post_attention_layernorm
model.norm                            # Final RMSNorm
lm_head                              # Language-model head
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Set

import torch
from torch import nn

logger = logging.getLogger("es_llm")

# ── Component filters ────────────────────────────────────────────────
COMPONENT_PATTERNS: dict[str, str] = {
    "all":            r".*",
    "attention":      r"self_attn\.",
    "attention_qkv":  r"self_attn\.(q_proj|k_proj|v_proj)\.",
    "attention_o":    r"self_attn\.o_proj\.",
    "mlp":            r"mlp\.",
    "layernorm":      r"(input_layernorm|post_attention_layernorm)\.",
}


def _layer_prefix(idx: int) -> str:
    return f"model.layers.{idx}."


class LayerSelector:
    """Select which model parameters to perturb with ES.

    Freezes everything, then un-marks the selected parameters for ES perturbation.
    (We don't set ``requires_grad`` – ES does not use autograd.)
    """

    def __init__(
        self,
        model: nn.Module,
        strategy: str = "by_index",
        layer_indices: Optional[List[int]] = None,
        layer_names: Optional[List[str]] = None,
        layer_regex: Optional[str] = None,
        components: str = "all",
    ):
        self.model = model
        self.strategy = strategy
        self.layer_indices = layer_indices or []
        self.layer_names = layer_names or []
        self.layer_regex = layer_regex or ""
        self.components = components

        # Resolve the set of parameter names to perturb
        self._target_names: Set[str] = self._resolve()
        logger.info(
            "LayerSelector: %d target parameters selected (%s total elements)",
            len(self._target_names),
            f"{self.num_target_elements:,}",
        )

    # ── Public API ────────────────────────────────────────────────
    @property
    def target_names(self) -> list[str]:
        return sorted(self._target_names)

    @property
    def num_target_elements(self) -> int:
        return sum(
            p.numel() for n, p in self.model.named_parameters() if n in self._target_names
        )

    def get_target_params(self) -> list[tuple[str, nn.Parameter]]:
        """Return list of (name, Parameter) for all selected targets."""
        return [(n, p) for n, p in self.model.named_parameters() if n in self._target_names]

    def get_flat_params(self) -> torch.Tensor:
        """Concatenate all target parameter values into a flat 1-D tensor."""
        return torch.cat([p.data.view(-1) for _, p in self.get_target_params()])

    def set_flat_params(self, flat: torch.Tensor) -> None:
        """Write a flat 1-D tensor back into the target parameters."""
        offset = 0
        for _, p in self.get_target_params():
            numel = p.numel()
            p.data.copy_(flat[offset : offset + numel].view(p.shape))
            offset += numel

    # ── Resolution logic ──────────────────────────────────────────
    def _resolve(self) -> Set[str]:
        all_names = {n for n, _ in self.model.named_parameters()}
        comp_re = re.compile(COMPONENT_PATTERNS.get(self.components, self.components))

        if self.strategy == "by_index":
            prefixes = [_layer_prefix(i) for i in self.layer_indices]
            candidates = {n for n in all_names if any(n.startswith(p) for p in prefixes)}
        elif self.strategy == "by_name":
            candidates = {n for n in all_names if n in set(self.layer_names)}
        elif self.strategy == "by_regex":
            regex = re.compile(self.layer_regex)
            candidates = {n for n in all_names if regex.search(n)}
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Filter by component type
        selected = set()
        for name in candidates:
            # Extract the part after the layer prefix
            for prefix in [_layer_prefix(i) for i in range(100)]:
                if name.startswith(prefix):
                    suffix = name[len(prefix):]
                    if comp_re.search(suffix):
                        selected.add(name)
                    break
            else:
                # Not a numbered layer (e.g., lm_head) – include if strategy matches
                if self.strategy in ("by_name", "by_regex"):
                    selected.add(name)

        if not selected:
            logger.warning("LayerSelector: NO parameters matched the selection criteria!")

        return selected

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [f"Strategy: {self.strategy}  |  Components: {self.components}"]
        lines.append(f"Total target params: {len(self._target_names)}  ({self.num_target_elements:,} elements)")
        lines.append("")
        for name in self.target_names:
            p = dict(self.model.named_parameters())[name]
            lines.append(f"  {name:60s}  shape={list(p.shape)}  numel={p.numel():>10,}")
        return "\n".join(lines)
