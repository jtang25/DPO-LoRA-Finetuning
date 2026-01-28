"""LoRA helpers for causal language models."""
from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Linear layer with a low-rank adaptation path."""

    def __init__(self, original_linear: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features
        dtype = original_linear.weight.dtype
        device = original_linear.weight.device

        # Freeze original weights.
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        # LoRA matrices match the original dtype/device.
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features, dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=dtype, device=device))

        # Initialize A with Kaiming and B with zeros so BA starts at 0.
        nn.init.kaiming_uniform_(self.lora_A)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_out = self.original(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return original_out + self.scaling * lora_out


def _resolve_parent(root: nn.Module, name: str) -> tuple[nn.Module, str]:
    if "." not in name:
        return root, name
    parent_path, attr = name.rsplit(".", 1)
    return root.get_submodule(parent_path), attr


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: Sequence[str] | None = None,
    freeze_base: bool = True,
) -> nn.Module:
    """Replace targeted Linear layers with LoRA-wrapped versions."""
    targets = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and any(t in name for t in targets):
            parent, attr = _resolve_parent(model, name)
            setattr(parent, attr, LoRALinear(module, rank=rank, alpha=alpha))

    if freeze_base:
        for param_name, param in model.named_parameters():
            param.requires_grad = "lora_" in param_name

    return model


def lora_parameters(model: nn.Module) -> list[torch.nn.Parameter]:
    """Return trainable LoRA parameters."""
    return [p for n, p in model.named_parameters() if "lora_" in n]


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return a state dict containing only LoRA parameters."""
    return {n: p.detach().cpu() for n, p in model.named_parameters() if "lora_" in n}

