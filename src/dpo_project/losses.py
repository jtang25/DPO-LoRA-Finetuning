"""Losses and log-prob utilities for DPO."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def get_log_probs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)

    batch_size, seq_len = shift_labels.shape
    positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
    prompt_lens = prompt_lens.to(input_ids.device)
    response_mask = (positions >= (prompt_lens.unsqueeze(1) - 1)).float()
    full_mask = response_mask * shift_mask.float()

    sum_log_probs = (token_log_probs * full_mask).sum(dim=-1)

    if normalize:
        num_tokens = full_mask.sum(dim=-1).clamp(min=1)
        return sum_log_probs / num_tokens

    return sum_log_probs


def dpo_loss(policy_model, ref_model, batch, beta: float = 0.1) -> tuple[torch.Tensor, dict[str, float]]:
    chosen_ids, chosen_mask, rejected_ids, rejected_mask, prompt_lens = batch

    policy_chosen_logps = get_log_probs(policy_model, chosen_ids, chosen_mask, prompt_lens)
    policy_rejected_logps = get_log_probs(policy_model, rejected_ids, rejected_mask, prompt_lens)

    with torch.no_grad():
        ref_chosen_logps = get_log_probs(ref_model, chosen_ids, chosen_mask, prompt_lens)
        ref_rejected_logps = get_log_probs(ref_model, rejected_ids, rejected_mask, prompt_lens)

    chosen_rewards = policy_chosen_logps - ref_chosen_logps
    rejected_rewards = policy_rejected_logps - ref_rejected_logps

    logits = beta * (chosen_rewards - rejected_rewards)
    loss = F.softplus(-logits).mean()

    acc = (logits > 0).float().mean().item()
    margin = (chosen_rewards - rejected_rewards).mean().item()

    return loss, {"acc": acc, "margin": margin}

