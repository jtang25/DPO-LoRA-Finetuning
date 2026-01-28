"""Dataset utilities for UltraFeedback DPO."""
from __future__ import annotations

from typing import Iterable

import torch
from datasets import load_dataset
from tqdm import tqdm

DATASET_NAME = "HuggingFaceH4/ultrafeedback_binarized"


def load_ultrafeedback(
    tokenizer,
    max_len: int = 512,
    n_samples: int | None = 1000,
    split: str = "train_prefs",
    seed: int = 42,
    dataset_name: str = DATASET_NAME,
) -> list[dict[str, torch.Tensor]]:
    """Load and tokenize UltraFeedback preference pairs."""
    ds = load_dataset(dataset_name, split=split)
    if n_samples is not None:
        ds = ds.shuffle(seed=seed).select(range(n_samples))

    data: list[dict[str, torch.Tensor]] = []
    for ex in tqdm(ds, desc=f"Tokenizing {split}"):
        prompt = ex["prompt"]
        chosen = ex["chosen"][1]["content"]
        rejected = ex["rejected"][1]["content"]

        prompt_toks = tokenizer(prompt, add_special_tokens=False)
        prompt_len = len(prompt_toks["input_ids"])
        if tokenizer.bos_token_id is not None:
            prompt_len += 1
        prompt_len = min(prompt_len, max_len)

        chosen_full = tokenizer(
            prompt + chosen,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        rejected_full = tokenizer(
            prompt + rejected,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        data.append(
            {
                "chosen_ids": chosen_full["input_ids"].squeeze(0),
                "chosen_mask": chosen_full["attention_mask"].squeeze(0),
                "rejected_ids": rejected_full["input_ids"].squeeze(0),
                "rejected_mask": rejected_full["attention_mask"].squeeze(0),
                "prompt_len": torch.tensor(prompt_len),
            }
        )

    return data


def load_ultrafeedback_prompts(
    n_samples: int = 50,
    split: str = "test_prefs",
    seed: int = 42,
    dataset_name: str = DATASET_NAME,
) -> list[str]:
    ds = load_dataset(dataset_name, split=split)
    ds = ds.shuffle(seed=seed).select(range(n_samples))
    return [ex["prompt"] for ex in ds]


def collate_fn(batch: Iterable[dict[str, torch.Tensor]]):
    """Stack batch into tensors for DPO loss."""
    return (
        torch.stack([x["chosen_ids"] for x in batch]),
        torch.stack([x["chosen_mask"] for x in batch]),
        torch.stack([x["rejected_ids"] for x in batch]),
        torch.stack([x["rejected_mask"] for x in batch]),
        torch.stack([x["prompt_len"] for x in batch]),
    )

