"""Training loops for DPO."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dpo_project.data import collate_fn
from dpo_project.lora import lora_parameters
from dpo_project.losses import dpo_loss


def evaluate_accuracy(
    model,
    ref_model,
    eval_data,
    batch_size: int,
    beta: float,
    device: torch.device | None = None,
) -> float:
    device = device or next(model.parameters()).device
    eval_loader = DataLoader(eval_data, batch_size=batch_size, collate_fn=collate_fn)

    model.eval()
    total_acc = 0.0
    with torch.no_grad():
        for batch in eval_loader:
            batch = tuple(t.to(device) for t in batch)
            _, metrics = dpo_loss(model, ref_model, batch, beta=beta)
            total_acc += metrics["acc"]

    return total_acc / max(1, len(eval_loader))


def train(
    model,
    ref_model,
    train_data,
    eval_data,
    epochs: int = 1,
    batch_size: int = 4,
    eval_batch_size: int | None = None,
    lr: float = 1e-4,
    beta: float = 0.1,
    log_every: int = 10,
    grad_clip: float = 1.0,
    device: torch.device | None = None,
) -> dict[str, list[float]]:
    """Train LoRA adapters with DPO and return a log dict."""
    device = device or next(model.parameters()).device
    eval_batch_size = eval_batch_size or batch_size

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    lora_params = lora_parameters(model)
    optimizer = torch.optim.AdamW(lora_params, lr=lr)

    print(f"Training on {len(train_data)} samples, evaluating on {len(eval_data)} samples")
    print(f"LoRA params: {sum(p.numel() for p in lora_params):,}")

    log: dict[str, list[float]] = {"train_loss": [], "train_acc": [], "eval_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_acc = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for step, batch in enumerate(pbar):
            batch = tuple(t.to(device) for t in batch)

            loss, metrics = dpo_loss(model, ref_model, batch, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, grad_clip)
            optimizer.step()

            total_loss += loss.item()
            total_acc += metrics["acc"]
            pbar.set_postfix(loss=loss.item(), acc=metrics["acc"])

            if step % log_every == 0:
                log["train_loss"].append(loss.item())
                log["train_acc"].append(metrics["acc"])

        eval_acc = evaluate_accuracy(
            model,
            ref_model,
            eval_data,
            batch_size=eval_batch_size,
            beta=beta,
            device=device,
        )
        log["eval_acc"].append(eval_acc)

        n_train = len(train_loader)
        print(
            f"Epoch {epoch + 1}: train_loss={total_loss / n_train:.4f}, "
            f"train_acc={total_acc / n_train:.2%}, eval_acc={eval_acc:.2%}"
        )

    return log


def save_log(log: dict[str, list[float]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

