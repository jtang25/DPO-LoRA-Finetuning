"""Plot training logs into a PNG."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot DPO training curves.")
    parser.add_argument("--log-path", default="outputs/run/training_log.json")
    parser.add_argument("--out-path", default="assets/training_curves.png")
    parser.add_argument("--show", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    log_path = Path(args.log_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log = json.loads(log_path.read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(log["train_loss"], label="Train Loss", color="blue")
    axes[0].set_xlabel("Step (x10)")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("DPO Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(log["train_acc"], label="Train Acc", color="blue")
    if log.get("eval_acc"):
        steps_per_epoch = len(log["train_acc"]) // max(1, len(log["eval_acc"]))
        eval_x = [(i + 1) * steps_per_epoch - 1 for i in range(len(log["eval_acc"]))]
        axes[1].scatter(eval_x, log["eval_acc"], label="Eval Acc", color="red", s=100, zorder=5)
    axes[1].axhline(y=0.5, color="gray", linestyle="--", label="Random Baseline")
    axes[1].set_xlabel("Step (x10)")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("DPO Preference Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if args.show:
        plt.show()
    plt.close(fig)

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

