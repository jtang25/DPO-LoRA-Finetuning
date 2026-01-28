"""Train a LoRA-adapted policy with DPO."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dpo_project.data import load_ultrafeedback
from dpo_project.lora import apply_lora, lora_state_dict
from dpo_project.training import evaluate_accuracy, save_log, train


def _parse_modules(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_dtype(device: torch.device, dtype_arg: str | None) -> torch.dtype:
    if dtype_arg:
        mapping = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        return mapping[dtype_arg]
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a DPO LoRA adapter.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrafeedback_binarized")
    parser.add_argument("--split", default="train_prefs")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--n-samples", type=int, default=50000)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--eval-size", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.9)

    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument(
        "--target-modules",
        type=_parse_modules,
        default=_parse_modules("q_proj,k_proj,v_proj,o_proj"),
        help="Comma-separated module name fragments for LoRA injection.",
    )

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None)
    parser.add_argument("--skip-baseline", action="store_true")

    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--lora-path", default=None)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(device, args.dtype)

    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs/run")
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log_path) if args.log_path else output_dir / "training_log.json"
    lora_path = Path(args.lora_path) if args.lora_path else output_dir / "lora_weights.pt"

    print(f"Using device: {device}")
    print(f"Loading {args.model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    model = apply_lora(model, rank=args.rank, alpha=args.alpha, target_modules=args.target_modules)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    print("Loading UltraFeedback...")
    data = load_ultrafeedback(
        tokenizer,
        max_len=args.max_len,
        n_samples=args.n_samples,
        split=args.split,
        dataset_name=args.dataset,
    )

    train_size = args.train_size
    eval_size = args.eval_size
    if train_size is None:
        train_size = int(len(data) * args.train_fraction)
    if eval_size is None:
        eval_size = len(data) - train_size

    train_data = data[:train_size]
    eval_data = data[train_size : train_size + eval_size]

    if not args.skip_baseline:
        print("Evaluating baseline (before DPO)...")
        baseline_acc = evaluate_accuracy(
            model,
            ref_model,
            eval_data,
            batch_size=args.eval_batch_size,
            beta=args.beta,
            device=device,
        )
        print(f"Baseline preference accuracy: {baseline_acc:.2%}")

    log = train(
        model,
        ref_model,
        train_data,
        eval_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        beta=args.beta,
        log_every=args.log_every,
        grad_clip=args.grad_clip,
        device=device,
    )

    save_log(log, log_path)
    torch.save(lora_state_dict(model), lora_path)

    print(f"Saved training log to {log_path}")
    print(f"Saved LoRA weights to {lora_path}")


if __name__ == "__main__":
    main()

