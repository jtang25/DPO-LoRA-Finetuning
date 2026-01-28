"""LLM-as-judge evaluation for DPO adapters."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from dpo_project.data import load_ultrafeedback_prompts
from dpo_project.evaluation import run_llm_judge
from dpo_project.lora import apply_lora


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
    parser = argparse.ArgumentParser(description="Run LLM-as-judge evaluation.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrafeedback_binarized")
    parser.add_argument("--split", default="test_prefs")
    parser.add_argument("--n-prompts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument(
        "--target-modules",
        type=_parse_modules,
        default=_parse_modules("q_proj,k_proj,v_proj,o_proj"),
        help="Comma-separated module name fragments for LoRA injection.",
    )

    parser.add_argument("--lora-path", default="outputs/run/lora_weights.pt")
    parser.add_argument("--max-new-tokens", type=int, default=256)

    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None)

    parser.add_argument("--output-path", default=None)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(device, args.dtype)

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Add it to .env or your environment.")
    client = OpenAI()

    print(f"Using device: {device}")
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype).to(device)
    ref_model.eval()

    print("Loading DPO model...")
    dpo_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype).to(device)
    dpo_model = apply_lora(dpo_model, rank=args.rank, alpha=args.alpha, target_modules=args.target_modules)

    print("Loading LoRA weights...")
    lora_state = torch.load(args.lora_path, map_location="cpu")
    dpo_model.load_state_dict(lora_state, strict=False)
    dpo_model.eval()

    print("Loading eval prompts...")
    eval_prompts = load_ultrafeedback_prompts(
        n_samples=args.n_prompts,
        split=args.split,
        seed=args.seed,
        dataset_name=args.dataset,
    )

    print("Judging...")
    result = run_llm_judge(
        eval_prompts,
        dpo_model,
        ref_model,
        tokenizer,
        client,
        judge_model=args.judge_model,
        max_new_tokens=args.max_new_tokens,
    )

    total = result.total
    print("\nLLM-as-Judge Results (position-debiased):")
    print(f"  DPO Wins: {result.wins}/{total} ({100 * result.wins / total:.1f}%)")
    print(f"  SFT Wins: {result.losses}/{total} ({100 * result.losses / total:.1f}%)")
    print(f"  Ties: {result.ties}/{total} ({100 * result.ties / total:.1f}%)")
    if result.wins + result.losses > 0:
        print(f"  Win Rate (excluding ties): {100 * result.win_rate:.1f}%")

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "wins": result.wins,
            "losses": result.losses,
            "ties": result.ties,
            "total": total,
            "win_rate": result.win_rate,
            "judge_model": args.judge_model,
            "model_name": args.model_name,
            "lora_path": args.lora_path,
            "n_prompts": args.n_prompts,
            "split": args.split,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()

