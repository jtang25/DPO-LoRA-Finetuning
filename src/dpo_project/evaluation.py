"""LLM-as-judge evaluation helpers."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from openai import OpenAI


@dataclass
class JudgeResult:
    wins: int
    losses: int
    ties: int

    @property
    def total(self) -> int:
        return self.wins + self.losses + self.ties

    @property
    def win_rate(self) -> float:
        return self.wins / max(1, self.wins + self.losses)


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return response.strip()


def judge_with_llm(
    prompt: str,
    response_a: str,
    response_b: str,
    client: OpenAI,
    model: str,
) -> str:
    judge_prompt = (
        "You are an impartial judge. Given a prompt and two responses, decide which response is better.\n"
        "Consider: helpfulness, accuracy, clarity, and overall quality.\n\n"
        f"Prompt: {prompt}\n\n"
        "Response A:\n"
        f"{response_a}\n\n"
        "Response B:\n"
        f"{response_b}\n\n"
        'Which response is better? Answer with ONLY "A", "B", or "tie".'
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": judge_prompt}],
        max_tokens=5,
        temperature=0,
    )

    answer = response.choices[0].message.content.strip().upper()
    if "A" in answer:
        return "A"
    if "B" in answer:
        return "B"
    return "tie"


def run_llm_judge(
    prompts: list[str],
    dpo_model,
    ref_model,
    tokenizer,
    client: OpenAI,
    judge_model: str,
    max_new_tokens: int = 256,
) -> JudgeResult:
    wins = losses = ties = 0

    for prompt in prompts:
        dpo_response = generate_response(dpo_model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        sft_response = generate_response(ref_model, tokenizer, prompt, max_new_tokens=max_new_tokens)

        judgment1 = judge_with_llm(prompt, dpo_response, sft_response, client, judge_model)
        judgment2 = judge_with_llm(prompt, sft_response, dpo_response, client, judge_model)

        dpo_wins_1 = judgment1 == "A"
        dpo_wins_2 = judgment2 == "B"

        if dpo_wins_1 and dpo_wins_2:
            wins += 1
        elif not dpo_wins_1 and not dpo_wins_2 and judgment1 != "tie" and judgment2 != "tie":
            losses += 1
        else:
            ties += 1

    return JudgeResult(wins=wins, losses=losses, ties=ties)

