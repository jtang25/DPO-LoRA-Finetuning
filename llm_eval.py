import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from dpo import LoRALinear, apply_lora
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def generate_response(model, tokenizer, prompt, max_new_tokens=256):
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
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def judge_with_llm(prompt, response_a, response_b, client):
    judge_prompt = f"""You are an impartial judge. Given a prompt and two responses, decide which response is better.
Consider: helpfulness, accuracy, clarity, and overall quality.

Prompt: {prompt}

Response A:
{response_a}

Response B:
{response_b}

Which response is better? Answer with ONLY "A", "B", or "tie"."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": judge_prompt}],
        max_tokens=5,
        temperature=0,
    )
    
    answer = response.choices[0].message.content.strip().upper()
    if "A" in answer:
        return "A"
    elif "B" in answer:
        return "B"
    return "tie"


def main():
    device = torch.device("cuda")
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    ref_model.eval()
    
    print("Loading DPO model...")
    dpo_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    dpo_model = apply_lora(dpo_model, rank=8, alpha=16, target_modules=["q_proj", "v_proj"])
    
    print("Loading LoRA weights...")
    lora_state = torch.load("lora_weights.pt")
    dpo_model.load_state_dict(lora_state, strict=False)
    dpo_model.eval()
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    print("Loading eval prompts...")
    eval_prompts = [ex["prompt"] for ex in load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs").select(range(50))]
    
    wins, losses, ties = 0, 0, 0
    
    for prompt in tqdm(eval_prompts, desc="Judging"):
        dpo_response = generate_response(dpo_model, tokenizer, prompt)
        sft_response = generate_response(ref_model, tokenizer, prompt)
        
        # Position debiasing
        judgment1 = judge_with_llm(prompt, dpo_response, sft_response, client)
        judgment2 = judge_with_llm(prompt, sft_response, dpo_response, client)
        
        dpo_wins_1 = judgment1 == "A"
        dpo_wins_2 = judgment2 == "B"
        
        if dpo_wins_1 and dpo_wins_2:
            wins += 1
        elif not dpo_wins_1 and not dpo_wins_2 and judgment1 != "tie" and judgment2 != "tie":
            losses += 1
        else:
            ties += 1
    
    total = len(eval_prompts)
    print(f"\nLLM-as-Judge Results (position-debiased):")
    print(f"  DPO Wins: {wins}/{total} ({100*wins/total:.1f}%)")
    print(f"  SFT Wins: {losses}/{total} ({100*losses/total:.1f}%)")
    print(f"  Ties: {ties}/{total} ({100*ties/total:.1f}%)")
    if wins + losses > 0:
        print(f"  Win Rate (excluding ties): {100*wins/(wins+losses):.1f}%")


if __name__ == "__main__":
    main()