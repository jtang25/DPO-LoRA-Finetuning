import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import copy
import json

class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_f, out_f = original_linear.in_features, original_linear.out_features
        dtype = original_linear.weight.dtype  # Match original dtype
        device = original_linear.weight.device
        
        # Freeze original
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
        
        # LoRA matrices - same dtype as original
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f, dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, dtype=dtype, device=device))
        
        # Init: A with kaiming, B with zeros (so BA=0 initially)
        nn.init.kaiming_uniform_(self.lora_A)
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        # Original path (frozen) + LoRA path (trainable)
        original_out = self.original(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return original_out + self.scaling * lora_out


def apply_lora(model, rank=8, alpha=16.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            # Get parent module
            parts = name.rsplit(".", 1)
            parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
            attr = parts[-1]
            
            # Replace with LoRA
            setattr(parent, attr, LoRALinear(module, rank=rank, alpha=alpha))
    
    # Freeze all non-LoRA params
    for n, p in model.named_parameters():
        p.requires_grad = "lora_" in n
    
    return model

def get_log_probs(model, input_ids, attention_mask, prompt_lens, normalize=True):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:]
    
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
    
    batch_size, seq_len = shift_labels.shape
    positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
    response_mask = (positions >= (prompt_lens.unsqueeze(1) - 1)).float()
    full_mask = response_mask * shift_mask.float()
    
    sum_log_probs = (token_log_probs * full_mask).sum(dim=-1)
    
    if normalize:
        # Divide by number of response tokens
        num_tokens = full_mask.sum(dim=-1).clamp(min=1)
        return sum_log_probs / num_tokens
    
    return sum_log_probs

def dpo_loss(policy_model, ref_model, batch, beta=0.1):
    chosen_ids, chosen_mask, rejected_ids, rejected_mask, prompt_lens = batch
    
    # Policy log probs
    policy_chosen_logps = get_log_probs(policy_model, chosen_ids, chosen_mask, prompt_lens)
    policy_rejected_logps = get_log_probs(policy_model, rejected_ids, rejected_mask, prompt_lens)
    
    # Reference log probs (no grad)
    with torch.no_grad():
        ref_chosen_logps = get_log_probs(ref_model, chosen_ids, chosen_mask, prompt_lens)
        ref_rejected_logps = get_log_probs(ref_model, rejected_ids, rejected_mask, prompt_lens)
    
    # Log ratios (implicit rewards)
    chosen_rewards = policy_chosen_logps - ref_chosen_logps
    rejected_rewards = policy_rejected_logps - ref_rejected_logps
    
    # DPO loss: -log(sigmoid(beta * (r_chosen - r_rejected)))
    logits = beta * (chosen_rewards - rejected_rewards)
    loss = F.softplus(-logits).mean()
    
    # Metrics
    acc = (logits > 0).float().mean().item()
    margin = (chosen_rewards - rejected_rewards).mean().item()
    
    return loss, {"acc": acc, "margin": margin}

def load_ultrafeedback(tokenizer, max_len=512, n_samples=1000):
    """
    Load UltraFeedback binarized dataset.
    Each sample has a prompt, chosen response, and rejected response.
    """
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    ds = ds.shuffle(seed=42).select(range(n_samples))
    
    data = []
    for ex in tqdm(ds, desc="Tokenizing"):
        prompt = ex["prompt"]
        chosen = ex["chosen"][1]["content"]   # [1] is assistant turn
        rejected = ex["rejected"][1]["content"]
        
        # Tokenize prompt to get its length
        prompt_toks = tokenizer(prompt, add_special_tokens=False)
        prompt_len = len(prompt_toks["input_ids"]) + 1  # +1 for bos
        
        # Tokenize full sequences
        chosen_full = tokenizer(prompt + chosen, max_length=max_len, truncation=True, padding="max_length", return_tensors="pt")
        rejected_full = tokenizer(prompt + rejected, max_length=max_len, truncation=True, padding="max_length", return_tensors="pt")
        
        data.append({
            "chosen_ids": chosen_full["input_ids"].squeeze(0),
            "chosen_mask": chosen_full["attention_mask"].squeeze(0),
            "rejected_ids": rejected_full["input_ids"].squeeze(0),
            "rejected_mask": rejected_full["attention_mask"].squeeze(0),
            "prompt_len": prompt_len,
        })
    
    return data


def collate_fn(batch):
    """Stack batch into tensors."""
    return (
        torch.stack([x["chosen_ids"] for x in batch]),
        torch.stack([x["chosen_mask"] for x in batch]),
        torch.stack([x["rejected_ids"] for x in batch]),
        torch.stack([x["rejected_mask"] for x in batch]),
        torch.tensor([x["prompt_len"] for x in batch]),
    )

def train(model, ref_model, train_data, eval_data, epochs=1, batch_size=4, lr=1e-4, beta=0.1):
    """Minimal training loop with logging."""
    device = next(model.parameters()).device
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_data, batch_size=batch_size, collate_fn=collate_fn)
    
    # Only optimize LoRA params
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
    optimizer = torch.optim.AdamW(lora_params, lr=lr)
    
    print(f"Training on {len(train_data)} samples, evaluating on {len(eval_data)} samples")
    print(f"LoRA params: {sum(p.numel() for p in lora_params):,}")
    
    # Logging
    log = {"train_loss": [], "train_acc": [], "eval_acc": []}
    
    for epoch in range(epochs):
        model.train()
        total_loss, total_acc = 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(pbar):
            batch = tuple(t.to(device) for t in batch)
            
            loss, metrics = dpo_loss(model, ref_model, batch, beta=beta)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_acc += metrics["acc"]
            pbar.set_postfix(loss=loss.item(), acc=metrics["acc"])
            
            # Log every 10 steps
            if step % 10 == 0:
                log["train_loss"].append(loss.item())
                log["train_acc"].append(metrics["acc"])
        
        # Eval
        model.eval()
        eval_acc = 0
        with torch.no_grad():
            for batch in eval_loader:
                batch = tuple(t.to(device) for t in batch)
                _, metrics = dpo_loss(model, ref_model, batch, beta=beta)
                eval_acc += metrics["acc"]
        
        n_train, n_eval = len(train_loader), len(eval_loader)
        final_eval_acc = eval_acc / n_eval
        log["eval_acc"].append(final_eval_acc)
        print(f"Epoch {epoch+1}: train_loss={total_loss/n_train:.4f}, train_acc={total_acc/n_train:.2%}, eval_acc={final_eval_acc:.2%}")
    
    # Save log
    with open("training_log.json", "w") as f:
        json.dump(log, f)
    print("Saved training_log.json")
    
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    
    # Reference model (frozen copy)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    
    # Apply LoRA to policy model
    # Qwen uses: q_proj, k_proj, v_proj, o_proj in attention
    model = apply_lora(model, rank=8, alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # Load data
    print("Loading UltraFeedback...")
    data = load_ultrafeedback(tokenizer, max_len=512, n_samples=50000)
    train_data = data[:45000]
    eval_data = data[45000:]

    # Baseline eval (before any training)
    print("Evaluating baseline (before DPO)...")
    eval_loader = DataLoader(eval_data, batch_size=8, collate_fn=collate_fn)
    baseline_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in eval_loader:
            batch = tuple(t.to(device) for t in batch)
            _, metrics = dpo_loss(model, ref_model, batch, beta=0.2)
            baseline_acc += metrics["acc"]
    baseline_acc /= len(eval_loader)
    print(f"Baseline preference accuracy: {baseline_acc:.2%}")

    # Train
    model = train(model, ref_model, train_data, eval_data, epochs=1, batch_size=16, lr=5e-5, beta=0.2)
    
    # Save
    torch.save({n: p for n, p in model.named_parameters() if "lora_" in n}, "lora_weights.pt")
    print("Saved LoRA weights to lora_weights.pt")


if __name__ == "__main__":
    main()
