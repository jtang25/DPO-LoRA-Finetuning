# DPO Fine-Tuning with LoRA

Minimal from-scratch implementation of Direct Preference Optimization (DPO) with Low-Rank Adaptation (LoRA) for efficient LLM alignment.

## Results

| Metric | Value |
|--------|-------|
| Pairwise Preference Accuracy | **68%** (vs 50% random baseline) |
| Trainable Parameters | **0.03%** of total |
| LLM-as-Judge Win Rate | **63%** over SFT baseline |

Trained on UltraFeedback with Qwen2.5-7B-Instruct.

## Overview

**DPO** directly optimizes a language model to prefer chosen responses over rejected ones without explicit reward modeling. The loss maximizes the margin between log-probability ratios of chosen vs rejected completions relative to a frozen reference model.

**LoRA** enables efficient fine-tuning by decomposing weight updates into low-rank matrices `BA` where `B ∈ R^{d×r}` and `A ∈ R^{r×k}` with rank `r << min(d,k)`. Only these small matrices are trained while the base model stays frozen.

## Implementation Details

### LoRA Architecture
- Applied to attention projections: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Rank: 8, Alpha: 16 (scaling factor)
- Initialization: Kaiming uniform for A, zeros for B (ensures BA=0 at init)

### DPO Training
- Beta (KL penalty): 0.2
- Learning rate: 5e-5
- Batch size: 16
- Length-normalized log probabilities over response tokens only

### Evaluation
- **Preference accuracy**: Policy model's chosen/rejected log-prob margin vs reference
- **LLM-as-Judge**: GPT-4o-mini with position debiasing (swap A/B order, require consistent winner)

## Usage

### Training
```bas
python dpo.py
```

Trains for 1 epoch on 45k UltraFeedback samples, evaluates on 5k held-out. Saves `lora_weights.pt` and `training_log.json`.

### Evaluation
```bash
python eval.py
```

Runs position-debiased LLM-as-judge comparison on 50 test prompts.

## Files

```
├── dpo.py          # LoRA implementation + DPO training loop
├── eval.py         # LLM-as-judge evaluation
├── lora_weights.pt # Trained LoRA parameters
└── training_log.json
```

## Dependencies

```
torch
transformers
datasets
openai
tqdm
python-dotenv
```

## References

- [DPO Paper](https://arxiv.org/abs/2305.18290) - Rafailov et al. 2023
- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Hu et al. 2021
- [UltraFeedback](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)
