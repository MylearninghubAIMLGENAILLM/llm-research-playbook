# Fine-tuning & PEFT


## Overview
Parameter-efficient fine-tuning (LoRA, adapters), quantization-aware training, and inference efficiency.

## Key Papers
- Hu et al., 2021: LoRA
- Lester et al., 2021: Prompt Tuning
- Dettmers et al., 2023: QLoRA

## Tutorials
- Hugging Face PEFT docs

## Code Starters
- `src/lora_gpt2_finetune.py`
- `src/quantized_inference.py`

## Exercises
- [ ] Fine-tune GPT-2 on a domain dataset with LoRA
- [ ] Compare full FT vs. LoRA vs. prompt tuning
- **Deliverables**: leaderboard + cost analysis
- **Success Metrics**: Similar quality with ≤10% trainable params
