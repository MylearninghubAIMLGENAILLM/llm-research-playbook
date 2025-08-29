# RLHF & Alignment


## Overview
Supervised fine-tuning, reward modeling, PPO/DPO, and safety policies.

## Key Papers
- Ouyang et al., 2022: InstructGPT (SFT + RM + PPO)
- Rafailov et al., 2023: DPO
- Bai et al., 2022: Constitutional AI

## Tutorials
- PPO basics, preference datasets

## Code Starters
- `src/reward_model.py`
- `src/dpo_trainer.py`

## Exercises
- [ ] Train a toy reward model
- [ ] Compare PPO vs. DPO on summaries
- **Deliverables**: human eval + safety checklist
- **Success Metrics**: Preference win-rate ≥60% on held-out pairs
