# Pretraining & Scaling


## Overview
Language modeling objectives, tokenizer choices, scaling laws, and data mixtures.

## Key Papers
- Kaplan et al., 2020: Scaling Laws
- Hoffman et al., 2022: Chinchilla
- Brown et al., 2020: GPT-3

## Tutorials
- Tokenizer trade-offs, data deduplication notes

## Code Starters
- `src/gpt_block.py`
- `src/data_pipeline_wikitext.py`

## Exercises
- [ ] Train a small GPT on WikiText-2
- [ ] Scaling sweep (depth/width/context length)
- **Deliverables**: loss vs. compute plots, scaling fit
- **Success Metrics**: Predictable loss scaling; clean training curves
