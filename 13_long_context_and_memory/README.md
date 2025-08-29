# Long-Context & Memory


## Overview
Efficient attention (sliding, sparse, linear), position strategies, memory modules.

## Key Papers
- Beltagy et al., 2020: Longformer
- Press et al., 2021/22: ALiBi & position methods
- Dao et al., 2022: FlashAttention (efficiency)

## Tutorials
- Sliding-window & chunked attention trade-offs

## Code Starters
- `src/sliding_attention.py`
- `src/rope_alibi.py`

## Exercises
- [ ] Implement sliding attention; compare perplexity vs. full
- **Deliverables**: speed/quality curves
- **Success Metrics**: ≥2× speedup with ≤1% perplexity delta on long docs
