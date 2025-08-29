# Seq2Seq RNN


## Overview
Encoder-decoder with RNN/LSTM/GRU for translation and summarization.

## Key Papers
- Sutskever et al., 2014: Sequence to Sequence Learning
- Bahdanau et al., 2014: Additive attention (preview for next module)

## Tutorials
- "The Illustrated Seq2Seq"
- Stanford CS224n sequence models

## Code Starters
- `src/seq2seq_lstm.py` (teacher forcing + scheduled sampling)
- `src/beam_search.py`

## Exercises
- [ ] Train LSTM translator on a toy dataset (Multi30k)
- [ ] Add attention and compare BLEU
- **Deliverables**: BLEU table, error analysis
- **Success Metrics**: BLEU ↑ with attention vs. baseline
