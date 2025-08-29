# Embeddings


## Overview
Tokenization & vector representations: word2vec, GloVe, subword (BPE, SentencePiece), contextual embeddings.

## Key Papers
- Mikolov et al., 2013: Efficient Estimation of Word Representations in Vector Space
- Pennington et al., 2014: GloVe
- Sennrich et al., 2016: BPE for NMT
- Devlin et al., 2018: BERT (contextual embeddings)

## Tutorials
- "The Illustrated Word2Vec/GloVe"
- SentencePiece documentation

## Code Starters
- `src/word2vec_skipgram.py`
- `src/tokenizers_bpe_sentencepiece.py`

## Exercises
- [ ] Train skip-gram with negative sampling on text8
- [ ] Visualize with t-SNE/UMAP
- **Deliverables**: embeddings plot, nearest-neighbor qualitative eval
- **Success Metrics**: Semantic analogies ≥50% on WordSim/Analogy subset
