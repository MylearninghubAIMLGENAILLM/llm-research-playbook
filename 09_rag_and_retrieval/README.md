# RAG & Retrieval


## Overview
Classical (BM25), dense retrieval (FAISS), and retrieval-augmented generation patterns.

## Key Papers
- Karpukhin et al., 2020: DPR
- Guu et al., 2020: REALM
- Lewis et al., 2020: RAG

## Tutorials
- FAISS quickstart, vector DB ecosystem notes

## Code Starters
- `src/build_faiss_index.py`
- `src/rag_pipeline.py`

## Exercises
- [ ] Build DPR/FAISS index over Wikipedia subset
- [ ] Latency/quality trade-off evaluation
- **Deliverables**: retrieval diagnostics + EM/F1 for QA
- **Success Metrics**: RAG > Base LM on open-domain QA
