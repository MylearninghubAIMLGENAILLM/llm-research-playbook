# Deployment & Serving


## Overview
Serving LLMs with FastAPI/gRPC, optimized inference (vLLM, TensorRT/Triton), and monitoring.

## References
- vLLM, Text Generation Inference (TGI)
- Triton Inference Server
- Quantization runtimes (bitsandbytes, GPTQ)

## Code Starters
- `src/app_fastapi.py` (streaming endpoints)
- `src/infer_vllm_client.py`

## Exercises
- [ ] Serve a GPT-2 endpoint with streaming tokens
- [ ] Load-test and profile latency/throughput
- **Deliverables**: latency P50/P95, cost-per-1k tokens
- **Success Metrics**: Meet target TPS and cost budget
