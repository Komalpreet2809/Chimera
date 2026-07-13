---
title: Chimera Inference Engine
emoji: 🧬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: LLM inference engine with live internals — KV cache, batching, PagedAttention
---

# Chimera — backend

The inference engine behind **[chimera.komalpreet.me](https://chimera.komalpreet.me)**.

A GPT-2 built from scratch in PyTorch — hand-written multi-head causal attention, KV
cache, continuous-batching scheduler, PagedAttention block allocator, and speculative
decoding — exposed over FastAPI + WebSockets so the frontend can stream not just tokens
but the engine's *internals*: per-token latency, KV-cache growth, block allocation, and
scheduler decisions.

Source: **https://github.com/Komalpreet2809/Chimera**

## Endpoints

| | |
|---|---|
| `GET /api/health` | liveness + device |
| `GET /api/config` | model dimensions |
| `WS /api/generate` | stream tokens + per-token telemetry |
| `WS /api/simulate` | stream scheduler ticks for an N-user workload |
| `POST /api/attention` | attention weights for any of the 144 heads |
| `POST /api/benchmark/cache` | naive vs KV-cached decode latency, measured live |
| `POST /api/benchmark/paged` | contiguous reservation vs paged blocks |

## Note on this Space

Runs on **CPU** (free tier), so generation is ~90ms/token and the live cache benchmark is
capped at 512-token sequences. That's not a limitation of the engine — it's the hardware.
The full-length benchmarks (and CUDA) run from the repo:

```bash
pytest backend/tests -q
python backend/scripts/phase2_benchmark.py
```
