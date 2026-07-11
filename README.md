# Chimera

**An interactive LLM inference operating system** — a tool to *understand, benchmark, and optimize* transformer serving from first principles.

> If Wireshark is for networks and Chrome DevTools is for browsers, Chimera is for LLM inference.

Modern LLM tools (ChatGPT, Claude, Cursor) hide *why* generation is slow or fast, why memory explodes, and what techniques like KV cache, continuous batching, PagedAttention, and speculative decoding actually do. Chimera exposes all of it — a real, hand-built inference engine emitting per-token telemetry, made watchable in the browser.

## Results so far

Every number below is reproducible from the scripts in this repo.

| | Result | How it's measured |
|---|---|---|
| **Correctness** | Logits match HuggingFace GPT-2 to **5e-5** | Hand-built modules vs. the reference implementation, same weights |
| **KV cache** | Up to **46× faster** decode steps (453ms → 96ms @ 900 tokens) | `phase2_benchmark.py` — output stays token-for-token identical |
| **Continuous batching** | **3.7× throughput** (9.8 → 36.2 tok/s) | `phase4_showdown.py` — 20-user workload, 8 seats |
| **Time-to-first-token** | **7.6× faster** (25.8s → 3.4s) | same run, vs. sequential serving |
| **GPU seat utilization** | **43% → 80%** vs. static batching | stragglers no longer hold seats hostage |

## Approach

The engine is built **from first principles** — every transformer module (embedding, attention, MLP, LayerNorm) is implemented by hand and loads real pretrained GPT-2 weights, so output is coherent *and* every tensor is understood. No `generate()` black boxes.

Runs **CPU-first, GPU-ready** (CUDA auto-detected).

## Build phases

Each phase is *understand-first, then build*:

- **Phase 0** — Foundations: what inference is, why generation is a loop
- **Phase 1** — The transformer / tiny GPT ✅
- **Phase 2** — KV Cache ✅
- **Phase 3** — Inference Engine ✅
- **Phase 4** — Scheduling & continuous batching ✅
- **Phase 5** — PagedAttention *(flagship)*
- **Phase 6** — Speculative decoding
- **Phase 7** — Telemetry pipeline
- **Phase 8** — Frontend & the visual lenses
- **Phase 9** — Productionization

## Stack

**Backend:** Python · PyTorch · FastAPI · AsyncIO
**Frontend (later):** Next.js · TypeScript · Tailwind · D3 / React Flow / Recharts
**Telemetry:** Prometheus · ClickHouse / TimescaleDB

## Running (so far)

```bash
cd backend
python scripts/step1_smoke.py       # verify tokenizer + real GPT-2 weights load
python scripts/generate_demo.py     # our from-scratch GPT generates real text
python scripts/phase2_benchmark.py  # naive vs KV-cached generation, measured
python scripts/phase3_engine_demo.py # engine interleaving two requests, live events
python scripts/phase4_showdown.py   # sequential vs static vs continuous batching
```

## Status

**Phases 1–4 complete.** A GPT-2 built entirely from first principles — hand-written
embedding, multi-head causal attention, MLP, LayerNorm, residuals, and unembed —
that loads real pretrained weights and generates coherent text. Correctness is
verified against HuggingFace GPT-2 (logits match to ~5e-5).

Phase 2 adds a hand-built KV cache: prefill fills it in one pass, decode feeds
only the newest token per step. Cached generation is token-for-token identical
to the naive loop, with per-step cost ~constant instead of growing with sequence
length (measured on CPU: 5.9× faster at 64 tokens, 46× at 900).

Phase 3 adds the InferenceEngine: each prompt becomes a Request with a
lifecycle (WAITING → PREFILLING → DECODING → FINISHED) and its own KV cache;
`step()` advances any request by exactly one unit, so multiple generations
interleave; every step emits a StepEvent (token, latency, cache size) — the
per-token stream the telemetry and UI will consume.

Phase 4 adds batched decode (many requests, one forward pass — per-row
positions, left-padded stacked caches, padding mask; batched output verified
token-identical to solo) and a scheduler with three raceable policies.
Measured on the same 20-user workload: sequential 9.8 tok/s (avg TTFT 25.8s) →
static batching 26.3 tok/s (43% seat utilization: stragglers) → continuous
batching 36.2 tok/s, TTFT 3.4s, 80% utilization.

Model code: [`backend/chimera/model/`](backend/chimera/model/) · Engine: [`backend/chimera/engine/`](backend/chimera/engine/).
