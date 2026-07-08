# Chimera

**An interactive LLM inference operating system** — a tool to *understand, benchmark, and optimize* transformer serving from first principles.

> If Wireshark is for networks and Chrome DevTools is for browsers, Chimera is for LLM inference.

Modern LLM tools (ChatGPT, Claude, Cursor) hide *why* generation is slow or fast, why memory explodes, and what techniques like KV cache, continuous batching, PagedAttention, and speculative decoding actually do. Chimera exposes all of it — a real, hand-built inference engine emitting per-token telemetry, made watchable in the browser.

## Approach

The engine is built **from first principles** — every transformer module (embedding, attention, MLP, LayerNorm) is implemented by hand and loads real pretrained GPT-2 weights, so output is coherent *and* every tensor is understood. No `generate()` black boxes.

Runs **CPU-first, GPU-ready** (CUDA auto-detected).

## Build phases

Each phase is *understand-first, then build*:

- **Phase 0** — Foundations: what inference is, why generation is a loop
- **Phase 1** — The transformer / tiny GPT ✅
- **Phase 2** — KV Cache
- **Phase 3** — Inference Engine
- **Phase 4** — Scheduling & continuous batching
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
python scripts/step1_smoke.py     # verify tokenizer + real GPT-2 weights load
python scripts/generate_demo.py   # our from-scratch GPT generates real text
```

## Status

**Phase 1 complete.** A GPT-2 built entirely from first principles — hand-written
embedding, multi-head causal attention, MLP, LayerNorm, residuals, and unembed —
that loads real pretrained weights and generates coherent text. Correctness is
verified against HuggingFace GPT-2 (logits match to ~5e-5). The naive generation
loop deliberately has no KV cache yet, so its per-token cost grows with sequence
length — the motivation for Phase 2.

Model code lives in [`backend/chimera/model/`](backend/chimera/model/).
