# Chimera

**An LLM inference engine you can see inside.**

Chimera is a from-scratch LLM serving engine — hand-written transformer, KV cache,
continuous-batching scheduler, PagedAttention, and speculative decoding — wrapped in a
UI that streams the engine's *internals* alongside its tokens. Not a chatbot: an
instrument.

> If Wireshark is for networks and DevTools is for browsers, Chimera is for LLM inference.

Modern LLM tools hide *why* generation is slow, why memory explodes, and what techniques
like KV caching, continuous batching, and PagedAttention actually do. Chimera exposes all
of it — every number below is measured live by the engine, not animated.

## Results

Every figure is reproducible from this repo.

| | Result | Reproduce |
|---|---|---|
| **Correctness** | Hand-built GPT-2 matches HuggingFace logits to **5e-5** | `pytest backend/tests` |
| **KV cache** | Per-step decode **4× → 37× faster** as sequences grow (402ms → 4574ms naive vs a flat ~90-124ms cached), output **token-identical** | `phase2_benchmark.py` |
| **Continuous batching** | **3.7× throughput** (9.8 → 36.2 tok/s), **7.6× faster** time-to-first-token (25.8s → 3.4s), seat utilization 43% → 80% vs static | `phase4_showdown.py` |
| **PagedAttention** | **4.6× more concurrent users** in the same 512MB budget (14 → 65); memory waste **80% → 6.2%** | `phase5_paged_memory.py` |
| **Speculative decoding** | 65–86% draft acceptance, **3.4–4.0× fewer** big-model passes, output identical to GPT-2 alone | `phase6_speculative.py` |

Every optimization above is **free** — it changes speed or memory, never the tokens. That
claim is enforced by tests, not asserted: cached ≡ naive, batched ≡ solo, paged ≡
contiguous, speculative ≡ greedy.

### Two honest findings

Numbers that flatter a project are easy; these two don't, and they're the more
interesting ones.

**Speculative decoding barely helps here — and that's the correct result.** We cut
big-model forward passes by ~4×, but wall-clock only improves 1.0–1.3×. Why: distilgpt2
is only ~2× cheaper than gpt2, so the draft model's own passes eat the savings.
Speculation pays off when the target *dwarfs* the draft (a 70B verified by a 1B), not at
GPT-2 scale. The implementation is correct; the technique simply isn't profitable in this
regime.

**The KV cache benchmark runs on CPU on purpose.** The cache saves *compute*, so it's only
visible where compute is the bottleneck. On a laptop GPU it isn't: GPT-2's ops are so
small the card never leaves its idle power state (measured **210 MHz of a 3105 MHz clock,
1.65 W**), so every decode step costs a flat ~80ms of kernel-launch overhead regardless of
sequence length — completely masking the algorithmic win. On CPU the work is compute-bound
and the O(N)→O(1) difference appears exactly as theory predicts.

## What you can do with it

- **Inference Lab** — generate text and watch per-token latency, KV-cache growth, and
  block allocation in real time. Toggle the KV cache off and watch the cost of
  recomputing the whole sequence every step.
- **Scheduler** — 12 users arrive at once; race *sequential* vs *static* vs *continuous*
  batching and watch seats fill, stragglers strand capacity, and queued requests slot into
  freed seats mid-flight.
- **Paged Memory** — a live memory map of naive worst-case reservation (a sea of wasted
  red) versus on-demand paged blocks.
- **Attention** — browse all 144 attention heads of the running model; the causal mask is
  visible as the empty upper triangle.
- **Benchmarks** — run the engine's benchmarks live on your own machine.

## Architecture

```
backend/chimera/
  model/       hand-written GPT-2: embedding, multi-head causal attention,
               MLP, LayerNorm, residuals, unembed, sampling, KV cache
  engine/      Request lifecycle, step()-based InferenceEngine, batched decode,
               continuous-batching Scheduler, speculative decoding
  memory/      PagedAttention: BlockAllocator, BlockTable (address translation),
               PagedKVPool
  telemetry/   StepEvent aggregation -> TTFT, TPOT, throughput
  api/         FastAPI + WebSocket streaming of tokens *and* engine internals

frontend/      Next.js + TypeScript + Tailwind UI (the five lenses above)
```

The engine is built from first principles — no `generate()` black boxes. Every module
loads real pretrained GPT-2 weights, so output is coherent *and* every tensor is
accounted for.

## Running it

```bash
# backend  (http://127.0.0.1:8000)
cd backend
pip install -r requirements.txt
uvicorn chimera.api.server:app --reload

# frontend (http://localhost:3000)
cd frontend
npm install && npm run dev
```

Runs CPU-first, GPU-ready (CUDA auto-detected).

```bash
# the correctness suite — every "this is free" claim, enforced
pytest backend/tests -q

# the benchmarks behind the table above
python backend/scripts/phase2_benchmark.py    # KV cache
python backend/scripts/phase4_showdown.py     # batching policies
python backend/scripts/phase5_paged_memory.py # paged vs contiguous
python backend/scripts/phase6_speculative.py  # speculative decoding
```

## How it was built

Each phase was understood from first principles before a line of it was written:

0. Foundations — why generation is a loop, and why that loop is O(N²)
1. The transformer — embedding, Q/K/V attention, causal masking, MLP, LayerNorm
2. KV cache — what's safe to reuse, and what it costs in memory
3. Inference engine — requests, lifecycles, and `step()` as the unit of progress
4. Scheduling — why the GPU is bored, and how continuous batching fills it
5. PagedAttention — the operating system's answer to memory fragmentation
6. Speculative decoding — trading cheap guesses for expensive verification
7. Telemetry — making every step observable
8. Frontend — turning the event stream into something you can watch
9. Tests, docs, and honest reporting
