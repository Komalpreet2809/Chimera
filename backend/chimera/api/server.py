"""FastAPI server — streams tokens AND the engine's internals to the UI.

This is what makes Chimera an observability tool rather than a chatbot: every
websocket frame carries not just the token, but what the engine did to produce
it — latency, KV cache size, block allocations, scheduler state.

Endpoints:
  GET  /api/health
  GET  /api/config              model + engine configuration
  WS   /api/generate            live single-request generation + telemetry
  WS   /api/simulate            multi-user scheduler simulation + telemetry
  POST /api/attention           tokenization + attention weights (explorer)
  POST /api/benchmark/cache     naive vs KV-cached, measured live
  POST /api/benchmark/paged     contiguous reservation vs paged blocks
"""

from __future__ import annotations

import asyncio
import os
import time

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..engine.engine import InferenceEngine
from ..engine.scheduler import Scheduler
from ..memory.allocator import BlockAllocator
from ..memory.paged_cache import PagedKVPool
from ..model.config import GPT2_SMALL
from ..model.generate import SampleConfig
from ..model.kv_cache import KVCache
from ..model.tokenizer import Tokenizer
from ..model.weights import load_pretrained
from ..telemetry.metrics import MetricsCollector

app = FastAPI(title="Chimera", description="LLM inference, made visible")

# The deployed backend is a public read-only demo API, but there's no reason to
# let arbitrary origins call it — pin to the real frontends plus local dev.
_ALLOWED = [
    "https://chimera.komalpreet.me",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED,
    # Vercel preview deployments get a generated *.vercel.app hostname.
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)

# On a free CPU Space a naive 900-token forward pass takes seconds; cap the
# live benchmark so it can't wedge the box. Full-length runs stay available
# locally via backend/scripts/phase2_benchmark.py.
BENCH_MAX_SEQ = int(os.environ.get("CHIMERA_BENCH_MAX_SEQ", "900"))
BENCH_MAX_ITERS = int(os.environ.get("CHIMERA_BENCH_MAX_ITERS", "5"))

# ---- loaded once at startup ----
STATE: dict = {}


@app.on_event("startup")
def _load() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    STATE["device"] = device
    STATE["tokenizer"] = Tokenizer("gpt2")
    STATE["model"] = load_pretrained("gpt2").to(device)
    STATE["model"].eval()


def _engine(paged: bool, greedy: bool = False, temperature: float = 0.8) -> InferenceEngine:
    pool = PagedKVPool(GPT2_SMALL, num_blocks=1024, block_size=16) if paged else None
    return InferenceEngine(
        STATE["model"],
        STATE["tokenizer"],
        device=STATE["device"],
        sample_cfg=SampleConfig(greedy=greedy, temperature=temperature, top_k=40),
        kv_pool=pool,
    )


# ------------------------------------------------------------------ health
@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "device": STATE.get("device", "loading")}


@app.get("/api/config")
def config() -> dict:
    c = GPT2_SMALL
    return {
        "model": "gpt2",
        "device": STATE.get("device"),
        "n_layer": c.n_layer,
        "n_head": c.n_head,
        "n_embd": c.n_embd,
        "head_dim": c.head_dim,
        "vocab_size": c.vocab_size,
        "n_ctx": c.n_ctx,
    }


# ------------------------------------------------- live single generation
@app.websocket("/api/generate")
async def ws_generate(ws: WebSocket) -> None:
    """Stream one request's tokens with full per-token telemetry."""
    await ws.accept()
    try:
        req_msg = await ws.receive_json()
        prompt = req_msg.get("prompt", "The future of AI is")
        max_new = int(req_msg.get("max_new_tokens", 40))
        use_cache = bool(req_msg.get("use_cache", True))
        paged = bool(req_msg.get("paged", False))
        temperature = float(req_msg.get("temperature", 0.8))
        greedy = bool(req_msg.get("greedy", False))

        engine = _engine(paged, greedy, temperature)
        metrics = MetricsCollector()
        req = engine.submit(prompt, max_new_tokens=max_new)
        metrics.on_arrival(req.id, req.arrived_at)

        await ws.send_json({
            "type": "start",
            "request_id": req.id,
            "prompt_tokens": [
                {"id": i, "text": t} for i, t in STATE["tokenizer"].tokens(prompt)
            ],
            "use_cache": use_cache,
            "paged": paged,
        })

        if use_cache:
            while not req.is_finished:
                ev = engine.step(req)
                metrics.on_event(ev)
                await ws.send_json({
                    "type": "token",
                    "kind": ev.kind,
                    "token": ev.token_text,
                    "token_id": ev.token_id,
                    "latency_ms": round(ev.latency_ms, 2),
                    "cache_tokens": ev.cache_tokens,
                    "cache_bytes": ev.cache_bytes,
                    "index": ev.num_generated,
                    "blocks": (
                        list(req.kv_cache.table.physical_blocks)
                        if paged and req.kv_cache is not None
                        and hasattr(req.kv_cache, "table") else None
                    ),
                })
                await asyncio.sleep(0)  # let the event loop flush
        else:
            # the naive no-cache path — recomputes everything, visibly slower
            await _stream_uncached(ws, prompt, max_new, greedy, temperature, metrics)

        metrics.on_finish(req.id)
        await ws.send_json({"type": "done", "metrics": metrics.summary()})
    except WebSocketDisconnect:
        return
    except Exception as exc:  # surface errors to the UI instead of dying silently
        await ws.send_json({"type": "error", "message": str(exc)})


@torch.no_grad()
async def _stream_uncached(ws, prompt, max_new, greedy, temperature, metrics) -> None:
    """No KV cache: feed the whole sequence every step. Latency climbs visibly."""
    from ..engine.events import StepEvent
    from ..model.generate import _pick_next

    tok, model, device = STATE["tokenizer"], STATE["model"], STATE["device"]
    cfg = SampleConfig(greedy=greedy, temperature=temperature, top_k=40)
    ids = torch.tensor([tok.encode(prompt)], device=device)

    for i in range(max_new):
        t0 = time.perf_counter()
        logits, _ = model(ids)                       # THE CRIME: full recompute
        if device == "cuda":
            torch.cuda.synchronize()
        latency = (time.perf_counter() - t0) * 1000
        nxt = _pick_next(logits[0, -1], cfg)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)

        ev = StepEvent(
            request_id=0, kind="decode", token_id=nxt, token_text=tok.decode([nxt]),
            latency_ms=latency, cache_tokens=0, cache_bytes=0, num_generated=i + 1,
        )
        metrics.on_event(ev)
        await ws.send_json({
            "type": "token", "kind": "decode", "token": ev.token_text,
            "token_id": nxt, "latency_ms": round(latency, 2),
            "cache_tokens": 0, "cache_bytes": 0, "index": i + 1, "blocks": None,
        })
        await asyncio.sleep(0)


# ------------------------------------------------ multi-user simulation
@app.websocket("/api/simulate")
async def ws_simulate(ws: WebSocket) -> None:
    """Run N users through the scheduler, streaming every tick's state."""
    await ws.accept()
    try:
        msg = await ws.receive_json()
        prompts: list[str] = msg.get("prompts") or ["The future of AI is"] * 8
        lengths: list[int] = msg.get("lengths") or [20] * len(prompts)
        policy = msg.get("policy", "continuous")
        batch_size = int(msg.get("max_batch_size", 4))
        paged = bool(msg.get("paged", True))

        engine = _engine(paged, greedy=True)
        sched = Scheduler(engine, max_batch_size=batch_size, policy=policy)
        metrics = MetricsCollector()

        for p, n in zip(prompts, lengths):
            r = sched.submit(p, max_new_tokens=n)
            metrics.on_arrival(r.id, r.arrived_at)

        await ws.send_json({
            "type": "start", "policy": policy, "max_batch_size": batch_size,
            "num_requests": len(prompts), "paged": paged,
        })

        tick = 0
        while sched.has_work:
            before = {r.id for r in sched.seated}
            sched.tick()
            tick += 1
            after = {r.id for r in sched.seated}

            pool_stats = engine.kv_pool.stats() if engine.kv_pool else None
            await ws.send_json({
                "type": "tick",
                "tick": tick,
                "seated": [
                    {"id": r.id, "tokens": r.num_generated, "target": r.max_new_tokens}
                    for r in sched.seated
                ],
                "queued": len(sched.queue),
                "completed": len(sched.done),
                "admitted": sorted(after - before),
                "evicted": sorted(before - after),
                "capacity": sched.capacity,
                "utilization": len(sched.seated) / sched.capacity if sched.capacity else 0,
                "pool": pool_stats,
            })
            await asyncio.sleep(0)

        await ws.send_json({
            "type": "done",
            "stats": sched.stats.snapshot(sched.capacity, sched.done),
            "outputs": [
                {"id": r.id, "text": engine.text_of(r)} for r in sched.done
            ],
        })
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await ws.send_json({"type": "error", "message": str(exc)})


# ------------------------------------------------- transformer explorer
class AttentionReq(BaseModel):
    text: str = "The animal didn't cross the street because it was too tired"
    layer: int = 0
    head: int = 0


@app.post("/api/attention")
@torch.no_grad()
def attention(req: AttentionReq) -> dict:
    """Tokenize + return one head's attention weights — the heatmap lens."""
    tok, model, device = STATE["tokenizer"], STATE["model"], STATE["device"]
    pieces = tok.tokens(req.text)
    ids = torch.tensor([[i for i, _ in pieces]], device=device)

    _, attns = model(ids, return_attn=True)
    layer = max(0, min(req.layer, len(attns) - 1))
    head = max(0, min(req.head, GPT2_SMALL.n_head - 1))
    weights = attns[layer][0, head].cpu().tolist()   # (seq, seq)

    return {
        "tokens": [{"id": i, "text": t} for i, t in pieces],
        "layer": layer,
        "head": head,
        "n_layer": GPT2_SMALL.n_layer,
        "n_head": GPT2_SMALL.n_head,
        "attention": weights,
    }


# ------------------------------------------------------------ benchmarks
class CacheBenchReq(BaseModel):
    seq_lens: list[int] = [64, 128, 256, 512]
    # CPU by default, deliberately. See the note below.
    device: str = "cpu"
    iters: int = 5


@app.post("/api/benchmark/cache")
@torch.no_grad()
def bench_cache(req: CacheBenchReq) -> dict:
    """Per-step decode latency, naive vs KV-cached, at growing sequence lengths.

    Why the default is CPU, not the GPU:

    The KV cache saves *compute* — it stops us re-processing the whole sequence
    every step. You can only observe that saving on hardware where compute is
    actually the bottleneck.

    On a small model like GPT-2 (124M) on a laptop GPU, it isn't. Each op is so
    tiny that the GPU never leaves its idle power state (measured: 210 MHz of a
    3105 MHz clock, 1.65W), so every step is dominated by fixed kernel-launch
    overhead at a throttled clock — ~80ms regardless of sequence length. The
    algorithmic difference is real but completely masked by that floor.

    On CPU the work is compute-bound, so the O(N)-per-step vs O(1)-per-step
    difference shows up exactly as the theory predicts. Both are honest
    measurements; only one of them is measuring the thing we're asking about.
    """
    model = STATE["model"]
    device = "cuda" if (req.device == "cuda" and torch.cuda.is_available()) else "cpu"
    model = model.to(device)
    one = torch.zeros(1, 1, dtype=torch.long, device=device)

    # Clamp the requested work — a public endpoint shouldn't let a caller ask
    # for a 4000-token naive forward pass on a 2-vCPU box.
    seq_lens = sorted({n for n in req.seq_lens if 1 <= n <= BENCH_MAX_SEQ})[:6]
    iters = max(1, min(req.iters, BENCH_MAX_ITERS))

    def sync() -> None:
        if device == "cuda":
            torch.cuda.synchronize()

    def median_ms(fn, iters: int) -> float:
        # Median, not mean: a single scheduler hiccup on a busy laptop can skew
        # a mean badly (we saw a 3-sample mean report an impossible 220x).
        samples = []
        for _ in range(iters):
            t = time.perf_counter()
            fn()
            sync()
            samples.append((time.perf_counter() - t) * 1000)
        samples.sort()
        return samples[len(samples) // 2]

    rows = []
    for n in seq_lens:
        seq = torch.zeros(1, n, dtype=torch.long, device=device)

        # Warm up at THIS shape — the first call at a new shape pays one-time
        # allocation/planning cost that would otherwise pollute the timing.
        for _ in range(2):
            model(seq)
        sync()
        naive = median_ms(lambda: model(seq), iters)

        cache = KVCache(GPT2_SMALL)
        model(seq, cache=cache)          # prefill (not timed — happens once)
        for _ in range(2):
            model(one, cache=cache)
        sync()
        cached = median_ms(lambda: model(one, cache=cache), iters + 3)

        rows.append({
            "seq_len": n,
            "naive_ms": round(naive, 2),
            "cached_ms": round(cached, 2),
            "speedup": round(naive / cached, 1) if cached else 0,
        })

    # restore the serving device so generation stays fast
    model.to(STATE["device"])
    return {"device": device, "rows": rows}


class PagedBenchReq(BaseModel):
    budget_mb: int = 512
    max_seq: int = 512
    block_size: int = 16
    lengths: list[int] | None = None


@app.post("/api/benchmark/paged")
def bench_paged(req: PagedBenchReq) -> dict:
    """Worst-case contiguous reservation vs on-demand paged blocks."""
    KB_PER_TOKEN = 72
    lengths = req.lengths or ([12, 340, 25, 60, 8, 480, 33, 15, 120, 44,
                               19, 200, 7, 95, 28, 410, 51, 16, 73, 22] * 5)
    budget_kb = req.budget_mb * 1024

    reserve_kb = req.max_seq * KB_PER_TOKEN
    naive_fit = int(budget_kb // reserve_kb)
    served = lengths[:naive_fit]
    naive_used = sum(served) * KB_PER_TOKEN
    naive_reserved = max(len(served) * reserve_kb, 1)

    kb_per_block = req.block_size * KB_PER_TOKEN
    alloc = BlockAllocator(int(budget_kb // kb_per_block), req.block_size)
    tables = []
    for n in lengths:
        t = alloc.new_table()
        if not alloc.can_append(t, n):
            break
        alloc.append_tokens(t, n)
        tables.append(t)

    paged_used = sum(t.num_tokens for t in tables) * KB_PER_TOKEN
    paged_alloc = max(sum(t.num_blocks for t in tables) * kb_per_block, 1)

    return {
        "budget_mb": req.budget_mb,
        "workload": len(lengths),
        "naive": {
            "fit": naive_fit,
            "used_mb": round(naive_used / 1024, 1),
            "reserved_mb": round(naive_reserved / 1024, 1),
            "waste": round(1 - naive_used / naive_reserved, 3),
        },
        "paged": {
            "fit": len(tables),
            "used_mb": round(paged_used / 1024, 1),
            "allocated_mb": round(paged_alloc / 1024, 1),
            "waste": round(1 - paged_used / paged_alloc, 3),
            "num_blocks": alloc.num_blocks,
        },
    }
