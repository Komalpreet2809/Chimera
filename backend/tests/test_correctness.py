"""The correctness invariants that hold this project together.

Every optimization in Chimera claims to be *free* — faster or smaller, but
producing exactly the same tokens. These tests are what make that claim
checkable rather than a hope:

  1. our from-scratch GPT-2  ==  HuggingFace GPT-2          (the model is right)
  2. KV-cached decode        ==  naive full recompute       (Phase 2 is free)
  3. batched decode          ==  one-at-a-time decode       (Phase 4 is free)
  4. paged KV cache          ==  contiguous KV cache        (Phase 5 is free)
  5. speculative decoding    ==  plain greedy decoding      (Phase 6 is free)

Run:  pytest backend/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chimera.engine.engine import InferenceEngine
from chimera.engine.scheduler import Scheduler
from chimera.memory.allocator import BlockAllocator, OutOfMemoryError
from chimera.memory.paged_cache import PagedKVPool
from chimera.model.config import GPT2_SMALL, GPTConfig
from chimera.model.generate import SampleConfig
from chimera.model.kv_cache import KVCache
from chimera.model.tokenizer import Tokenizer
from chimera.model.weights import load_pretrained

PROMPT = "The history of artificial intelligence"


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer("gpt2")


@pytest.fixture(scope="module")
def model():
    return load_pretrained("gpt2")


def greedy(model, ids: list[int], n: int, cache=None) -> list[int]:
    """Greedy decode n tokens. With a cache: prefill then one token per step."""
    out: list[int] = []
    with torch.no_grad():
        if cache is None:
            seq = list(ids)
            for _ in range(n):
                logits, _ = model(torch.tensor([seq]))
                nxt = int(logits[0, -1].argmax())
                seq.append(nxt)
                out.append(nxt)
        else:
            logits, _ = model(torch.tensor([ids]), cache=cache)
            nxt = int(logits[0, -1].argmax())
            out.append(nxt)
            for _ in range(n - 1):
                logits, _ = model(torch.tensor([[nxt]]), cache=cache)
                nxt = int(logits[0, -1].argmax())
                out.append(nxt)
    return out


# ---------------------------------------------------------------- 1. the model
def test_matches_huggingface(model, tok):
    """Our hand-built GPT-2 must produce the same logits as the real thing."""
    from transformers import GPT2LMHeadModel

    ids = torch.tensor([tok.encode("The cat sat on the mat and looked at")])
    hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    with torch.no_grad():
        ours, _ = model(ids)
        theirs = hf(ids).logits
    assert torch.allclose(ours, theirs, atol=1e-3)


def test_causal_mask_blocks_the_future(model, tok):
    """No token may attend to a token after it — the rule generation depends on."""
    ids = torch.tensor([tok.encode("The cat sat on the mat")])
    with torch.no_grad():
        _, attns = model(ids, return_attn=True)
    a = attns[0][0, 0]                     # layer 0, head 0: (seq, seq)
    seq = a.size(0)
    for i in range(seq):
        for j in range(i + 1, seq):        # strictly upper triangle = the future
            assert a[i, j].item() == 0.0
        assert a[i].sum().item() == pytest.approx(1.0, abs=1e-4)


# --------------------------------------------------------------- 2. KV cache
def test_kv_cache_is_free(model, tok):
    """Cached decode must produce the identical tokens to full recompute."""
    ids = tok.encode(PROMPT)
    assert greedy(model, ids, 12) == greedy(model, ids, 12, KVCache(GPT2_SMALL))


def test_cache_grows_one_per_step(model, tok):
    ids = tok.encode(PROMPT)
    cache = KVCache(GPT2_SMALL)
    with torch.no_grad():
        model(torch.tensor([ids]), cache=cache)
        assert cache.seq_len == len(ids)          # prefill fills it in one pass
        model(torch.tensor([[ids[-1]]]), cache=cache)
        assert cache.seq_len == len(ids) + 1      # decode adds exactly one


# ---------------------------------------------------------------- 3. batching
def test_batched_decode_is_free(tok, model):
    """A request must get the same tokens whether it rides alone or in a batch.

    Prompts of different lengths on purpose: that exercises the left-padding of
    the stacked caches and the per-row position offsets, which is exactly where
    a batching bug would hide.
    """
    engine = InferenceEngine(model, tok, device="cpu", sample_cfg=SampleConfig(greedy=True))
    prompts = ["The best way to learn is", "In the distant future, humanity will", "AI"]

    solo = []
    for p in prompts:
        r = engine.submit(p, max_new_tokens=10)
        while not r.is_finished:
            engine.step(r)
        solo.append(r.generated_ids)

    reqs = [engine.submit(p, max_new_tokens=10) for p in prompts]
    for r in reqs:
        engine.step(r)                    # prefill each
    live = [r for r in reqs if not r.is_finished]
    while live:
        engine.step_decode_batch(live)    # ONE forward pass for all rows
        live = [r for r in live if not r.is_finished]

    assert [r.generated_ids for r in reqs] == solo


# ------------------------------------------------------------- 4. paged memory
def test_paged_cache_is_free(model, tok):
    """Scattering the cache across physical blocks must not change the output."""
    ids = tok.encode(PROMPT)
    pool = PagedKVPool(GPT2_SMALL, num_blocks=64, block_size=16)
    assert greedy(model, ids, 12) == greedy(model, ids, 12, pool.new_cache())


def test_blocks_are_allocated_on_demand_and_returned():
    alloc = BlockAllocator(num_blocks=4, block_size=16)
    table = alloc.new_table()

    alloc.append_tokens(table, 1)          # 1 token -> 1 block
    assert table.num_blocks == 1
    alloc.append_tokens(table, 15)         # fills that block exactly
    assert table.num_blocks == 1
    alloc.append_tokens(table, 1)          # spills into a second
    assert table.num_blocks == 2
    assert alloc.num_allocated == 2

    freed = alloc.free_table(table)        # request finishes
    assert len(freed) == 2
    assert alloc.num_allocated == 0        # every block returned to the pool


def test_allocator_refuses_to_oversubscribe():
    alloc = BlockAllocator(num_blocks=2, block_size=16)
    t = alloc.new_table()
    assert alloc.can_append(t, 32)         # exactly fits
    assert not alloc.can_append(t, 33)     # one token too many
    alloc.append_tokens(t, 32)
    with pytest.raises(OutOfMemoryError):
        alloc.allocate_block()


def test_paged_pool_frees_when_request_finishes(model, tok):
    """A finished request must hand its blocks back — or we leak the GPU away."""
    pool = PagedKVPool(GPT2_SMALL, num_blocks=64, block_size=16)
    engine = InferenceEngine(
        model, tok, device="cpu", sample_cfg=SampleConfig(greedy=True), kv_pool=pool
    )
    sched = Scheduler(engine, max_batch_size=2, policy="continuous")
    for p in ["The future of AI is", "Once upon a time", "Python is a"]:
        sched.submit(p, max_new_tokens=6)

    while sched.has_work:
        sched.tick()

    assert len(sched.done) == 3
    assert pool.allocator.num_allocated == 0    # nothing leaked


# --------------------------------------------------------- 5. speculative
def test_speculative_decoding_is_free(model, tok):
    """Speculation must buy speed, never a different answer."""
    from chimera.engine.speculative import speculative_generate

    draft = load_pretrained("distilgpt2", GPTConfig(n_layer=6))
    ids = tok.encode("The capital of France is")

    plain = greedy(model, ids, 24, KVCache(GPT2_SMALL))
    spec, stats = speculative_generate(
        model, draft, ids, max_new_tokens=24, lookahead=4, device="cpu"
    )
    assert spec == plain
    # and it really did save big-model passes, not just get lucky
    assert stats.target_passes < 24
    assert 0.0 <= stats.acceptance_rate <= 1.0


# ------------------------------------------------------------- 6. scheduler
def test_continuous_batching_beats_static_on_utilization(model, tok):
    """Stragglers must not hold empty seats hostage under continuous batching."""
    lengths = [2, 20, 3, 18, 4]          # deliberately unequal -> stragglers
    prompts = ["The future of AI is"] * len(lengths)

    def run(policy: str) -> float:
        engine = InferenceEngine(
            model, tok, device="cpu", sample_cfg=SampleConfig(greedy=True)
        )
        sched = Scheduler(engine, max_batch_size=2, policy=policy)
        for p, n in zip(prompts, lengths):
            sched.submit(p, max_new_tokens=n)
        return sched.run_to_completion()["utilization"]

    assert run("continuous") > run("static")
