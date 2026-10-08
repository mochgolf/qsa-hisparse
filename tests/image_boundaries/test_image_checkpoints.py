"""I3 items 1, 3 and 4: supported image requests checkpoint and restore at
every page64 boundary, including inside an image, through the host prefix
cache; PLE n-gram history over pad tokens is part of the restored state; a
hit leaves the logits tail and keeps the input-logprob limit.

The scaled CPU runtime mirrors tests/prefix (no CUDA, no TP group). Requests
reach the cache methods directly, in M01's order, so no hook is needed.
"""

from array import array
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, MambaPool, ReqToTokenPool
from sglang.srt.mem_cache.ple_state_pool import NGramPool, ShortConvPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.models import qwen4_exp
from sglang_qsa_hisparse.hisparse.prefix import HostPrefixCache
from sglang_qsa_hisparse.hisparse.prefix_cache import QSAHostPrefixCache
from sglang_qsa_hisparse.hisparse.runtime import QSAHiSparseRuntime
from sglang_qsa_hisparse.hisparse.slots import QSAHiSparseSlots

from image_boundaries.prompts import (
    BYPASS,
    STANDARD,
    VISION_END,
    VISION_START,
    VOCAB,
    identity_of,
    image_prompt,
)

NGRAM = 4  # The fixture's n-gram context holds NGRAM - 1 tokens.


class Event:
    def record(self, stream=None):
        self.recorded = True

    def synchronize(self):
        assert self.recorded

    def query(self):
        return self.recorded


class Host:
    """tests/prefix's scaled runtime: 2 rows, 256-token context, 2 layers."""

    def __init__(self):
        a = self.a = QSAHiSparseRuntime.__new__(QSAHiSparseRuntime)
        a.mode, a.device, a.strict, a.path = "p2-offload", "cpu", True, None
        a.capacity, a.max_requests, a.layer_ids = 256, 2, [3, 7]
        a.slots = QSAHiSparseSlots(a.capacity, 64, 2)
        raw = a.slots.raw_pool_size + 64
        a.full = SimpleNamespace(
            k_buffer=[torch.zeros((raw, 1, 256), dtype=torch.uint8) for _ in a.layer_ids],
            v_buffer=[torch.zeros((raw, 1, 256), dtype=torch.uint8) for _ in a.layer_ids],
        )
        a.pool = SimpleNamespace(
            dtype=torch.uint8,
            index_state_dtype=torch.bfloat16,
            qsa_key_state_buffer_pool=[
                torch.zeros((12, 1, 16), dtype=torch.bfloat16) for _ in a.layer_ids
            ],
            qsa_rope_position_buffer=torch.zeros((12, 3), dtype=torch.int64),
            qsa_compressed_k_buffer_pool=[
                torch.zeros((144, 1, 16), dtype=torch.bfloat16) for _ in a.layer_ids
            ],
        )
        a.pool.qsa_hisparse = a
        rp = a.req_pool = HybridReqToTokenPool.__new__(HybridReqToTokenPool)
        ReqToTokenPool.__init__(rp, 2, a.capacity, "cpu", False)
        rp.layer_transfer_counter = None  # Set by HybridReqToTokenPool.__init__.
        rp.enable_mamba_extra_buffer = False
        rp.req_index_to_mamba_index_mapping = torch.zeros(3, dtype=torch.int32)
        rp.mamba_allocator = MambaSlotAllocator(2, "cpu")
        mamba = rp.mamba_pool = MambaPool.__new__(MambaPool)
        mamba.replayssm_write_pos = None
        mamba.mamba_cache = MambaPool.State(
            conv=[torch.zeros((3, 3, 2, 3), dtype=torch.bfloat16)],
            temporal=torch.zeros((3, 3, 2, 2), dtype=torch.float32),
        )
        short = rp.short_conv_pool = ShortConvPool.__new__(ShortConvPool)
        short.conv_state = torch.zeros((2, 3, 2, 3), dtype=torch.bfloat16)
        short.layer_map = {3: 0, 7: 1}
        ngram = rp.ngram_pool = NGramPool.__new__(NGramPool)
        ngram.context = torch.zeros((3, NGRAM - 1), dtype=torch.int64)
        ngram.intermediate_context = None
        mamba._slot_siblings = (short, ngram)
        a.req_table = rp.req_to_token
        a.runner = SimpleNamespace(
            token_to_kv_pool_allocator=PagedTokenToKVPoolAllocator(
                512, 64, torch.uint8, "cpu", a.pool, False
            )
        )
        a.copy_stream, a.producer_stream = Mock(), Mock()
        a.host_slabs = torch.empty((2, 2, 64, 2048), dtype=torch.uint8)
        a._allocate_decode_workspace()
        a.requests, a.pending_releases, a.batch_requests = {}, [], []
        a.record = Mock()
        a.prefix_cache = HostPrefixCache(4_000_000)
        a.prefix_namespace = ("fixture-model", "normal-text-position=0", "TP2")
        params = CacheInitParams(True, rp, a.runner.token_to_kv_pool_allocator, 64)
        self.cache = QSAHostPrefixCache(ChunkCache(params), a, None)

    @property
    def allocator(self):
        return self.a.runner.token_to_kv_pool_allocator

    def req(self, name, prompt):
        return SimpleNamespace(
            rid=name,
            kv=ReqKvInfo(),
            cache_request_handle=CacheRequestHandle(name, 1),
            full_untruncated_fill_ids=array("q", prompt.ids),
            prefix_indices=torch.empty(0, dtype=torch.int64),
            multimodal_inputs=prompt.mm,
            extra_key=None,
            cache_salt=None,
            lora_id=None,
            last_node=None,
        )

    def match(self, req, limit=None):
        ids = req.full_untruncated_fill_ids
        result = self.cache.match_prefix(
            MatchPrefixParams(
                RadixKey(ids, limit=len(ids) - 1 if limit is None else limit), req=req
            )
        )
        req.prefix_indices = result.device_indices
        return len(req.prefix_indices)

    def ple(self, req, start, stop, *, commit):
        """Real Qwen4-Exp PLE n-gram layout for the extend chunk [start, stop)."""
        # embed_mm_inputs clamps forward_batch.input_ids before PLE reads them.
        ids = torch.tensor(req.full_untruncated_fill_ids[start:stop]).clamp(0, VOCAB - 1)
        batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=1,
            input_ids=ids,
            req_pool_indices=torch.tensor([req.kv.req_pool_idx]),
            seq_lens=torch.tensor([stop]),
            out_cache_loc=torch.arange(1, stop - start + 1),
            seq_lens_sum=stop,
            extend_seq_lens=torch.tensor([stop - start]),
            extend_seq_lens_cpu=[stop - start],
        )
        ple = qwen4_exp._prepare_ple_batch(
            ids, batch, ngram_size=NGRAM, ngram_eos_token_id=0
        )
        if commit:
            qwen4_exp._commit_ple_batch(ple, batch)
        return ple.ngram_context

    def cold(self, req, stops):
        """Uncached chunked prefill ending at each of ``stops``, capturing after each.

        Returns the live state at every stop: what the next chunk would read.
        """
        a = self.a
        a.req_pool.alloc([req])
        idx, mid = req.kv.req_pool_idx, req.kv.mamba_pool_idx
        pages = self.allocator.alloc(stops[-1])
        a.req_table[idx, : stops[-1]] = pages.int()
        req.prefix_indices = pages
        req.kv.kv_allocated_len = req.kv.kv_committed_len = stops[-1]
        state = a._acquire_request(idx, req.rid)
        ring = slice(idx * 4, (idx + 1) * 4)
        positions = req.multimodal_inputs.mrope_positions
        live, start = {}, 0
        for stop in stops:
            staging = a.slots.staging_slice(state.lease, start, stop)
            logical = a.req_table[idx, start:stop:4].long() // 4
            for li in range(len(a.layer_ids)):
                token = torch.arange(start, stop).reshape(-1, 1, 1)
                a.full.k_buffer[li][staging] = ((token + li * 31) % 251).to(torch.uint8)
                a.full.v_buffer[li][staging] = ((token * 3 + li) % 241).to(torch.uint8)
                compressed = torch.arange(start // 4, stop // 4).reshape(-1, 1, 1)
                a.pool.qsa_compressed_k_buffer_pool[li][logical] = (
                    compressed + li * 61
                ).to(torch.bfloat16)
                a.pool.qsa_key_state_buffer_pool[li][ring] = stop + li
            # The pending C4 ring keeps the M-RoPE rows of its last tokens.
            a.pool.qsa_rope_position_buffer[ring] = positions[:, stop - 4 : stop].T
            a.req_pool.mamba_pool.mamba_cache.conv[0][:, mid] = stop % 113
            a.req_pool.mamba_pool.mamba_cache.temporal[:, mid] = stop / 8
            a.req_pool.short_conv_pool.conv_state[:, mid] = stop % 37
            self.ple(req, start, stop, commit=True)
            state.seq_len = stop
            live[stop] = self.state(req, state.lease, stop)
            self.cache._capture(req)
            start = stop
        return live

    def state(self, req, lease, length):
        a, idx, mid = self.a, req.kv.req_pool_idx, req.kv.mamba_pool_idx
        staging = a.slots.staging_slice(lease, 0, length)
        logical = a.req_table[idx, :length:4].long() // 4
        ring = slice(idx * 4, (idx + 1) * 4)
        return [
            *(t[staging].clone() for t in a.full.k_buffer + a.full.v_buffer),
            *(t[logical].clone() for t in a.pool.qsa_compressed_k_buffer_pool),
            *(t[ring].clone() for t in a.pool.qsa_key_state_buffer_pool),
            a.pool.qsa_rope_position_buffer[ring].clone(),
            a.req_pool.mamba_pool.mamba_cache.conv[0][:, mid].clone(),
            a.req_pool.mamba_pool.mamba_cache.temporal[:, mid].clone(),
            a.req_pool.short_conv_pool.conv_state[:, mid].clone(),
            a.req_pool.ngram_pool.context[mid].clone(),
        ]

    def release(self, req):
        a = self.a
        lease = a.release(req.kv.req_pool_idx, req.rid)
        self.allocator.free(req.prefix_indices)
        a.req_pool.free_mamba_cache(req)
        a.req_pool.free(req)
        req.kv.mark_kv_released()
        a.after_release(lease)

    def poison(self):
        a = self.a
        for t in a.full.k_buffer + a.full.v_buffer:
            t.fill_(255)
        for t in a.pool.qsa_key_state_buffer_pool + a.pool.qsa_compressed_k_buffer_pool:
            t.fill_(-7)
        a.pool.qsa_rope_position_buffer.fill_(-1)
        a.req_pool.mamba_pool.mamba_cache.conv[0].fill_(-1)
        a.req_pool.mamba_pool.mamba_cache.temporal.fill_(-2)
        a.req_pool.short_conv_pool.conv_state.fill_(-3)
        a.req_pool.ngram_pool.context.fill_(-4)

    def restore(self, req):
        """alloc_for_extend's prefix steps (M01 order) for a matched request."""
        self.a.req_pool.alloc([req])
        self.cache.prepare_prefix_for_extend([req])
        length = len(req.prefix_indices)
        self.a.req_table[req.kv.req_pool_idx, :length] = req.prefix_indices.int()
        self.cache.restore_prefix_for_extend([req])
        return self.a.requests[req.kv.req_pool_idx].lease


@pytest.fixture
def host(monkeypatch):
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", Mock(return_value=Mock()))
    monkeypatch.setattr(
        "sglang.srt.mem_cache.memory_pool.current_platform.synchronize", Mock()
    )
    host = Host()
    monkeypatch.setattr(qwen4_exp, "get_req_to_token_pool", lambda: host.a.req_pool)
    return host


def test_namespace_admits_supported_images_and_keeps_every_bypass(host, identities):
    prompt = image_prompt(300)
    text = host.req("text", prompt)
    text.multimodal_inputs = None
    expected = (host.a.prefix_namespace, host.a.prefix_cache.epoch, None, None, None)
    assert host.cache._namespace(text) == expected

    image = host.req("image", prompt)
    assert host.cache._namespace(image) is None  # No identity: unsupported.
    identities["image"] = BYPASS
    assert host.cache._namespace(image) is None
    identities["image"] = identity_of(prompt)
    assert host.cache._namespace(image) == expected
    for name in (
        "positional_embed_overrides",
        "input_embeds",
        "position_ids",
        "mrope_positions",
        "session",
        "beam_group",
        "return_hidden_states",
    ):
        other = host.req("image-" + name, prompt)
        identities[other.rid] = identity_of(prompt)
        setattr(other, name, True if name == "return_hidden_states" else object())
        assert host.cache._namespace(other) is None, name


@pytest.mark.parametrize(
    "length, images, boundary",
    [
        (140, (("inside", 101, (1, 12, 12)),), 128),  # Image tokens 101..136.
        (200, (("starts", 192, (1, 4, 4)),), 192),  # Image tokens 192..195.
        (300, STANDARD, 256),  # After every image.
    ],
)
def test_checkpoint_limit_splits_image_requests_at_their_last_page(
    host, identities, length, images, boundary
):
    prompt = image_prompt(length, images)
    req = host.req("limit", prompt)
    assert host.cache.prefill_checkpoint_limit(req) is None  # Unsupported.
    identities["limit"] = identity_of(prompt)
    assert host.cache.prefill_checkpoint_limit(req) == boundary
    req.prefix_indices = torch.empty(boundary, dtype=torch.int64)
    assert host.cache.prefill_checkpoint_limit(req) is None
    identities["limit"] = BYPASS
    req.prefix_indices = torch.empty(0, dtype=torch.int64)
    assert host.cache.prefill_checkpoint_limit(req) is None
    aligned = image_prompt(192, STANDARD[:2])  # Complete final prefill instead.
    identities["aligned"] = identity_of(aligned)
    assert host.cache.prefill_checkpoint_limit(host.req("aligned", aligned)) is None


# Each hit request keeps the source's tokens and images before ``hit`` and
# diverges right after it, with a different image at or after ``hit``.
DIVERGENT = {
    64: (STANDARD[0], ("B2", 90, (1, 20, 20)), STANDARD[2]),
    128: STANDARD[:2] + (("D", 200, (1, 4, 4)),),  # Text at 191; B straddles 128.
    192: STANDARD[:2] + (("C2", 192, (1, 12, 12)),),  # C2 starts at the hit.
}


@pytest.mark.parametrize("hit", sorted(DIVERGENT))
def test_image_hit_restores_exact_state_at_every_page_including_inside_an_image(
    host, identities, hit
):
    source_prompt = image_prompt(300)
    source = host.req("source", source_prompt)
    identities["source"] = identity_of(source_prompt)
    live = host.cold(source, [64, 128, 192])
    assert sorted(s.length for s, _ in host.a.prefix_cache.entries.values()) == [
        64,
        128,
        192,
    ]
    host.release(source)
    host.poison()

    prompt = image_prompt(300, DIVERGENT[hit])
    req = host.req("divergent", prompt)
    identities["divergent"] = identity_of(prompt)
    key = identity_of(prompt).key_at(hit)
    assert key == identity_of(source_prompt).key_at(hit)
    straddling = [r.artifact_key for r in key[0] if r.start < hit < r.stop]
    assert straddling == (["B"] if hit == 128 else [])
    assert host.match(req) == hit
    lease = host.restore(req)
    restored = host.state(req, lease, hit)
    assert len(restored) == len(live[hit])
    for got, expected in zip(restored, live[hit]):
        assert torch.equal(got.view(torch.uint8), expected.view(torch.uint8))
    assert req.host_hit_length == req.host_loaded_length == hit
    assert host.a.prefix_cache.stats()["prefix_cache_readers"] == 0


def test_ple_ngram_history_over_image_pad_tokens_is_restored(host, identities):
    hit = 128  # Tokens 125..127 are B's pad tokens.
    source_prompt = image_prompt(300)
    source = host.req("source", source_prompt)
    identities["source"] = identity_of(source_prompt)
    host.cold(source, [64, hit])
    # The uncached chunked prefill's next chunk reads this history.
    control = host.ple(source, hit, 256, commit=False)
    pad = min(source_prompt.mm.mm_items[1].pad_value, VOCAB - 1)
    assert control[0, : NGRAM - 1].tolist() == [pad] * (NGRAM - 1)
    host.release(source)
    host.poison()

    warm_prompt = image_prompt(300, DIVERGENT[hit])
    warm = host.req("warm", warm_prompt)
    identities["warm"] = identity_of(warm_prompt)
    assert host.match(warm) == hit
    host.restore(warm)
    restored = host.ple(warm, hit, 256, commit=False)
    assert torch.equal(restored[:, : NGRAM - 1], control[:, : NGRAM - 1])

    # At 192 the history spans B's last pad token, <vision_end> and <vision_start>.
    host.release(warm)
    host.a.prefix_cache.reset()
    source = host.req("source-192", source_prompt)
    identities["source-192"] = identity_of(source_prompt)
    host.cold(source, [64, 128, 192])
    assert host.ple(source, 192, 256, commit=False)[0, : NGRAM - 1].tolist() == [
        pad,
        VISION_END,
        VISION_START,
    ]


def real_req(host, name, prompt, *, logprob_start_len=None):
    req = Req.__new__(Req)
    req.__dict__.update(host.req(name, prompt).__dict__)
    req.origin_input_ids, req.output_ids = array("q", prompt.ids), array("q")
    req.dllm_config = None
    req.return_logprob = logprob_start_len is not None
    req.logprob_start_len = -1 if logprob_start_len is None else logprob_start_len
    req.session = req.positional_embed_overrides = None
    req.is_retracted = False
    return req


def test_image_hit_leaves_the_logits_tail_and_keeps_logprob_limits(host, identities):
    # A page-aligned image prompt: A, then B (90..189), <vision_end>, one token.
    prompt = image_prompt(192, STANDARD[:2])
    source = host.req("source", prompt)
    identities["source"] = identity_of(prompt)
    host.cold(source, [64, 128, 192])  # The complete final prefill is captured.
    host.release(source)
    for start, expected in (
        (None, 128),  # The full 192-token snapshot would leave no logits tail.
        (-1, 128),
        (0, 0),
        (63, 0),
        (64, 64),
        (127, 64),
        (128, 128),
    ):
        req = real_req(host, f"logprobs-{start}", prompt, logprob_start_len=start)
        identities[req.rid] = identity_of(prompt)
        req.init_next_round_input(host.cache)
        assert len(req.prefix_indices) == expected, start
        assert host.cache.pending_prefix_tokens(req) == expected
        host.cache.release_aborted_request(req.cache_request_handle)
    assert host.a.prefix_cache.stats()["prefix_cache_readers"] == 0
