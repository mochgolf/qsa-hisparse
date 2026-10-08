"""Model runner coordinator, ForwardBatch metadata and decode graph hooks.

Rows R01, R02 (``model_executor/model_runner.py``), F01
(``model_executor/forward_batch_info.py``, deviation D3) and G01-G03
(``model_executor/runner/decode_cuda_graph_runner.py``). All are hisparse: every
fork change is gated on a ``qsa_hisparse`` runtime on the KV pool or on the
coordinator's ``adapter``, which upstream ``HiSparseCoordinator`` lacks.
REPLACE bodies are the pinned definitions with the fork's change carried over
(PLAN.md Phase 4 rule P1); they run with this module's globals, which import
the same objects the pinned modules use.
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
from typing import Callable, Optional, Union, cast

from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import SharedReadEnds
from sglang.srt.layers.dp_attention import set_dp_buffer_len, set_is_extend_in_batch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    has_forward_context,
)
from sglang.srt.model_executor.model_runner import (
    ModelRunnerOutput,
    _prefill_cuda_graph_allows_context_parallel,
)
from sglang.srt.model_executor.runner import EagerRunner
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    build_replay_fb_view,
    logger,
)
from sglang.srt.model_executor.runner.flashinfer_autotune import (
    maybe_flashinfer_autotune_speculative_draft,
)
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import FullCudaGraphBackend
from sglang.srt.multiplex.pdmux_context import get_current_stream_idx
from sglang.srt.runtime_context import get_exec, get_global_dwdp_manager
from sglang.srt.speculative.ragged_verify import resolve_ragged_verify_layout
from sglang.srt.utils import empty_context
from sglang.srt.utils.device_timer import device_timer_ctx
from sglang_qsa_hisparse.features import HISPARSE
from sglang_qsa_hisparse.patching import patch

_RUNNER = "sglang.srt.model_executor.model_runner.ModelRunner"
_GRAPH = "sglang.srt.model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner"
_GRAPH_MODULE = "sglang.srt.model_executor.runner.decode_cuda_graph_runner"
_FULL_BACKEND = (
    "sglang.srt.model_executor.runner_backend.full_cuda_graph_backend.FullCudaGraphBackend"
)


# F01 ---------------------------------------------------------------------------


@patch(
    "sglang.srt.model_executor.forward_batch_info.ForwardBatch.init_new",
    "after",
    feature=HISPARSE,
    row="F01",
    depends=(
        "sglang.srt.managers.schedule_batch.ScheduleBatch",
        "sglang.srt.model_executor.cuda_graph_buffer_registry.CudaGraphBufferRegistry.extract_buffer",
        "sglang.srt.model_executor.cuda_graph_buffer_registry.build_decode_registry",
        "sglang.srt.model_executor.cuda_graph_buffer_registry.build_eager_registry",
        "sglang.srt.model_executor.forward_batch_info.ForwardBatch",
        "sglang.srt.model_executor.runner.eager_runner.EagerRunner.load_batch",
    ),
    reason=(
        "Deviation D3: the fork adds ForwardBatch fields req_pool_indices_cpu "
        "(read by the runtime's begin_batch) and kv_allocated_lens_cpu (no reader; "
        "dropped) and fills req_pool_indices_cpu in init_new for every batch. "
        "v0.5.21 declares the req_pool_indices_cpu field and fills it only for "
        "extend without speculative decoding. When the KV pool carries a "
        "qsa_hisparse runtime this hook sets the field from the ScheduleBatch for "
        "every mode (for that extend case it is the object upstream already set); "
        "otherwise the field stays exactly as upstream sets it. The after hook "
        "covers every return of init_new, and nothing in init_new reads the field "
        "after constructing the batch. EagerRunner.load_batch's copy "
        "(dataclasses.replace directly or in extract_buffer, whose eager registry "
        "from build_eager_registry/build_decode_registry has no "
        "req_pool_indices_cpu slot) carries the field, so the eager paths hand it "
        "to init_forward_metadata without a second hook."
    ),
)
def _carry_req_pool_indices_cpu(ret, cls, batch, model_runner, *args, **kwargs):
    if getattr(model_runner.token_to_kv_pool, "qsa_hisparse", None) is not None:
        ret.req_pool_indices_cpu = batch.req_pool_indices_cpu


# R01 ---------------------------------------------------------------------------


@patch(
    f"{_RUNNER}.init_attention_backends",
    "after",
    feature=HISPARSE,
    row="R01",
    depends=(f"{_RUNNER}._prepare_replicated_q_proj",),
    reason=(
        "Fork creates QSAHiSparseCoordinator in init_attention_backends when the "
        "KV pool's runtime uses QSA leases, before the DCP q_proj tail; that tail "
        "(_prepare_replicated_q_proj) does not touch the coordinator, so an after "
        "hook is equivalent. Lines copied from the fork; mechanical edit: the "
        "coordinator import is rewritten to the plugin package. hisparse: no "
        "upstream pool carries qsa_hisparse."
    ),
)
def _attach_qsa_coordinator(result, self):
    qsa = getattr(self.token_to_kv_pool, "qsa_hisparse", None)
    if getattr(qsa, "uses_qsa_hisparse_leases", False):
        from sglang_qsa_hisparse.hisparse.coordinator import QSAHiSparseCoordinator

        self.hisparse_coordinator = QSAHiSparseCoordinator(
            qsa, self.tp_group.cpu_group
        )


# R02 ---------------------------------------------------------------------------


@patch(
    f"{_RUNNER}._forward_raw",
    "replace",
    feature=HISPARSE,
    row="R02",
    depends=(
        f"{_GRAPH}.can_run_graph",
        f"{_GRAPH}.execute",
        f"{_RUNNER}._extend_forward_kwargs",
        f"{_RUNNER}._maybe_execute_deferred_mamba_cow_and_clear",
        f"{_RUNNER}._prepare_eager_forward_batch",
        f"{_RUNNER}.forward_split_prefill",
        "sglang.srt.model_executor.model_runner.ModelRunnerOutput",
        "sglang.srt.model_executor.model_runner._prefill_cuda_graph_allows_context_parallel",
        "sglang.srt.model_executor.forward_context.ForwardContext",
        "sglang.srt.model_executor.forward_context.forward_context",
        "sglang.srt.model_executor.forward_context.has_forward_context",
        "sglang.srt.runtime_context.get_global_dwdp_manager",
        "sglang.srt.utils.device_timer.device_timer_ctx",
    ),
    reason=(
        "Pinned body with the fork change carried over, no edits: raise when QSA "
        "full-graph decode cannot run the graph (needs the local can_run_graph) "
        "and skip num_real_reqs.fill_ when the QSA graph is enabled (a "
        "mid-function statement; no hook can drop it without cross-call state). "
        "hisparse: gated on hisparse_coordinator.adapter, which upstream "
        "HiSparseCoordinator lacks."
    ),
)
def _forward_raw(
    self,
    forward_batch: ForwardBatch,
    pp_proxy_tensors: Optional[PPProxyTensors],
    reinit_attn_backend: bool = False,
    split_forward_count: int = 1,
) -> ModelRunnerOutput:
    if has_forward_context():
        ctx_mgr = contextlib.nullcontext()
    else:
        ctx_mgr = forward_context(ForwardContext(attn_backend=self.attn_backend))
    with ctx_mgr:
        mode_check = (
            forward_batch.forward_mode.is_cpu_graph
            if self.device == "cpu"
            else forward_batch.forward_mode.is_cuda_graph
        )
        can_run_graph = bool(
            mode_check()
            and self.decode_cuda_graph_runner
            and self.decode_cuda_graph_runner.can_run_graph(forward_batch)
        )
        qsa = getattr(self.hisparse_coordinator, "adapter", None)
        if (
            forward_batch.forward_mode.is_decode()
            and getattr(qsa, "graph_enabled", False)
            and not can_run_graph
        ):
            raise RuntimeError("QSA full graph decode cannot fall back to eager")

        if (
            forward_batch.forward_mode.is_decode()
            and self.hisparse_coordinator is not None
        ):
            forward_batch.hisparse_coordinator = self.hisparse_coordinator
            self.hisparse_coordinator.wait_for_pending_backup()
            if not getattr(qsa, "graph_enabled", False):
                self.hisparse_coordinator.num_real_reqs.fill_(
                    forward_batch.batch_size
                )

        # Replay cuda graph if applicable
        if can_run_graph:
            ret = self.decode_cuda_graph_runner.execute(
                forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
            )
            return ModelRunnerOutput(logits_output=ret, can_run_graph=can_run_graph)

        # DP / MLP-sync padding + attn-tp normalization. Only the decode
        # cuda-graph path above pre-pads its static buffers and returns
        # early; split prefill, the prefill cuda graph, and the eager
        # forward all run the live batch and need this first — it sets
        # global_dp_buffer_len / padded token counts that graph eligibility
        # and the collectives depend on.
        self._prepare_eager_forward_batch(forward_batch)

        # Deferred mamba COW/clear on the forward stream, before the extend
        # dispatch below reads the pool.
        self._maybe_execute_deferred_mamba_cow_and_clear(forward_batch)

        dwdp_mgr = get_global_dwdp_manager()
        if dwdp_mgr is not None:
            dwdp_mgr.prefetch_first_layers()

        if forward_batch.forward_mode.is_split_prefill():
            # Layer-split mode; stays on ModelRunner, not the eager runner.
            ret = self.forward_split_prefill(
                forward_batch,
                reinit_attn_backend=reinit_attn_backend,
                forward_count=split_forward_count,
            )
        elif (
            forward_batch.forward_mode.is_extend(include_draft_extend_v2=True)
            and not isinstance(self.prefill_cuda_graph_runner, EagerRunner)
            and self.prefill_cuda_graph_runner is not None
            and self.prefill_cuda_graph_runner.can_run_graph(forward_batch)
            and forward_batch.token_indices_to_pool is None
            and _prefill_cuda_graph_allows_context_parallel(
                self.prefill_cuda_graph_runner, forward_batch
            )
        ):
            # Prefill cuda graph (piecewise).
            kwargs = self._extend_forward_kwargs(forward_batch, pp_proxy_tensors)
            category = (
                "target_verify"
                if forward_batch.forward_mode.is_target_verify()
                else "extend"
            )
            # TODO: the timing here is too broad -- it also includes
            # load_batch time. Move it into the prefill cuda graph runner
            # to capture only the model.forward part.
            with device_timer_ctx(self.device_timer, category):
                ret = self.prefill_cuda_graph_runner.execute(
                    forward_batch, **kwargs
                )
            can_run_graph = True
        else:
            # Eager: decode / extend / idle dispatched inside the runner.
            ret = self.eager_runner.execute(
                forward_batch, pp_proxy_tensors=pp_proxy_tensors
            )

        if (
            forward_batch.global_num_tokens_cpu is not None
            and self.pp_group.is_last_rank
        ):
            forward_batch.post_forward_mlp_sync_batch(ret)

        return ModelRunnerOutput(logits_output=ret, can_run_graph=can_run_graph)


# G01-G03 -----------------------------------------------------------------------


@patch(
    f"{_GRAPH}.capture_one_shape",
    "replace",
    feature=HISPARSE,
    row="G01",
    depends=(
        f"{_GRAPH}._capture_graph_size",
        f"{_GRAPH}._make_graph_key",
        f"{_GRAPH}._ragged_capture_slots",
        f"{_GRAPH}._record_in_graph_metadata_prep_done",
        f"{_GRAPH}.capture_prepare",
        _FULL_BACKEND,
        "sglang.srt.layers.dp_attention.set_dp_buffer_len",
        "sglang.srt.layers.dp_attention.set_is_extend_in_batch",
        "sglang.srt.model_executor.forward_batch_info.PPProxyTensors",
        "sglang.srt.model_executor.forward_context.ForwardContext",
        "sglang.srt.model_executor.forward_context.forward_context",
        "sglang.srt.model_executor.runner.flashinfer_autotune.maybe_flashinfer_autotune_speculative_draft",
        "sglang.srt.runtime_context.get_exec",
        "sglang.srt.utils.common.empty_context",
    ),
    reason=(
        "Pinned body with the fork change carried over, no edits: require "
        "FullCudaGraphBackend under the QSA graph, compute shape_key before "
        "forward_context (moved out of canary_ctx; pure, inventory section 4 "
        "item 3), enter qsa.graph_capture after capture_prepare (which fills "
        "num_real_reqs, the runtime's real-row count that graph_capture zeroes, "
        "so an around on this method would not reproduce it), and chain "
        "qsa.after_graph_warmup after the attention warmup hook (a closure). "
        "FullCudaGraphBackend is imported here as the fork imports it into the "
        "runner module. hisparse: gated on hisparse_coordinator.adapter."
    ),
)
def capture_one_shape(
    self,
    size: int,
    forward: Callable,
    stream_idx: Optional[int] = None,
    variant_label: Optional[str] = None,
    attention_variant: Optional[str] = None,
):
    num_tokens = size * self.captured_req_width
    bs = self._ragged_capture_slots(num_tokens) if self.ragged_verify_mode else size

    # Sanity-check: --debug-cuda-graph requires breakable backend.
    if get_exec().graph.debug_cuda_graph:
        assert isinstance(self.backend, BreakableCudaGraphBackend), (
            "Breakable CUDA graph is required for --debug-cuda-graph"
        )

    forward_batch, attn_backend, pp_proxy_tensors = self.capture_prepare(
        bs, stream_idx=stream_idx, num_tokens=num_tokens
    )

    # All setup hooks below read get_attn_backend() (TboForwardBatchPreparer,
    # DeepEP adapter, …) so they must run inside the same ForwardContext
    # that wraps the warmup/capture forward.
    qsa = getattr(self.model_runner.hisparse_coordinator, "adapter", None)
    if getattr(qsa, "graph_enabled", False) and not isinstance(self.backend, FullCudaGraphBackend):
        raise RuntimeError("QSA bounded capture requires the native full graph backend")
    shape_key = self._make_graph_key(
        self._capture_graph_size(bs=bs, num_tokens=num_tokens),
        stream_idx, variant_label, attention_variant)
    qsa_capture = (qsa.graph_capture(bs, native_key=shape_key, native_backend=self.backend)
                   if getattr(qsa, "graph_enabled", False)
                   else empty_context())
    with forward_context(ForwardContext(attn_backend=attn_backend)), qsa_capture:
        self.tbo_plugin.capture_one_batch_size(forward_batch, num_tokens=num_tokens)

        if forward_batch.lora_ids is not None:
            self.model_runner.lora_manager.prepare_lora_batch(forward_batch)

        attn_backend.init_forward_metadata_out_graph(forward_batch, in_capture=True)

        def run_once():
            # Graph-recordable metadata-prep hook. The unified memory pool
            # records ZERO translate nodes here: all its read/write translates
            # run eagerly in `init_forward_metadata_out_graph` (replay-prep), so
            # the captured graph reads already-physical locs. Base no-op for triton.
            attn_backend.init_forward_metadata_in_graph(forward_batch)
            self._record_in_graph_metadata_prep_done()

            # No invalidate_loc_cache() here: the unified pool translates its
            # locs in `init_forward_metadata_out_graph`, so no cache to invalidate.

            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = (
                None
            )
            set_dp_buffer_len(
                forward_batch.global_dp_buffer_len,
                num_tokens,
                forward_batch.dp_padding_mode.is_max_len(),
                forward_batch.global_num_tokens_cpu,
            )
            set_is_extend_in_batch(False)

            kwargs = {}
            if (
                self.pp_size > 1
                and "pp_proxy_tensors" in inspect.signature(forward).parameters
            ):
                kwargs["pp_proxy_tensors"] = PPProxyTensors(
                    {k: v.clone() for k, v in pp_proxy_tensors.tensors.items()}
                )
            if (
                (
                    self.dllm_uses_input_embeds
                    or (
                        self.model_runner.spec_algorithm.is_dflash_family()
                        and self.model_runner.is_draft_worker
                    )
                )
                and "input_embeds" in inspect.signature(forward).parameters
                and not hasattr(self.model_runner.model, "forward_embed")
            ):
                kwargs["input_embeds"] = self.buffers.input_embeds[:num_tokens]

            out = forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )
            for capture_hook in self.model_runner.capture_tail_hooks:
                capture_hook(self, out, forward_batch, num_tokens)
            return out

        self.deepep_adapter.capture(is_extend_in_batch=False)
        canary_ctx = (
            c.with_active_single_forward_manager(0)
            if (c := self.model_runner.canary_manager) is not None
            else contextlib.nullcontext()
        )
        # Full-physical write loc lives in the attention metadata (the backend's
        # `out_cache_loc_full_physical` -> KVWriteLoc.full_loc), so the runner
        # wires no buffer here. (SWA write loc rides the `swa_out_cache_loc` rail.)

        with canary_ctx:
            # Adaptive runners may own a different backend than model_runner.
            post_warmup_hook = getattr(
                attn_backend,
                "on_after_cuda_graph_warmup",
                None,
            )
            if getattr(qsa, "graph_enabled", False):
                attn_warmup_hook = post_warmup_hook

                def post_warmup_hook():
                    if attn_warmup_hook is not None:
                        attn_warmup_hook()
                    qsa.after_graph_warmup()
            maybe_flashinfer_autotune_speculative_draft(
                self,
                run_once,
                post_warmup_hook=post_warmup_hook,
                run_lm_head=True,
            )
            self.backend.capture_one(
                shape_key,
                run_once,
                capture_inputs=None,
                post_warmup_hook=post_warmup_hook,
            )


@patch(
    f"{_GRAPH}.load_batch",
    "replace",
    feature=HISPARSE,
    row="G02",
    depends=(
        f"{_GRAPH}._capture_graph_size",
        f"{_GRAPH}._global_num_tokens_for_graph",
        f"{_GRAPH}._make_graph_key",
        f"{_GRAPH}._max_dp_batch_size",
        f"{_GRAPH}._ragged_capture_slots",
        f"{_GRAPH}._ragged_graph_num_tokens",
        f"{_GRAPH}._replay_attn_backend",
        f"{_GRAPH}._resolve_attention_variant",
        f"{_GRAPH}._resolve_lora_variant",
        f"{_GRAPH}._stage_ragged_verify_layout",
        f"{_GRAPH}._validate_capture_hidden_mode",
        f"{_GRAPH_MODULE}.build_replay_fb_view",
        "sglang.srt.model_executor.cuda_graph_buffer_registry.CudaGraphBufferRegistry.fill_from",
        "sglang.srt.multiplex.pdmux_context.get_current_stream_idx",
        "sglang.srt.speculative.ragged_verify.resolve_ragged_verify_layout",
    ),
    reason=(
        "Pinned body with the fork change carried over, no edits: reject external "
        "preplanning under the QSA graph, call qsa.prepare_graph_replay(batch, bs) "
        "after bs is computed and before buffer_registry.fill_from, and skip "
        "num_real_reqs.fill_ (three mid-function insertions; no runner seam lies "
        "between the bucket choice and fill_from). hisparse: gated on "
        "hisparse_coordinator.adapter."
    ),
)
def load_batch(
    self,
    forward_batch: ForwardBatch,
    pp_proxy_tensors: Optional[PPProxyTensors] = None,
):
    if self.dllm_uses_input_embeds and forward_batch.input_embeds is None:
        raise ValueError(
            "Diffusion graph replay requires prepared input embeddings"
        )
    ragged_layout = (
        resolve_ragged_verify_layout(forward_batch)
        if self.ragged_verify_mode
        else None
    )
    is_ragged = ragged_layout is not None

    self.deepep_adapter.replay()

    qsa = getattr(self.model_runner.hisparse_coordinator, "adapter", None)
    if (getattr(qsa, "graph_enabled", False)
            and not forward_batch.needs_forward_metadata_init()):
        raise RuntimeError("QSA graph replay does not support external preplanning")

    if not forward_batch.needs_forward_metadata_init():
        # Pre-planned (plan-stream load_batch already ran).
        # In speculative decoding, these two fields are still needed.
        graph_size_key = (
            self._ragged_graph_size
            if is_ragged
            else self._capture_graph_size(
                bs=self.bs, num_tokens=self.bs * self.captured_req_width
            )
        )
        if is_ragged:
            assert self.raw_num_token == ragged_layout.graph_num_tokens, (
                f"stale ragged raw_num_token {self.raw_num_token} != "
                f"{ragged_layout.graph_num_tokens}"
            )
            self._stage_ragged_verify_layout(ragged_layout, graph_size_key)
        self.buffers.input_ids[: self.raw_num_token].copy_(forward_batch.input_ids)
        self.buffers.positions[: self.raw_num_token].copy_(forward_batch.positions)
        if (
            pp_proxy_tensors is not None
            and self.buffers.pp_proxy_tensors is not None
        ):
            # PP + spec verify: the pre-planned load ran without the proxy
            # (eagle_prepare_for_verify has no access to it), so the
            # graph's proxy input buffers must be refreshed here -- the
            # captured graph reads these rows (mirrors fill_from's
            # side-slot copy).
            for k, v in pp_proxy_tensors.tensors.items():
                buf = self.buffers.pp_proxy_tensors.get(k)
                if buf is not None:  # skip markers like __msg_type__
                    buf[: v.shape[0]].copy_(v)
        if (
            not is_ragged
            and (
                self.dllm_uses_input_embeds
                or (
                    self.model_runner.spec_algorithm.is_dflash_family()
                    and self.model_runner.is_draft_worker
                )
            )
            and forward_batch.input_embeds is not None
        ):
            self.buffers.input_embeds[: self.raw_num_token].copy_(
                forward_batch.input_embeds
            )
        variant_label = self._resolve_lora_variant(forward_batch)
        attention_variant = self._resolve_attention_variant(forward_batch)
        stream_idx = get_current_stream_idx() if self.enable_pdmux else None
        self._replay_graph_key = self._make_graph_key(
            graph_size_key, stream_idx, variant_label, attention_variant
        )
        return

    buffers = self.buffers
    self._validate_capture_hidden_mode(forward_batch)

    raw_bs = forward_batch.batch_size

    if is_ragged:
        raw_num_token = ragged_layout.graph_num_tokens
        graph_size_key = self._ragged_graph_num_tokens(raw_num_token)
        assert graph_size_key == ragged_layout.graph_num_tokens, (
            f"ragged verify tier mismatch: runner tier {graph_size_key} != "
            f"layout graph_num_tokens {ragged_layout.graph_num_tokens}"
        )
        bs = self._ragged_capture_slots(graph_size_key)
        assert bs >= raw_bs, (
            f"ragged capture slots {bs} (tier {graph_size_key}) < raw_bs "
            f"{raw_bs}; the planner must reject this batch before replay"
        )
        padded_num_tokens = graph_size_key
        self._stage_ragged_verify_layout(ragged_layout, graph_size_key)
    else:
        raw_num_token = raw_bs * self.captured_req_width
        if self.require_mlp_tp_gather:
            max_batch_size = self._max_dp_batch_size(forward_batch)
            bs = self._pad_to_bucket(max_batch_size, self.capture_bs)
        else:
            bs = self._pad_to_bucket(raw_bs, self.capture_bs)
        padded_num_tokens = bs * self.captured_req_width
        graph_size_key = self._capture_graph_size(
            bs=bs, num_tokens=padded_num_tokens
        )

    if getattr(qsa, "graph_enabled", False):
        qsa.prepare_graph_replay(forward_batch, bs)

    self.buffer_registry.fill_from(
        forward_batch,
        raw_bs=raw_bs,
        padded_bs=bs,
        raw_num_tokens=raw_num_token,
        padded_num_tokens=padded_num_tokens,
        pp_proxy_tensors=pp_proxy_tensors,
    )

    if (
        not is_ragged
        and (
            self.dllm_uses_input_embeds
            or (
                self.model_runner.spec_algorithm.is_dflash_family()
                and self.model_runner.is_draft_worker
            )
        )
        and forward_batch.input_embeds is not None
    ):
        buffers.input_embeds[:raw_num_token].copy_(forward_batch.input_embeds)
    # Padded tokens aren't read, so skip zeroing. Ragged input_ids arrive
    # from the planner already padded to the tier, invalid slots zeroed.
    if self.enable_two_batch_overlap:
        self.tbo_plugin.replay_prepare(
            forward_mode=self.capture_forward_mode,
            bs=bs,
            num_token_non_padded=len(forward_batch.input_ids),
            spec_info=forward_batch.spec_info,
        )
    if (
        not is_ragged
        and forward_batch.forward_mode.is_idle()
        and forward_batch.spec_info is not None
    ):
        forward_batch.spec_info.custom_mask = buffers.custom_mask

    attn_backend = self._replay_attn_backend()
    fb_view = build_replay_fb_view(
        forward_batch=forward_batch,
        buffers=buffers,
        bs=bs,
        raw_bs=raw_bs,
        num_tokens=padded_num_tokens,
        global_num_tokens_cpu=self._global_num_tokens_for_graph(padded_num_tokens),
        seq_len_fill_value=self.seq_len_fill_value,
        capture_forward_mode=self.capture_forward_mode,
        is_encoder_decoder=self.is_encoder_decoder,
    )
    if (
        self.model_runner.lora_manager is not None
        and self.model_runner.lora_manager.enable_dp_attention
    ):
        self.model_runner.lora_manager.prepare_lora_batch(
            cast(ForwardBatch, fb_view)
        )
    # Glue-graph fast path: pointer-stable prep (static buffers + pool
    # tensors only) is captured per key; guards keep every python-visible
    # branch inside the backends constant for that key.
    if (
        self._metadata_glue is not None
        and not self._metadata_glue.disabled
        and raw_bs == bs
        and not self.enable_two_batch_overlap
        and not self.enable_pdmux
        and self.model_runner.lora_manager is None
    ):
        # actual_forward_mode belongs in the key even though the captured
        # graph always targets capture_forward_mode: DSV4's replay prep
        # substitutes seq_lens / seq_lens_cpu / seq_lens_sum /
        # req_pool_indices / out_cache_loc when the runtime mode is IDLE,
        # so IDLE and active DECODE are different python branches and must
        # not share a captured graph.
        self._metadata_glue.run(
            attn_backend,
            fb_view,
            (
                bs,
                str(self.capture_forward_mode),
                str(fb_view.actual_forward_mode),
            ),
        )
    else:
        attn_backend.init_forward_metadata_out_graph(fb_view)

    self.raw_bs = raw_bs
    self.raw_num_token = raw_num_token
    self.bs = bs
    if is_ragged:
        self._ragged_graph_size = graph_size_key

    if self.model_runner.hisparse_coordinator is not None and not getattr(qsa, "graph_enabled", False):
        self.model_runner.hisparse_coordinator.num_real_reqs.fill_(raw_bs)

    variant_label = self._resolve_lora_variant(forward_batch)
    attention_variant = self._resolve_attention_variant(forward_batch)
    stream_idx = get_current_stream_idx() if self.enable_pdmux else None
    self._replay_graph_key = self._make_graph_key(
        graph_size_key, stream_idx, variant_label, attention_variant
    )


@patch(
    f"{_GRAPH}.execute",
    "replace",
    feature=HISPARSE,
    row="G03",
    depends=(
        f"{_GRAPH}._process_output_after_replay",
        f"{_GRAPH}._publish_read_done",
        f"{_GRAPH}._ragged_capture_slots",
        f"{_GRAPH}._replay_attn_backend",
        f"{_GRAPH}._resolve_shared_read_ends",
        f"{_GRAPH}.load_batch",
        f"{_FULL_BACKEND}.replay",
        "sglang.srt.layers.logits_processor.LogitsProcessorOutput",
        "sglang.srt.model_executor.forward_batch_info.PPProxyTensors",
        "sglang.srt.utils.common.empty_context",
        "sglang.srt.utils.device_timer.device_timer_ctx",
    ),
    reason=(
        "Pinned body with the fork change carried over, no edits: enter "
        "qsa.graph_replay_scope inside timer_ctx and replay_session, and call "
        "qsa.finish_graph_replay directly after backend.replay, before "
        "_publish_read_done (the pinned _process_output_after_replay seam runs "
        "after _publish_read_done, so it does not reproduce that order). logger "
        "is the runner module's logger. hisparse: gated on "
        "hisparse_coordinator.adapter."
    ),
)
def execute(
    self,
    forward_batch: ForwardBatch,
    pp_proxy_tensors: Optional[PPProxyTensors] = None,
) -> Union[LogitsProcessorOutput, PPProxyTensors]:
    timer_ctx = device_timer_ctx(
        self.model_runner.device_timer, forward_batch.forward_mode.name.lower()
    )
    shared_read_ends = self._resolve_shared_read_ends(
        self._replay_attn_backend(), forward_batch.forward_mode
    )
    qsa = getattr(self.model_runner.hisparse_coordinator, "adapter", None)
    qsa_scope = (qsa.graph_replay_scope() if getattr(qsa, "graph_enabled", False)
                 else empty_context())
    with timer_ctx, self.backend.replay_session(), qsa_scope:
        self.load_batch(forward_batch, pp_proxy_tensors)
        if envs.SGLANG_LOG_DECODE_GRAPH_KEY.get():
            logger.info(
                "Decode graph replay: worker=%s key_size=%s (%s) mode=%s raw_bs=%d%s",
                "draft" if self.model_runner.is_draft_worker else "target",
                self._replay_graph_key.size,
                "num_tokens" if self.ragged_verify_mode else "bs",
                forward_batch.forward_mode.name,
                forward_batch.batch_size,
                (
                    f" slots={self._ragged_capture_slots(self._replay_graph_key.size)}"
                    if self.ragged_verify_mode
                    else ""
                ),
            )
        if shared_read_ends is SharedReadEnds.PRE_REPLAY:
            self._publish_read_done(in_graph=False)

        output = self.backend.replay(self._replay_graph_key, forward_batch)
        if getattr(qsa, "graph_enabled", False):
            qsa.finish_graph_replay(self.bs, native_key=self._replay_graph_key,
                                    native_backend=self.backend)

        if shared_read_ends is SharedReadEnds.IN_REPLAY:
            self._publish_read_done(in_graph=True)

        if shared_read_ends is SharedReadEnds.POST_REPLAY:
            self._publish_read_done(in_graph=False)

        output = self._process_output_after_replay(output, forward_batch)

    if isinstance(output, LogitsProcessorOutput):
        if self.is_dllm:
            next_token_logits = None
            full_logits = (
                output.full_logits[: self.raw_num_token]
                if output.full_logits is not None
                else None
            )
        else:
            full_logits = None
            next_token_logits = (
                output.next_token_logits[: self.raw_num_token]
                if output.next_token_logits is not None
                else None
            )

        # Preserve extension fields produced by the eager output processor.
        return dataclasses.replace(
            output,
            next_token_logits=next_token_logits,
            full_logits=full_logits,
            hidden_states=(
                output.hidden_states[: self.raw_num_token]
                if output.hidden_states is not None
                else None
            ),
        )
    else:
        assert isinstance(output, PPProxyTensors)
        # Slice in token rows, not request rows: under speculative verify
        # each request carries captured_req_width tokens (identical for
        # plain decode, where captured_req_width == 1).
        return PPProxyTensors(
            {
                k: v[: self.bs * self.captured_req_width]
                for k, v in output.tensors.items()
            }
        )
