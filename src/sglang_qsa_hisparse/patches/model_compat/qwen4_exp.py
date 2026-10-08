"""Qwen4Exp INT8-row PLE and stable HC helper (inventory E01, E03, E06-E08).

E02 (``mix(..., stable=_stable_hc())`` at the three call sites) is reproduced
by H08's default in ``hyperconnection.py``; E04 (the int8/row-scale gather
kernel) lives in ``sglang_qsa_hisparse.kernels.ple_gather``; E05 is a docstring.

Every row changes model loading or PLE gathers with HiSparse unset, so it
belongs to ``model_compat`` (PLAN.md rule 1). The patched classes are
Qwen4Exp's, but the inventory treats them as shared paths (any model built
from ``qwen4_exp``), so every REPLACE delegates to the pinned method unless
``scope.target_model_active()`` (rule 9).

The ``Fork*`` classes are namespaces holding the copied method bodies at their
original indentation (the reference's definition, production ``897286b12a``,
PLAN.md Phase 5 rule Q1); they are never instantiated. Globals are imported
from the modules the pinned ``qwen4_exp`` imports them from, so every name
resolves to the same object as in the pinned module.
"""

from contextlib import nullcontext
from typing import Iterable, Optional, Set, Tuple

import torch
import triton
from torch import nn

from sglang.srt.configs.qwen4_exp import Qwen4ExpTextConfig
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPinnedHostEmbedding,
    _ple_table_is_fp8,
    _use_aiter,
    _use_attn_tp_ngram,
)
from sglang.srt.models.qwen4_exp_ple_table import (
    allocate_ple_host_table,
    make_ple_file_prefetcher,
    make_ple_file_rss_trimmer,
)
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.utils import logger
from sglang_qsa_hisparse.features import MODEL_COMPAT
from sglang_qsa_hisparse.kernels.ple_gather import (
    _gather_ple_embedding_from_pinned_kernel,
)
from sglang_qsa_hisparse.patches.model_compat.quantization import scoped
from sglang_qsa_hisparse.patching import patch

_QWEN4_EXP = "sglang.srt.models.qwen4_exp"
_NGRAM = f"{_QWEN4_EXP}.Qwen4ExpNGramEmbedding"
_PINNED_HOST = f"{_QWEN4_EXP}.Qwen4ExpPinnedHostEmbedding"
_MODEL = f"{_QWEN4_EXP}.Qwen4ExpForConditionalGeneration"
_PLE_TABLE = "sglang.srt.models.qwen4_exp_ple_table"


# E01, verbatim; H08 uses it as the default for stable.
def _stable_hc() -> bool:
    try:
        return bool(get_exec().deterministic.enable_deterministic_inference)
    except ValueError:
        return False


class ForkQwen4ExpNGramEmbedding:
    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        embedding_dim: int,
        ple_layer_index: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super(Qwen4ExpNGramEmbedding, self).__init__()
        self.config = config
        self.ngram_embed_dim = int(embedding_dim)
        self.ngram_size = int(config.ngram_size)
        self.heads_per_ngram = int(config.heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.ple_layer_index = int(ple_layer_index)
        self.unigram_vocab_size = int(config.vocab_size)
        if self.ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {self.ngram_size}")
        if self.heads_per_ngram <= 0:
            raise ValueError(f"heads_per_ngram must be > 0, got {self.heads_per_ngram}")
        if self.ngram_embed_dim % self.ngram_heads != 0:
            raise ValueError(
                "ple_embed_dim must be divisible by total ngram heads: "
                f"{self.ngram_embed_dim} % {self.ngram_heads} != 0"
            )
        self.ngram_vocab_size_base = int(config.ngram_vocab_size_base)
        if self.ngram_vocab_size_base <= 0:
            raise ValueError("ngram_vocab_size_base must be > 0")
        self.make_ngram_vocab_size_divisible_by = int(
            config.make_ngram_vocab_size_divisible_by
        )
        self.head_dim_per_ngram = self.ngram_embed_dim // self.ngram_heads
        self.eos_token_id = int(config.eos_token_id)
        self.enable_ple_fusion = envs.SGLANG_ENABLE_QWEN4_PLE_FUSION.get()

        self.register_buffer(
            "layer_multipliers",
            self._build_layer_multipliers(self.ngram_size),
            persistent=True,
        )
        head_vocab_sizes, head_offsets, total_vocab_size = (
            self._build_head_vocab_and_offsets()
        )
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.tensor(head_vocab_sizes, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "ngram_heads_offsets",
            torch.tensor(head_offsets, dtype=torch.long),
            persistent=True,
        )
        padded_vocab_size = (
            (total_vocab_size + self.make_ngram_vocab_size_divisible_by - 1)
            // self.make_ngram_vocab_size_divisible_by
        ) * self.make_ngram_vocab_size_divisible_by
        self.use_attn_tp_ngram = _use_attn_tp_ngram()
        self.gather_dp_tokens = (
            is_dp_attention_enabled()
            and get_parallel().attn_dp_size > 1
            and not self.use_attn_tp_ngram
        )
        ngram_prefix = f"{prefix}.ngram_embedding" if prefix else "ngram_embedding"
        offload_embedding = bool(config.ple_offload_embedding)
        # Offload only needs this embedding's metadata: build it on meta so the
        # shard is never allocated on the device.
        with torch.device("meta") if offload_embedding else nullcontext():
            ngram_embedding = VocabParallelEmbedding(
                padded_vocab_size,
                self.head_dim_per_ngram,
                params_dtype=(
                    torch.float8_e4m3fn
                    if _ple_table_is_fp8(config, quant_config, ngram_prefix)
                    else torch.int8
                    if getattr(config, "ple_embedding_dtype", None)
                    in ("int8", "int8_row")
                    else torch.bfloat16
                ),
                output_dtype=torch.bfloat16,
                use_attn_tp_group=self.use_attn_tp_ngram,
            )
        ngram_embedding.register_buffer(
            "weight_scale", torch.ones(1, dtype=torch.bfloat16), persistent=True
        )
        # "int8_row": per-row symmetric scale, applied inside the pinned-host
        # gather kernel; the global weight_scale scalar then stays 1.0 (no-op).
        self.ple_row_scale_mode = (
            getattr(config, "ple_embedding_dtype", None) == "int8_row"
        )
        ngram_embedding.ple_row_scale_mode = self.ple_row_scale_mode
        if self.ple_row_scale_mode:
            if ngram_embedding.weight.dtype != torch.int8:
                raise ValueError(
                    "ple_embedding_dtype='int8_row' conflicts with the fp8 "
                    "embedding storage selected by the quant config"
                )
            if not getattr(config, "ple_offload_embedding", False):
                raise ValueError(
                    "ple_embedding_dtype='int8_row' requires "
                    "ple_offload_embedding (row scales are applied inside the "
                    "pinned-host gather kernel)"
                )
        if offload_embedding:
            ngram_embedding = Qwen4ExpPinnedHostEmbedding(
                ngram_embedding,
                backend=getattr(config, "ple_offload_backend", "pinned"),
                table_dir=getattr(config, "ple_offload_dir", None),
            )
        self.ngram_embedding = ngram_embedding


class ForkQwen4ExpPinnedHostEmbedding:
    def __init__(
        self,
        embedding: VocabParallelEmbedding,
        *,
        backend: str = "pinned",
        table_dir: Optional[str] = None,
    ) -> None:
        nn.Module.__init__(self)
        if not isinstance(embedding.quant_method, UnquantizedEmbeddingMethod):
            raise NotImplementedError(
                "PLE embedding offload requires an unquantized embedding table"
            )
        if embedding.weight.dtype not in (
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.int8,
        ):
            raise TypeError(
                "PLE embedding offload requires bfloat16, fp8, or int8 weights, "
                f"got {embedding.weight.dtype}"
            )
        if embedding.num_added_embeddings:
            raise NotImplementedError(
                "PLE embedding offload does not support added vocabulary rows"
            )
        for name in self._COPIED_ATTRIBUTES:
            setattr(self, name, getattr(embedding, name))
        # The unquantized CUDA post-load hook is a no-op. Exclude this CPU-only
        # table so the generic loader does not stage it back to GPU unnecessarily.
        self.quant_method = None

        source_weight = embedding.weight
        host_table = allocate_ple_host_table(
            shape=source_weight.shape,
            dtype=source_weight.dtype,
            backend=backend,
            table_dir=table_dir,
            # Each TP rank holds a different vocabulary shard of the same shape.
            tag=(
                f"rows{self.shard_indices.org_vocab_start_index}"
                f"-{self.shard_indices.org_vocab_end_index}"
            ),
        )
        # Only the file backend has anything to prefetch (rows live on storage).
        self._file_prefetcher = make_ple_file_prefetcher(host_table)
        # ... and only it needs its resident set bounded: a fault maps a whole
        # folio, so the mapping would otherwise creep towards the full table.
        self._file_rss_trimmer = make_ple_file_rss_trimmer(host_table)
        cpu_weight = nn.Parameter(host_table, requires_grad=False)
        for name, value in vars(source_weight).items():
            setattr(cpu_weight, name, value)
        cpu_weight.weight_loader = self.weight_loader
        self.register_parameter("weight", cpu_weight)
        # The scale is tiny; keep it with the model instead of offloading it
        # with the table.
        self.register_buffer("weight_scale", embedding.weight_scale, persistent=True)
        self.ple_row_scale_mode = bool(getattr(embedding, "ple_row_scale_mode", False))
        if self.ple_row_scale_mode:
            self.register_buffer(
                "row_scale",
                torch.full(
                    (source_weight.shape[0],),
                    float("nan"),
                    dtype=torch.bfloat16,
                    device="cpu",
                ).pin_memory(),
                persistent=False,
            )
        del embedding.weight
        self._block_d = triton.next_power_of_2(self.embedding_dim)

    def gather(
        self, input_ids: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        expected_shape = (*input_ids.shape, self.embedding_dim)
        if out is None:
            output = self.allocate_output(expected_shape, input_ids.device)
        else:
            if tuple(out.shape) != expected_shape:
                raise ValueError(
                    f"invalid PLE prefetch output shape: {tuple(out.shape)} != "
                    f"{expected_shape}"
                )
            if out.dtype != torch.bfloat16 or out.device != input_ids.device:
                raise ValueError(
                    "PLE prefetch output must be bfloat16 on the id device"
                )
            output = out

        flat_ids = input_ids.reshape(-1).long()
        if flat_ids.numel():
            if self._file_prefetcher is not None:
                self._file_prefetcher.enqueue(
                    flat_ids,
                    vocab_start=self.shard_indices.org_vocab_start_index,
                    vocab_end=self.shard_indices.org_vocab_end_index,
                )
            _gather_ple_embedding_from_pinned_kernel[(flat_ids.numel(),)](
                self.weight.data_ptr(),
                flat_ids,
                output,
                self.row_scale.data_ptr() if self.ple_row_scale_mode else 0,
                embedding_dim=self.embedding_dim,
                tp_vocab_start=self.shard_indices.org_vocab_start_index,
                tp_vocab_end=self.shard_indices.org_vocab_end_index,
                is_fp8=self.weight.dtype == torch.float8_e4m3fn,
                is_int8=self.weight.dtype == torch.int8,
                has_row_scale=self.ple_row_scale_mode,
                BLOCK_D=self._block_d,
            )
        return output


class ForkQwen4ExpForConditionalGeneration:
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            # Checkpoints use the qwen3.5 head-first in_proj layout,
            # matching Qwen3_5GatedDeltaNet's forward, not qwen3-next's group-first.
            ("in_proj_qkvz.", "in_proj_qkv.", (0, 1, 2)),
            ("in_proj_qkvz.", "in_proj_z.", 3),
            ("in_proj_ba.", "in_proj_b.", 0),
            ("in_proj_ba.", "in_proj_a.", 1),
        ]

        num_experts = getattr(self.config, "num_experts", None)
        # A fused shared expert lives in routed slot `num_experts`, so the
        # mapping has to cover one more expert than the config declares.
        num_fused_shared_experts = 0
        if _use_aiter:
            for module in self.modules():
                fused = getattr(module, "num_fused_shared_experts", 0)
                if fused:
                    num_fused_shared_experts = fused
                    break
        expert_params_mapping = (
            FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=num_experts + num_fused_shared_experts,
            )
            if num_experts is not None
            else []
        )
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale_inv",
            "_weight_scale_inv",
            ".input_scale_inv",
            "_input_scale_inv",
            "_weight_scale",
            "_input_scale",
        )

        def load_fused_expert_weights(
            name: str,
            params_dict: dict,
            loaded_weight: torch.Tensor,
            shard_id: str,
            num_experts: int,
        ) -> bool:
            if name not in params_dict:
                return False
            param = params_dict[name]
            weight_loader = param.weight_loader
            for expert_id in range(num_experts):
                weight_loader(
                    param,
                    loaded_weight[expert_id],
                    name,
                    shard_id,
                    expert_id,
                )
            return True

        def copy_ple_rows_to_tp_embedding(
            emb, loaded_weight: torch.Tensor, row_start: int, row_end: int
        ) -> None:
            tp_start = emb.shard_indices.org_vocab_start_index
            tp_end = emb.shard_indices.org_vocab_end_index
            ov_start = max(row_start, tp_start)
            ov_end = min(row_end, tp_end)
            if ov_start < ov_end:
                local_start = ov_start - tp_start
                src_start = ov_start - row_start
                n_rows = ov_end - ov_start
                emb.weight.data[local_start : local_start + n_rows].copy_(
                    loaded_weight[src_start : src_start + n_rows].to(
                        device=emb.weight.device, dtype=emb.weight.dtype
                    )
                )

        def load_qwen4_exp_ple_shard(name: str, loaded_weight: torch.Tensor) -> bool:
            if ".ngram_embedding.shard_" not in name:
                return False
            import re

            match = re.search(
                r"\.ngram_embedding\.shard_(\d+)\.(weight|row_scale)$", name
            )
            if not match:
                return False
            shard_idx = int(match.group(1))
            mod_prefix = name[: name.index(".ngram_embedding.shard_")]
            ple_mod = ple_modules.get(mod_prefix)
            if ple_mod is None:
                return False
            emb = ple_mod.ngram_embedding
            if match.group(2) == "row_scale":
                if (
                    not isinstance(emb, Qwen4ExpPinnedHostEmbedding)
                    or not emb.ple_row_scale_mode
                ):
                    raise ValueError(
                        f"PLE checkpoint ships row_scale shards ({name}) but "
                        "ple_embedding_dtype is not 'int8_row'"
                    )
                shard_size = (
                    emb.org_vocab_size + ple_num_sync_shards - 1
                ) // ple_num_sync_shards
                shard_start = shard_idx * shard_size
                tp_start = emb.shard_indices.org_vocab_start_index
                tp_end = emb.shard_indices.org_vocab_end_index
                ov_start = max(shard_start, tp_start)
                ov_end = min(shard_start + loaded_weight.shape[0], tp_end)
                if ov_start < ov_end:
                    dst = emb.row_scale.data
                    dst[ov_start - tp_start : ov_end - tp_start].copy_(
                        loaded_weight[ov_start - shard_start : ov_end - shard_start].to(
                            device=dst.device, dtype=torch.bfloat16
                        )
                    )
                loaded_shard_params.add(f"{mod_prefix}.ngram_embedding.row_scale")
                return True
            if (
                loaded_weight.dtype == torch.float8_e4m3fn
                and emb.weight.dtype != torch.float8_e4m3fn
            ):
                if isinstance(emb, Qwen4ExpPinnedHostEmbedding):
                    # offload gathers from pinned host memory; a swapped-in
                    # pageable tensor would fault in the Triton kernel.
                    raise ValueError(
                        "fp8 PLE auto-switch is unsupported with "
                        "ple_offload_embedding; set "
                        'text_config.ple_embedding_dtype="float8_e4m3fn" instead'
                    )
                logger.info(
                    "PLE embedding switched to fp8 storage: %s (%s)",
                    mod_prefix,
                    tuple(emb.weight.data.shape),
                )
                old_weight_data = emb.weight.data
                # StartupWeightLoadManager enforces tensor identity/dtype; this
                # swap breaks that contract if the model is ever enrolled.
                emb.weight = torch.nn.Parameter(
                    torch.empty_like(old_weight_data, dtype=torch.float8_e4m3fn),
                    requires_grad=False,
                )
                del old_weight_data
                # params_dict was snapshotted before the loop; drop the stale
                # entry or it pins the old bf16 storage until load end.
                params_dict.pop(f"{mod_prefix}.ngram_embedding.weight", None)
                torch.cuda.empty_cache()
            if (
                emb.weight.dtype == torch.float8_e4m3fn
                and loaded_weight.dtype != torch.float8_e4m3fn
            ):
                if not getattr(load_qwen4_exp_ple_shard, "_warned_downcast", False):
                    load_qwen4_exp_ple_shard._warned_downcast = True
                    logger.warning(
                        "PLE checkpoint shards are %s but the embedding storage "
                        "is fp8 (ple_embedding_dtype / fp8 quant config); "
                        "downcasting is lossy",
                        loaded_weight.dtype,
                    )
            if loaded_weight.dtype == torch.int8 and emb.weight.dtype != torch.int8:
                raise ValueError(
                    "int8 PLE checkpoint shards require "
                    'text_config.ple_embedding_dtype="int8" (the offload path '
                    "cannot auto-switch table storage)"
                )
            if emb.weight.dtype == torch.int8 and loaded_weight.dtype != torch.int8:
                raise ValueError(
                    "PLE embedding storage is int8 but checkpoint shards are "
                    f"{loaded_weight.dtype}; rebuild the table as int8"
                )
            shard_size = (
                emb.org_vocab_size + ple_num_sync_shards - 1
            ) // ple_num_sync_shards
            shard_start = shard_idx * shard_size
            actual_rows = loaded_weight.shape[0]
            shard_end = shard_start + actual_rows
            copy_ple_rows_to_tp_embedding(emb, loaded_weight, shard_start, shard_end)
            loaded_shard_params.add(f"{mod_prefix}.ngram_embedding.weight")
            return True

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        buffers = dict(self.named_buffers())

        ple_modules = {
            mod_name: mod
            for mod_name, mod in self.named_modules()
            if isinstance(mod, Qwen4ExpNGramEmbedding)
        }
        text_config = getattr(self.config, "text_config", self.config)
        ple_num_sync_shards = int(
            getattr(
                text_config,
                "split_ngram_parts",
                getattr(self.config, "split_ngram_parts", 512),
            )
        )
        loaded_params: Set[str] = set()
        loaded_buffers: Set[str] = set()
        loaded_shard_params: Set[str] = set()
        skipped_visual_count = 0

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "mtp" in name:
                continue
            if "visual" in name and self.language_model_only:
                skipped_visual_count += 1
                continue
            if "language_model" in name:
                name = name.replace("model.language_model.", "model.")
            if ".self_attn." in name:
                name = name.replace(".self_attn", "")
            if name.endswith(".k_proj.k_scale"):
                name = name.replace(".k_proj.k_scale", ".attn.k_scale")
            elif name.endswith(".v_proj.v_scale"):
                name = name.replace(".v_proj.v_scale", ".attn.v_scale")

            layer_id = get_layer_id(name)
            if layer_id is not None and (
                layer_id < self.start_layer or layer_id >= self.end_layer
            ):
                continue

            if self._load_qwen4_exp_ple_buffer(
                name, loaded_weight, buffers, loaded_buffers
            ):
                continue
            if load_qwen4_exp_ple_shard(name, loaded_weight):
                continue
            if ".ple.ple_embedding.ngram_embedding." in name and name.endswith(
                ".weight"
            ):
                raise ValueError(
                    f"unsupported PLE weight layout (expected shard_N shards): {name}"
                )

            if (
                self.config.tie_word_embeddings
                and self.pp_group.is_last_rank
                and "model.embed_tokens.weight" in name
                and "lm_head.weight" in params_dict
            ):
                lm_head_param = params_dict["lm_head.weight"]
                weight_loader = getattr(
                    lm_head_param, "weight_loader", default_weight_loader
                )
                weight_loader(lm_head_param, loaded_weight)

            if (
                not self.pp_group.is_last_rank
                and "model.hyper_connection_mixer." in name
            ):
                continue

            if (
                _use_aiter
                and num_fused_shared_experts > 0
                and "mlp.shared_expert." in name
            ):
                # Map mlp.shared_expert.xx_proj to mlp.experts.{num_experts}.xx_proj
                name = name.replace(
                    "mlp.shared_expert.",
                    f"mlp.experts.{num_experts}.",
                )

            is_fused_expert = (
                "experts.gate_up_proj" in name or "experts.down_proj" in name
            )

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "visual" in name or "mlp.experts" in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if (
                    mapped_name.endswith(ignore_suffixes)
                    and mapped_name not in params_dict
                ):
                    continue
                if mapped_name not in params_dict:
                    continue
                param = params_dict[mapped_name]
                param.weight_loader(param, loaded_weight, shard_id)
                name = mapped_name
                break
            else:
                is_expert_weight = False
                current_expert_params_mapping = (
                    fused_expert_params_mapping
                    if is_fused_expert
                    else expert_params_mapping
                )
                for mapping in current_expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    if "visual" in name or self.config.encoder_only:
                        continue
                    is_expert_weight = True
                    mapped_name = name.replace(weight_name, param_name)
                    if is_fused_expert:
                        if "experts.gate_up_proj" in name:
                            gate_weight, up_weight = loaded_weight.chunk(2, dim=-2)
                            if not load_fused_expert_weights(
                                mapped_name,
                                params_dict,
                                gate_weight,
                                "w1",
                                num_experts,
                            ):
                                raise KeyError(f"Parameter {mapped_name} not found")
                            if not load_fused_expert_weights(
                                mapped_name,
                                params_dict,
                                up_weight,
                                "w3",
                                num_experts,
                            ):
                                raise KeyError(f"Parameter {mapped_name} not found")
                        else:
                            if not load_fused_expert_weights(
                                mapped_name,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            ):
                                raise KeyError(f"Parameter {mapped_name} not found")
                    else:
                        if (
                            mapped_name.endswith(ignore_suffixes)
                            and mapped_name not in params_dict
                        ):
                            continue
                        param = params_dict[mapped_name]
                        weight_loader = param.weight_loader
                        weight_loader(
                            param,
                            loaded_weight,
                            mapped_name,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    name = mapped_name
                    break
                else:
                    if is_expert_weight:
                        continue
                    if "visual" in name:
                        name = name.replace("attn.qkv.", "attn.qkv_proj.")
                        name = name.replace("model.visual.", "visual.")
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue
                    if name.endswith("_scale") and name not in params_dict:
                        assert abs(loaded_weight.item() - 1.0) < 1e-6, (
                            f"Expected 1.0, got {loaded_weight.item()} in skipped {name}"
                        )
                        continue
                    if name in params_dict:
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning(
                            "Parameter %s not found while loading Qwen4-Exp VL weights",
                            name,
                        )
                        continue
            loaded_params.add(name)

        loaded_params.update(loaded_buffers)
        loaded_params.update(loaded_shard_params)

        for mod_prefix, ple_mod in ple_modules.items():
            emb = ple_mod.ngram_embedding
            if isinstance(emb, Qwen4ExpPinnedHostEmbedding) and emb.ple_row_scale_mode:
                n_rows = emb.num_org_embeddings_per_partition
                missing = int(torch.isnan(emb.row_scale.data[:n_rows]).sum())
                if missing:
                    raise ValueError(
                        f"PLE row_scale shards missing for {mod_prefix}: "
                        f"{missing} of {n_rows} rows unloaded"
                    )

        if skipped_visual_count > 0:
            logger.info(
                f"[language_model_only] Qwen4 load_weights: skipped "
                f"{skipped_visual_count} visual weights"
            )

        for module in self.modules():
            if isinstance(module, Qwen3_5GatedDeltaNet):
                module.finalize_fused_in_proj()

        return loaded_params


patch(
    f"{_NGRAM}.__init__",
    "replace",
    feature=MODEL_COMPAT,
    row="E03",
    depends=(
        f"{_NGRAM}._build_layer_multipliers",
        f"{_NGRAM}._build_head_vocab_and_offsets",
        f"{_QWEN4_EXP}._ple_table_is_fp8",
        f"{_QWEN4_EXP}._use_attn_tp_ngram",
        "sglang.srt.layers.vocab_parallel_embedding.VocabParallelEmbedding.__init__",
        "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
        "sglang.srt.runtime_context.get_parallel",
        # E06's target: the copy wraps the offloaded table, as the pin does.
        f"{_PINNED_HOST}.__init__",
    ),
    reason=(
        "Fork: int8 storage for ple_embedding_dtype int8/int8_row and int8_row "
        "validation (the fork's meta-device table for ple_offload_embedding is "
        "upstream since v0.5.21, which also moved the Qwen4ExpPinnedHostEmbedding "
        "wrapping into this method). Mid-function (constructor dtype; "
        "ple_row_scale_mode must be set before the wrapping, which reads it in "
        "E06), so replace. Copy: the reference's definition (production "
        "re-merged the fork's change before the wrapping and dropped the "
        "weight_scale comment). "
        "model_compat: changes PLE construction with HiSparse unset. Scope "
        "(rule 9): the copy runs only when target_model_active(), otherwise the "
        "pinned method. Mechanical edit (inventory 6, G2): super().__init__() "
        "-> super(Qwen4ExpNGramEmbedding, self).__init__(), because the copy is "
        "defined outside the class."
    ),
)(scoped(Qwen4ExpNGramEmbedding.__init__, ForkQwen4ExpNGramEmbedding.__init__))

patch(
    f"{_PINNED_HOST}.__init__",
    "replace",
    feature=MODEL_COMPAT,
    row="E06",
    depends=(
        f"{_PLE_TABLE}.allocate_ple_host_table",
        f"{_PLE_TABLE}.make_ple_file_prefetcher",
        f"{_PLE_TABLE}.make_ple_file_rss_trimmer",
        "sglang.srt.layers.vocab_parallel_embedding.VocabParallelEmbedding.weight_loader",
    ),
    reason=(
        "Fork: accept int8 tables and, in int8_row mode, register a pinned "
        "NaN-filled per-row bf16 row_scale buffer. Mid-function, so replace. "
        "model_compat: PLE offload construction with HiSparse unset. Scope "
        "(rule 9): the copy runs only when target_model_active(), otherwise the "
        "pinned method. Copy ForkQwen4ExpPinnedHostEmbedding.__init__ is the "
        "reference's definition; mechanical edits: none."
    ),
)(
    scoped(
        Qwen4ExpPinnedHostEmbedding.__init__,
        ForkQwen4ExpPinnedHostEmbedding.__init__,
    )
)

patch(
    f"{_PINNED_HOST}.gather",
    "replace",
    feature=MODEL_COMPAT,
    row="E07",
    depends=(f"{_PINNED_HOST}.allocate_output",),
    reason=(
        "Fork: the gather kernel gains the row_scale pointer and the is_int8 / "
        "has_row_scale constexprs, so the launch arguments change. model_compat: "
        "PLE gathers with HiSparse unset. Scope (rule 9): the copy runs only "
        "when target_model_active(), otherwise the pinned method. Mechanical "
        "edit: the copy's _gather_ple_embedding_from_pinned_kernel is the plugin "
        "copy of the fork kernel (sglang_qsa_hisparse.kernels.ple_gather, E04), "
        "imported in place of the pinned kernel; the body is verbatim."
    ),
)(scoped(Qwen4ExpPinnedHostEmbedding.gather, ForkQwen4ExpPinnedHostEmbedding.gather))

patch(
    f"{_MODEL}.load_weights",
    "replace",
    feature=MODEL_COMPAT,
    row="E08",
    depends=(
        f"{_MODEL}._load_qwen4_exp_ple_buffer",
        "sglang.srt.layers.utils.common.get_layer_id",
        "sglang.srt.model_loader.weight_utils.default_weight_loader",
        "sglang.srt.layers.moe.fused_moe_triton.layer.FusedMoE.make_expert_params_mapping",
        "sglang.srt.models.qwen3_5.Qwen3_5GatedDeltaNet.finalize_fused_in_proj",
    ),
    reason=(
        "Fork: row_scale shards, int8 storage consistency errors and a post-load "
        "NaN row_scale coverage check. The shard changes are inside the nested "
        "closure load_qwen4_exp_ple_shard, which cannot be hooked, so replace. "
        "model_compat: model loading with HiSparse unset. Scope (rule 9): the "
        "copy runs only when target_model_active(), otherwise the pinned method. "
        "Copy ForkQwen4ExpForConditionalGeneration.load_weights is the "
        "reference's definition; mechanical edits: none (every global is imported from the module the "
        "pinned qwen4_exp imports it from)."
    ),
)(
    scoped(
        Qwen4ExpForConditionalGeneration.load_weights,
        ForkQwen4ExpForConditionalGeneration.load_weights,
    )
)
