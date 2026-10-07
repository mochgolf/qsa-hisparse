"""GPTQ / AutoRound Marlin MoE loading (inventory Z01-Z04), scoped to the target model.

All four rows change behavior while HiSparse is unset, so they belong to
``model_compat`` (PLAN.md rule 1). They sit on generic quantized-MoE loading
paths, so every hook delegates to the pinned code unless
``scope.target_model_active()`` (rule 9).

The ``Fork*`` classes are namespaces holding verbatim copies of the fork's
method bodies at their original indentation; they are never instantiated.
"""

import gc

import torch

# Initialize the quantization package first: at the pin, importing
# gptq_kernels before it is circular (gptq_kernels -> quantization -> gptq_kernels).
import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.hardware_backend.gpu.quantization.gptq_kernels import (
    GPTQMarlinMoEKernel,
    gptq_marlin_moe_repack,
)
from sglang.srt.layers.linear import set_weight_attrs
from sglang.srt.layers.quantization.auto_round import (
    AutoRoundConfig,
    _is_cpu,
    _is_npu,
    logger,
    scalar_types,
)
from sglang.srt.layers.quantization.gptq.schemes.gptq_moe import GPTQMarlinMoEScheme
from sglang.srt.layers.quantization.marlin_utils import marlin_moe_permute_scales
from sglang.srt.layers.quantization.utils import replace_parameter
from sglang_qsa_hisparse import scope
from sglang_qsa_hisparse.features import MODEL_COMPAT
from sglang_qsa_hisparse.patching import patch

_GPTQ_KERNELS = "sglang.srt.hardware_backend.gpu.quantization.gptq_kernels"
_GPTQ_MOE = "sglang.srt.layers.quantization.gptq.schemes.gptq_moe"
_AUTO_ROUND = "sglang.srt.layers.quantization.auto_round"
_MARLIN_UTILS = "sglang.srt.layers.quantization.marlin_utils"
_FUSED_MOE = "sglang.srt.layers.moe.fused_moe_triton.layer.FusedMoE"


def scoped(original, copy):
    """REPLACE body for a generic path: the fork copy only for the target model.

    ``original`` is captured here, at plugin import, before any hook is
    applied; a closure is not rebound by HookRegistry's patch propagation.
    """

    def dispatch(*args, **kwargs):
        if scope.target_model_active():
            return copy(*args, **kwargs)
        return original(*args, **kwargs)

    return dispatch


class ForkGPTQMarlinMoEKernel:
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:

        # Process act_order
        if self.quant_config.desc_act:
            # Get sorting based on g_idx
            num_experts = layer.w13_g_idx.shape[0]
            w13_g_idx_sort_indices = torch.empty_like(layer.w13_g_idx)
            w2_g_idx_sort_indices = torch.empty_like(layer.w2_g_idx)
            w13_sorted_g_idx = torch.empty_like(layer.w13_g_idx)
            w2_sorted_g_idx = torch.empty_like(layer.w2_g_idx)
            for e in range(num_experts):
                w13_g_idx_sort_indices[e] = torch.argsort(layer.w13_g_idx[e]).to(
                    torch.int32
                )
                w2_g_idx_sort_indices[e] = torch.argsort(layer.w2_g_idx[e]).to(
                    torch.int32
                )
                w13_sorted_g_idx[e] = layer.w13_g_idx[e][w13_g_idx_sort_indices[e]]
                w2_sorted_g_idx[e] = layer.w2_g_idx[e][w2_g_idx_sort_indices[e]]
            replace_parameter(layer, "w13_g_idx", w13_sorted_g_idx)
            replace_parameter(layer, "w2_g_idx", w2_sorted_g_idx)
            replace_parameter(layer, "w13_g_idx_sort_indices", w13_g_idx_sort_indices)
            replace_parameter(layer, "w2_g_idx_sort_indices", w2_g_idx_sort_indices)
        else:
            # Reset g_idx related tensors
            num_experts = layer.w13_g_idx.shape[0]
            device = layer.w13_g_idx.device
            layer.w13_g_idx = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w2_g_idx = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w13_g_idx_sort_indices = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w2_g_idx_sort_indices = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
        # Repack weights
        marlin_w13_qweight = gptq_marlin_moe_repack(
            layer.w13_qweight,
            layer.w13_g_idx_sort_indices,
            layer.w13_qweight.shape[1] * self.quant_config.pack_factor,
            layer.w13_qweight.shape[2],
            self.quant_config.weight_bits,
        )
        replace_parameter(layer, "w13_qweight", marlin_w13_qweight)
        gc.collect()
        torch.cuda.empty_cache()
        marlin_w2_qweight = gptq_marlin_moe_repack(
            layer.w2_qweight,
            layer.w2_g_idx_sort_indices,
            layer.w2_qweight.shape[1] * self.quant_config.pack_factor,
            layer.w2_qweight.shape[2],
            self.quant_config.weight_bits,
        )
        replace_parameter(layer, "w2_qweight", marlin_w2_qweight)
        gc.collect()
        torch.cuda.empty_cache()
        # Repack scales
        marlin_w13_scales = marlin_moe_permute_scales(
            s=layer.w13_scales,
            size_k=layer.w13_scales.shape[1]
            * (
                self.quant_config.group_size
                if self.quant_config.group_size != -1
                else self.quant_config.pack_factor
            ),
            size_n=layer.w13_scales.shape[2],
            group_size=self.quant_config.group_size,
        )
        replace_parameter(layer, "w13_scales", marlin_w13_scales)
        marlin_w2_scales = marlin_moe_permute_scales(
            s=layer.w2_scales,
            size_k=layer.w2_scales.shape[1]
            * (
                self.quant_config.group_size
                if self.quant_config.group_size != -1
                else self.quant_config.pack_factor
            ),
            size_n=layer.w2_scales.shape[2],
            group_size=self.quant_config.group_size,
        )
        replace_parameter(layer, "w2_scales", marlin_w2_scales)
        gc.collect()
        torch.cuda.empty_cache()


class ForkGPTQMarlinMoEScheme:
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        self.kernel.is_k_full = (
            not self.quant_config.desc_act
        ) or layer.moe_tp_size == 1

        if self.quant_config.group_size != -1:
            scales_size13 = hidden_size // self.quant_config.group_size
            if self.quant_config.desc_act:
                w2_scales_size = intermediate_size_per_partition
            else:
                w2_scales_size = intermediate_size_per_partition
            scales_size2 = w2_scales_size // self.quant_config.group_size
            strategy = FusedMoeWeightScaleSupported.GROUP.value
        else:
            scales_size13 = 1
            scales_size2 = 1
            strategy = FusedMoeWeightScaleSupported.CHANNEL.value

        extra_weight_attrs.update({"quant_method": strategy, "is_transposed": True})

        w13_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size // self.quant_config.pack_factor,
                2 * intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        w2_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition // self.quant_config.pack_factor,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

        w13_scales = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size13,
                2 * intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)

        w2_scales = torch.nn.Parameter(
            torch.empty(num_experts, scales_size2, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)
        set_weight_attrs(w2_scales, {"load_full_w2": self.quant_config.desc_act})

        w13_qzeros = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size13,
                2 * intermediate_size_per_partition // self.quant_config.pack_factor,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qzeros", w13_qzeros)
        set_weight_attrs(w13_qzeros, extra_weight_attrs)

        w2_qzeros = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size2,
                hidden_size // self.quant_config.pack_factor,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qzeros", w2_qzeros)
        set_weight_attrs(w2_qzeros, extra_weight_attrs)
        set_weight_attrs(w2_qzeros, {"load_full_w2": self.quant_config.desc_act})

        w13_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx", w13_g_idx)
        set_weight_attrs(w13_g_idx, extra_weight_attrs)

        w2_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx", w2_g_idx)
        set_weight_attrs(w2_g_idx, extra_weight_attrs)

        w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx_sort_indices", w13_g_idx_sort_indices)
        set_weight_attrs(w13_g_idx_sort_indices, extra_weight_attrs)

        w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx_sort_indices", w2_g_idx_sort_indices)
        set_weight_attrs(w2_g_idx_sort_indices, extra_weight_attrs)


class ForkAutoRoundConfig:
    def apply_gptq_quant_layer(
        self,
        layer,
        prefix: str,
        backend: str = "auto",
        additional_linear_types: tuple[type[torch.nn.Module], ...] = (),
    ):
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.quantization.gptq import (
            GPTQAscendConfig,
            GPTQLinearMethod,
            GPTQMoEMethod,
        )
        from sglang.srt.layers.quantization.marlin_utils import (
            check_marlin_supported,
            check_moe_marlin_supports_layer,
        )
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

        linear_types = (LinearBase, ParallelLMHead, *additional_linear_types)
        is_linear = isinstance(layer, linear_types)

        weight_bits, group_size, sym = self.get_layer_config(layer, prefix)
        if not self.check_quantized(weight_bits):
            if is_linear:
                return UnquantizedLinearMethod()
            else:
                return None

        logger.debug(
            "[%s] Type: %s, Bits: %s, Group Size: %s, Sym: %s",
            prefix,
            layer.__class__.__name__,
            weight_bits,
            group_size,
            sym,
        )
        self.log_gptq_default_assumptions_once()
        if _is_npu:
            quant_args = GPTQAscendConfig(
                **self.get_gptq_config_kwargs(weight_bits, group_size),
            )
            quant_args.sym = sym

            if isinstance(layer, FusedMoE):
                layer.scheme = quant_args.get_moe_scheme(layer)
                return GPTQMoEMethod(quant_args)

            if is_linear:
                layer.scheme = quant_args.get_linear_scheme(layer)
                return GPTQLinearMethod(quant_args)

            return None

        if _is_cpu:
            self.check_cpu_support(weight_bits)
            from sglang.srt.layers.quantization.gptq import CPUGPTQConfig

            quant_args = CPUGPTQConfig(
                **self.get_gptq_config_kwargs(weight_bits, group_size),
            )
            quant_args.sym = sym

            if isinstance(layer, FusedMoE):
                layer.scheme = quant_args.get_moe_scheme(layer)
                return GPTQMoEMethod(quant_args)

            if is_linear:
                layer.scheme = quant_args.get_linear_scheme(layer)
                return GPTQLinearMethod(quant_args)

            return None

        if backend == "auto" or "marlin" in backend:
            GPTQ_TYPE_MAP = {
                (4, True): scalar_types.uint4b8,
                (8, True): scalar_types.uint8b128,
            }
            use_marlin = (weight_bits, sym) in GPTQ_TYPE_MAP and check_marlin_supported(
                GPTQ_TYPE_MAP[(weight_bits, sym)], group_size, has_zp=not sym
            )
            if isinstance(layer, FusedMoE):
                use_marlin = use_marlin and check_moe_marlin_supports_layer(
                    layer, group_size
                )
                # Qwen3.8 experts use g128, but TP2 splits their K=640 down
                # projection at 320, which cannot preserve whole g128 groups.
                # Reading the same symmetric groups as two identical g64
                # groups keeps dequantization exact and unlocks Marlin.
                if not use_marlin and group_size == 128:
                    if check_moe_marlin_supports_layer(layer, 64):
                        use_marlin = True
                        group_size = 64
                        layer._marlin_g64_expand_scales = True
        else:
            use_marlin = False
        if use_marlin:
            from sglang.srt.layers.quantization.gptq import (
                GPTQMarlinConfig,
                GPTQMarlinLinearMethod,
                GPTQMarlinMoEMethod,
            )

            quant_args_marlin = GPTQMarlinConfig(
                weight_bits=weight_bits,
                group_size=group_size,
                is_sym=sym,
                lm_head_quantized=self.lm_head_quantized,
                desc_act=self.desc_act,
                dynamic=self.dynamic,
                full_config={},
            )
        else:
            from sglang.srt.layers.quantization.gptq import GPTQConfig, GPTQLinearMethod

            quant_args = GPTQConfig(
                **self.get_gptq_config_kwargs(weight_bits, group_size),
            )

        if isinstance(layer, FusedMoE):
            if use_marlin:
                return GPTQMarlinMoEMethod(quant_args_marlin)
            from sglang.srt.layers.quantization.moe_wna16 import MoeWNA16Config

            config = {
                "quant_method": "gptq",
                "bits": weight_bits,
                "group_size": group_size,
                "sym": sym,
                "lm_head": False,
            }
            return MoeWNA16Config.from_config(config).get_quant_method(layer, prefix)

        if is_linear:
            if use_marlin:
                return GPTQMarlinLinearMethod(quant_args_marlin)
            else:
                return GPTQLinearMethod(quant_args)

        return None


patch(
    f"{_GPTQ_KERNELS}.GPTQMarlinMoEKernel.process_weights_after_loading",
    "replace",
    feature=MODEL_COMPAT,
    row="Z01",
    depends=(
        f"{_GPTQ_KERNELS}.gptq_marlin_moe_repack",
        f"{_MARLIN_UTILS}.marlin_moe_permute_scales",
        "sglang.srt.layers.quantization.utils.replace_parameter",
    ),
    reason=(
        "Fork: gc.collect() and torch.cuda.empty_cache() after each repack, and "
        "the w13 scale permute takes size_k from the scale tensor "
        "(w13_scales.shape[1] * group) instead of intermediate_size_per_partition. "
        "Mid-function, so replace. model_compat: changes every GPTQ Marlin MoE "
        "load with HiSparse unset. Scope (rule 9): generic GPTQ Marlin MoE "
        "loading; the copy runs only when target_model_active(), otherwise the "
        "pinned method. Copy ForkGPTQMarlinMoEKernel.process_weights_after_loading "
        "is verbatim; mechanical edits: none (globals imported from the modules "
        "the fork module imports them from)."
    ),
)(
    scoped(
        GPTQMarlinMoEKernel.process_weights_after_loading,
        ForkGPTQMarlinMoEKernel.process_weights_after_loading,
    )
)

patch(
    f"{_GPTQ_MOE}.GPTQMarlinMoEScheme.create_weights",
    "replace",
    feature=MODEL_COMPAT,
    row="Z02",
    depends=(
        "sglang.srt.utils.common.set_weight_attrs",
        "sglang.srt.layers.moe.fused_moe_triton.layer.FusedMoeWeightScaleSupported",
    ),
    reason=(
        "Fork: w2 group scales are sized without moe_tp_size, and the w13/w2 "
        "scales use params_dtype instead of fp16. Mid-function values, so "
        "replace. model_compat: changes GPTQ Marlin MoE parameter shapes and "
        "dtypes with HiSparse unset. Scope (rule 9): generic GPTQ MoE weight "
        "creation; the copy runs only when target_model_active(), otherwise the "
        "pinned method. Copy ForkGPTQMarlinMoEScheme.create_weights is verbatim; "
        "mechanical edits: none."
    ),
)(scoped(GPTQMarlinMoEScheme.create_weights, ForkGPTQMarlinMoEScheme.create_weights))

patch(
    f"{_AUTO_ROUND}.AutoRoundConfig.apply_gptq_quant_layer",
    "replace",
    feature=MODEL_COMPAT,
    row="Z03",
    depends=(
        f"{_AUTO_ROUND}.AutoRoundConfig.get_layer_config",
        f"{_AUTO_ROUND}.AutoRoundConfig.check_quantized",
        f"{_AUTO_ROUND}.AutoRoundConfig.check_cpu_support",
        f"{_AUTO_ROUND}.AutoRoundConfig.get_gptq_config_kwargs",
        f"{_AUTO_ROUND}.AutoRoundConfig.log_gptq_default_assumptions_once",
        f"{_MARLIN_UTILS}.check_marlin_supported",
        f"{_MARLIN_UTILS}.check_moe_marlin_supports_layer",
        "sglang.srt.layers.quantization.unquant.UnquantizedLinearMethod",
    ),
    reason=(
        "Fork: when Marlin rejects a g128 MoE layer but accepts g64 (Qwen3.8 TP2 "
        "K=320 down projection), use Marlin g64 and set "
        "layer._marlin_g64_expand_scales for Z04. Mutates mid-function locals "
        "(use_marlin, group_size), so replace. model_compat: upstream would take "
        "the non-Marlin MoE path. Scope (rule 9): generic AutoRound dispatch; the "
        "copy runs only when target_model_active(), otherwise the pinned method. "
        "Copy ForkAutoRoundConfig.apply_gptq_quant_layer is verbatim; mechanical "
        "edits: none (logger, _is_npu, _is_cpu and scalar_types are the pinned "
        "auto_round module's own objects)."
    ),
)(
    scoped(
        AutoRoundConfig.apply_gptq_quant_layer,
        ForkAutoRoundConfig.apply_gptq_quant_layer,
    )
)

_WEIGHT_LOADER_PARAMS = (
    "self",
    "param",
    "loaded_weight",
    "weight_name",
    "shard_id",
    "expert_id",
)


@patch(
    f"{_FUSED_MOE}.weight_loader",
    "before",
    feature=MODEL_COMPAT,
    row="Z04",
    reason=(
        "Fork: on layers flagged by Z03, *_scales and *_qzeros rows are expanded "
        "with repeat_interleave(2, dim=0) (two identical g64 groups per g128 "
        "group). The fork inserts this after the static-mxfp4 early return, "
        "which requires quant_config.get_name() == 'mxfp4' and so is never taken "
        "for AutoRound-flagged layers; expanding before the call is therefore "
        "equivalent on every pinned path. Accepts loaded_weight positionally or "
        "by keyword. model_compat (part of Z03). Scope (rule 9): acts only on "
        "Z03-flagged layers and only when target_model_active(); otherwise the "
        "arguments pass through unchanged. The bound weight_loader is captured "
        "at layer construction, after activation."
    ),
)
def _expand_g64_marlin_scales(*args, **kwargs):
    call = dict(zip(_WEIGHT_LOADER_PARAMS, args), **kwargs)
    self, weight_name = call["self"], call["weight_name"]
    # The fork's condition, verbatim.
    if not (
        getattr(self, "_marlin_g64_expand_scales", False)
        and (weight_name.endswith("_scales") or weight_name.endswith("_qzeros"))
    ):
        return None
    if not scope.target_model_active():
        return None
    loaded_weight = call["loaded_weight"].repeat_interleave(2, dim=0)
    if len(args) > 2:
        args = (*args[:2], loaded_weight, *args[3:])
    else:
        kwargs = {**kwargs, "loaded_weight": loaded_weight}
    return args, kwargs
