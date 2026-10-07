"""Stable HC mixing under deterministic inference (inventory H05, H06, H08).

All three rows change Qwen4Exp's HC mix under deterministic inference with
HiSparse unset, so they belong to ``model_compat`` (PLAN.md rule 1), and they
delegate to the pinned code unless ``scope.target_model_active()`` (rule 9).
H07 (the base-class ``mix`` signature) is dropped: ``GatedResidual``, the only
subclass, overrides ``mix`` and nothing passes ``stable`` to the base.

``ForkGatedResidual`` is a namespace holding the fork's ``GatedResidual.mix``
verbatim at its original indentation; it is never instantiated. Its globals
``fused_hc_mix`` and ``fused_hc_mix_supported`` are the pinned
``hc_mix_triton`` names, which HookRegistry rebinds to the H05/H06 hooks when
it applies them (inventory 6, G3), exactly as it rebinds the pinned
``hyperconnection`` module's own imports.
"""

import torch

from sglang.srt.layers.hc_mix_triton import fused_hc_mix, fused_hc_mix_supported
from sglang_qsa_hisparse import scope
from sglang_qsa_hisparse.features import MODEL_COMPAT
from sglang_qsa_hisparse.kernels import hc_mix as fork_hc_mix
from sglang_qsa_hisparse.patches.model_compat.quantization import scoped
from sglang_qsa_hisparse.patches.model_compat.qwen4_exp import _stable_hc
from sglang_qsa_hisparse.patching import patch

_HC_MIX = "sglang.srt.layers.hc_mix_triton"
_GATED_RESIDUAL = "sglang.srt.layers.hyperconnection.GatedResidual"


class ForkGatedResidual:
    def mix(self, hyper_input: torch.Tensor, *, stable: bool = False):
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        if hyper_input.shape[0] == 0:
            mixed_input = hyper_input.new_empty(
                (*hyper_input.shape[:-1], self.hidden_size), dtype=self.params_dtype
            )
            return mixed_input, (hyper_input, hyper_input)

        if self.config.hc_per_branch_norm:
            hyper_input_normed = self.hc_norm(hyper_input)
        else:
            hyper_input_normed = self.hc_norm(
                hyper_input.unflatten(-1, (self.hc_count, self.hidden_size))
            ).flatten(-2)
        if stable and fused_hc_mix_supported(
            hyper_input_normed,
            self.input_mix_weight_down.weight,
            self.input_mix_weight_up.weight,
            stable=True,
        ):
            mixed_input = fused_hc_mix(
                hyper_input_normed,
                self.input_mix_weight_down.weight,
                self.input_mix_weight_up.weight,
                self.hc_count,
                self.hidden_size,
                stable=True,
            ).to(self.params_dtype)
        elif stable:
            mixed_input = self._mix_compute(
                hyper_input_normed,
                self.input_mix_weight_down.weight,
                self.input_mix_weight_up.weight,
                self.hc_count,
                self.hidden_size,
            ).to(self.params_dtype)
        elif (
            self._jit_mix_ok
            and hyper_input_normed.is_cuda
            and hyper_input_normed.dtype in (torch.bfloat16, torch.float16)
            and hyper_input_normed.shape[0] <= 24
        ):
            from sglang.kernels.ops.elementwise.hc_mix import (
                hc_mix,
                permute_pad_up_weight,
            )

            if self._mix_up_weight_padded is None:
                self._mix_up_weight_padded = permute_pad_up_weight(
                    self.input_mix_weight_up.weight, self.hc_count
                )
            mixed_input = hc_mix(
                hyper_input_normed,
                self.input_mix_weight_down.weight.data,
                self._mix_up_weight_padded,
                self.hc_count,
                self.hidden_size,
            ).to(self.params_dtype)
        elif fused_hc_mix_supported(
            hyper_input_normed,
            self.input_mix_weight_down.weight,
            self.input_mix_weight_up.weight,
        ):
            mixed_input = fused_hc_mix(
                hyper_input_normed,
                self.input_mix_weight_down.weight,
                self.input_mix_weight_up.weight,
                self.hc_count,
                self.hidden_size,
            ).to(self.params_dtype)
        else:
            mixed_input = self._mix_compute(
                hyper_input_normed,
                self.input_mix_weight_down.weight,
                self.input_mix_weight_up.weight,
                self.hc_count,
                self.hidden_size,
            ).to(self.params_dtype)
        return mixed_input, (hyper_input, hyper_input_normed)


patch(
    f"{_HC_MIX}.fused_hc_mix_supported",
    "replace",
    feature=MODEL_COMPAT,
    row="H05",
    depends=(f"{_HC_MIX}._deterministic_inference",),
    reason=(
        "Fork adds keyword stable=: the stable path skips the "
        "deterministic-inference early return, which is mid-function (an around "
        "would re-implement the predicate), so replace. model_compat: part of "
        "stable HC under deterministic inference. Scope (rule 9): the copy "
        "(sglang_qsa_hisparse.kernels.hc_mix.fused_hc_mix_supported, verbatim, no "
        "mechanical edits) runs only when target_model_active(), otherwise the "
        "pinned predicate. Without stable=True the copy equals the pinned "
        "predicate; stable=True is passed only by H08's copy."
    ),
)(scoped(fused_hc_mix_supported, fork_hc_mix.fused_hc_mix_supported))


@patch(
    f"{_HC_MIX}.fused_hc_mix",
    "around",
    feature=MODEL_COMPAT,
    row="H06",
    depends=(
        f"{_HC_MIX}._get_counters",
        f"{_HC_MIX}._grid_barrier",
        f"{_HC_MIX}._hc_mix_persistent_kernel",
    ),
    reason=(
        "Fork adds keywords stable= and stable_splits=; stable=True launches the "
        "fixed-order split-K kernel (H03). Around: stable=True in target scope "
        "runs the plugin copy of the fork's fused_hc_mix "
        "(sglang_qsa_hisparse.kernels.hc_mix.fused_hc_mix, verbatim); every other "
        "call goes to the pinned function, which is identical to the fork's "
        "non-stable branch. model_compat: stable HC under deterministic "
        "inference. Scope (rule 9): stable=True is passed only by H08's copy, "
        "and is honored only when target_model_active()."
    ),
)
def _fused_hc_mix(original, *args, **kwargs):
    if kwargs.get("stable") and scope.target_model_active():
        return fork_hc_mix.fused_hc_mix(*args, **kwargs)
    return original(*args, **kwargs)


@patch(
    f"{_GATED_RESIDUAL}.mix",
    "around",
    feature=MODEL_COMPAT,
    row="H08",
    depends=(
        f"{_GATED_RESIDUAL}.__init__",
        "sglang.srt.layers.hyperconnection.GroupedGemmaRMSNorm.forward",
        "sglang.kernels.ops.elementwise.hc_mix.hc_mix",
        "sglang.kernels.ops.elementwise.hc_mix.permute_pad_up_weight",
        "sglang.srt.runtime_context.get_exec",
    ),
    reason=(
        "Fork adds keyword stable= to GatedResidual.mix: the stable fused kernel "
        "(H05/H06), falling back to the torch.compile chain. Around: stable "
        "defaults to E01's _stable_hc() when omitted, which reproduces E02 (the "
        "fork's three mix() calls in qwen4_exp pass stable=_stable_hc(); they are "
        "the only mix() call sites at the pin). stable false goes to the pinned "
        "method, identical to the fork's non-stable branches; stable true runs "
        "the verbatim copy ForkGatedResidual.mix (no mechanical edits). "
        "model_compat: changes Qwen4Exp under deterministic inference. Scope "
        "(rule 9): outside target_model_active() the pinned method always runs."
    ),
)
def _gated_residual_mix(original, self, hyper_input, *, stable=None):
    if not scope.target_model_active():
        return original(self, hyper_input)
    if stable is None:
        stable = _stable_hc()
    if not stable:
        return original(self, hyper_input)
    return ForkGatedResidual.mix(self, hyper_input, stable=True)
