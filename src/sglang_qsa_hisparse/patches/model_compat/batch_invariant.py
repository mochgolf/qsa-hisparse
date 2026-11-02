"""Preserve the reference HC compiled GEMMs during target CUDA initialization.

Main #42087 adds an mm.out override. Inductor's unchanged HC fallback invokes
that overload, so this changes HC arithmetic before the first GDN layer. The
native module owns the mode, dispatcher library and bmm fallback throughout;
its existing disable function tears down exactly this library.
"""

import torch

from sglang.srt.batch_invariant_ops import batch_invariant_ops as native
from sglang.srt.utils.common import is_cuda
from sglang_qsa_hisparse import scope
from sglang_qsa_hisparse.features import MODEL_COMPAT
from sglang_qsa_hisparse.patching import patch

_MODULE = "sglang.srt.batch_invariant_ops.batch_invariant_ops"


@patch(
    f"{_MODULE}.enable_batch_invariant_mode",
    "around",
    feature=MODEL_COMPAT,
    row="D01",
    depends=(
        f"{_MODULE}.disable_batch_invariant_mode",
        f"{_MODULE}.mm_batch_invariant",
        f"{_MODULE}.addmm_batch_invariant",
        f"{_MODULE}._log_softmax_batch_invariant",
        f"{_MODULE}.mean_batch_invariant",
        f"{_MODULE}._rms_norm_aten_compat",
        f"{_MODULE}._mm_dtype_compat",
        f"{_MODULE}.bmm_batch_invariant",
        "sglang.srt.utils.common.get_dispatch_device_backend",
        "sglang.srt.utils.common.is_cuda",
    ),
    reason=(
        "Main-only reference compatibility: #42087 globally overrides mm.out, "
        "including unchanged Inductor HC GEMMs. Exact saved-input/weight GPU "
        "diagnosis on both TP ranks reproduces P with this registration and F "
        "without it. For the target model on CUDA, initialize the native MODE, "
        "LIB and optional torch.bmm fallback with all current registrations "
        "except mm.out. Native disable/idempotency and every callback remain "
        "native. Other models/devices delegate. The target fingerprint includes "
        "the full native module source, guarding this explicit registration list."
    ),
)
def _enable_batch_invariant_mode(original, enable_bmm: bool = True):
    if native._batch_invariant_MODE:
        return original(enable_bmm=enable_bmm)
    if not is_cuda() or native._is_npu or native.get_dispatch_device_backend() != "CUDA":
        return original(enable_bmm=enable_bmm)
    if not scope.target_model_active():
        return original(enable_bmm=enable_bmm)

    native._batch_invariant_MODE = True
    native._batch_invariant_LIB = torch.library.Library("aten", "IMPL")
    library = native._batch_invariant_LIB
    library.impl("aten::mm", native.mm_batch_invariant, "CUDA")
    library.impl("aten::addmm", native.addmm_batch_invariant, "CUDA")
    library.impl("aten::_log_softmax", native._log_softmax_batch_invariant, "CUDA")
    library.impl("aten::mean.dim", native.mean_batch_invariant, "CUDA")
    library.impl("aten::rms_norm", native._rms_norm_aten_compat, "CUDA")
    library.impl("aten::mm.dtype", native._mm_dtype_compat, "CUDA")
    if enable_bmm:
        library.impl("aten::bmm", native.bmm_batch_invariant, "CUDA")
        native._original_torch_bmm = torch.bmm
        torch.bmm = native.bmm_batch_invariant
