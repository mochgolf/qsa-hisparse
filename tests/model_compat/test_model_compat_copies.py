"""W5's copied fork code equals the fork except for its listed mechanical edits.

Rule 3: REPLACE bodies, kernels and the Marlin JIT sources are copied from
fork ee8fe158d6, not rewritten. Digests below are of the fork definitions
(decorators included, original indentation); each copy, with its mechanical
edits reverted, must hash to them. Set QSA_FORK_ROOT to a checkout of the
fork at ee8fe158d6 to re-derive the recorded digests from the fork itself.
"""

import ast
import builtins
import dis
import hashlib
import inspect
import os
import types
from pathlib import Path

import pytest
import torch

import sglang_qsa_hisparse.kernels as plugin_kernels
from sglang_qsa_hisparse.kernels import hc_mix, marlin_moe, ple_gather
from sglang_qsa_hisparse.patches.model_compat import (
    hyperconnection,
    marlin,
    quantization,
    qwen4_exp,
)

KERNELS = Path(plugin_kernels.__file__).resolve().parent
QWEN4_EXP = "sglang/srt/models/qwen4_exp.py"
HC_MIX = "sglang/srt/layers/hc_mix_triton.py"

# (plugin module, plugin qualname, fork file, fork qualname, fork sha256,
#  mechanical edits as (fork text, plugin text)).
COPIES = [
    (
        marlin,
        "fused_marlin_moe",
        "sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py",
        "fused_marlin_moe",
        "1c76340198c8bc6c7f22cbf55da8a1e3cc9261b0a294ae3f22324ac0e8665fb7",
        [
            (
                '@register_custom_op(out_shape="hidden_states")\ndef fused_marlin_moe(',
                "def fused_marlin_moe(",
            )
        ],
    ),
    (
        quantization,
        "ForkGPTQMarlinMoEKernel.process_weights_after_loading",
        "sglang/srt/hardware_backend/gpu/quantization/gptq_kernels.py",
        "GPTQMarlinMoEKernel.process_weights_after_loading",
        "1e7c2ffb994980bdbc3ff8ecb3031b346bbcfc52eeae5ad60003dd28acd1b926",
        [],
    ),
    (
        quantization,
        "ForkGPTQMarlinMoEScheme.create_weights",
        "sglang/srt/layers/quantization/gptq/schemes/gptq_moe.py",
        "GPTQMarlinMoEScheme.create_weights",
        "4e08c1f7d599c4a01f645aabb4cf99afec454de18d92db4a83060cb47c92fd3b",
        [],
    ),
    (
        quantization,
        "ForkAutoRoundConfig.apply_gptq_quant_layer",
        "sglang/srt/layers/quantization/auto_round.py",
        "AutoRoundConfig.apply_gptq_quant_layer",
        "292216b3bc7d69c628e2778d1a595037f8b3fe982ace5beaee64ce2d99e92181",
        [],
    ),
    (
        hc_mix,
        "_hc_mix_stable_persistent_kernel",
        HC_MIX,
        "_hc_mix_stable_persistent_kernel",
        "6a37dfda42edd7bb6a460409391280a5ba5490f5b00e3f27b64f0918b4353758",
        [],
    ),
    (
        hc_mix,
        "fused_hc_mix_supported",
        HC_MIX,
        "fused_hc_mix_supported",
        "68d374af1859ea65b478b1c120cf1404336724ae3d397a816aa7de546ef53bd4",
        [],
    ),
    (
        hc_mix,
        "fused_hc_mix",
        HC_MIX,
        "fused_hc_mix",
        "80c5d4cba069219a77bab40cdc59b2b11bc5ddcd03a63c2a63f330612e18f0bd",
        [],
    ),
    (
        hyperconnection,
        "ForkGatedResidual.mix",
        "sglang/srt/layers/hyperconnection.py",
        "GatedResidual.mix",
        "e2d18b73558f56d3c59007ffbfcbcc9a79e0ea83e36bbf6996462ca4616769d0",
        [],
    ),
    (
        qwen4_exp,
        "_stable_hc",
        QWEN4_EXP,
        "_stable_hc",
        "1e871fbe27afdb3285f8780280e161d9bf8c7eb5302097cccb74f3aef8d6c7d3",
        [],
    ),
    (
        qwen4_exp,
        "ForkQwen4ExpNGramEmbedding.__init__",
        QWEN4_EXP,
        "Qwen4ExpNGramEmbedding.__init__",
        "0468d50d20fb4e300d9221a44ff71046320e0b2bd52b5fc31ae674be4a6b7e12",
        [("super().__init__()", "super(Qwen4ExpNGramEmbedding, self).__init__()")],
    ),
    (
        ple_gather,
        "_gather_ple_embedding_from_pinned_kernel",
        QWEN4_EXP,
        "_gather_ple_embedding_from_pinned_kernel",
        "a7a98688f775cc3be95e38aeada10bf50f4954873871eeaeb5dabbb6a17b3bec",
        [],
    ),
    (
        qwen4_exp,
        "ForkQwen4ExpPinnedHostEmbedding.__init__",
        QWEN4_EXP,
        "Qwen4ExpPinnedHostEmbedding.__init__",
        "01617b60a4fed60a54acab1962e730db3d49a323731ba88d42a0d8851c59c5ef",
        [],
    ),
    (
        qwen4_exp,
        "ForkQwen4ExpPinnedHostEmbedding.gather",
        QWEN4_EXP,
        "Qwen4ExpPinnedHostEmbedding.gather",
        "1b3c983659fe8cb99cefae691ee6958d768ddaa1f02c6e64e2d7105d16d9d311",
        [],
    ),
    (
        qwen4_exp,
        "ForkQwen4ExpForConditionalGeneration.load_weights",
        QWEN4_EXP,
        "Qwen4ExpForConditionalGeneration.load_weights",
        "4f2f9704f7251c1c5e0d8282449277ef5e306c9fae0220f11f665b3e522dbfb8",
        [],
    ),
]

# J01: the whole op wrapper file, from its first statement after the plugin's
# module docstring; edits as (fork text, plugin text).
MARLIN_OP_FORK = "sglang/kernels/ops/moe/moe_wna16_marlin.py"
MARLIN_OP_SHA256 = "a0159f61b5217209bcbd6fcb5bda473d63675ca24083339db3a5e811f265a1b2"
MARLIN_OP_EDITS = [
    ("from typing", "from pathlib import Path\nfrom typing"),
    ("import cache_once", "import KERNEL_PATH, cache_once"),
    (
        "# Constants matching",
        '_CSRC = Path(__file__).resolve().parent / "csrc"\n\n# Constants matching',
    ),
    ('"moe_wna16_marlin",', '"qsa_hisparse_moe_wna16_marlin",'),
    (
        'cuda_files=["gemm/marlin_moe/moe_wna16_marlin.cuh"],',
        'cuda_files=[str(_CSRC / "marlin_moe" / "moe_wna16_marlin.cuh")],',
    ),
    ('"moe_wna16_marlin_gemm",\n', '"qsa_moe_wna16_marlin_gemm",\n'),
    (
        "        ],\n    )\n",
        "        ],\n"
        '        extra_include_paths=[str(KERNEL_PATH / "csrc" / "gemm" / "marlin")],\n'
        "    )\n",
    ),
    ("module.moe_wna16_marlin_gemm(", "module.qsa_moe_wna16_marlin_gemm("),
]

# J02: git blob ids at ee8fe158d6 (``git ls-tree ee8fe158d6`` in the fork).
MARLIN_HEADERS = {
    "kernel.h": "ccc47e73920dc6d81d65e3d53f22a3bb770d9dca",
    "marlin_template.h": "5f8207cbb8ee2c0178e1407f677a06908a492967",
    "moe_wna16_marlin.cuh": "2e11b1360fe9345c7a3d1ce8cc10e9229a7dd200",
    "stripe_schedule.h": "9945734d1b47abe47e2f6c05ccffed118096ac5c",
}


def _segment(source: str, qualname: str) -> str:
    node = ast.parse(source)
    for part in qualname.split("."):
        node = [
            child
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.ClassDef)) and child.name == part
        ][-1]
    lines = source.splitlines(keepends=True)
    first = min([node.lineno] + [d.lineno for d in node.decorator_list])
    return "".join(lines[first - 1 : node.end_lineno])


def _revert(text: str, edits) -> str:
    for fork_text, plugin_text in edits:
        assert text.count(plugin_text) == 1, plugin_text
        text = text.replace(plugin_text, fork_text)
    return text


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


@pytest.mark.parametrize("copy", COPIES, ids=[c[1] for c in COPIES])
def test_copied_definitions_differ_from_the_fork_only_by_listed_edits(copy):
    module, qualname, _, _, fork_sha256, edits = copy
    plugin = _segment(Path(module.__file__).read_text(), qualname)
    assert _sha256(_revert(plugin, edits)) == fork_sha256


def test_marlin_op_wrapper_differs_from_the_fork_only_by_listed_edits():
    source = Path(marlin_moe.__file__).read_text()
    body = source[source.index("from __future__ import annotations") :]
    assert _sha256(_revert(body, MARLIN_OP_EDITS)) == MARLIN_OP_SHA256


def test_marlin_headers_equal_the_fork_byte_for_byte():
    folder = KERNELS / "csrc" / "marlin_moe"
    assert sorted(p.name for p in folder.iterdir()) == sorted(MARLIN_HEADERS)
    for name, blob in MARLIN_HEADERS.items():
        assert _blob_id((folder / name).read_bytes()) == blob, name


@pytest.mark.skipif(not os.environ.get("QSA_FORK_ROOT"), reason="QSA_FORK_ROOT unset")
def test_recorded_digests_match_the_fork_checkout():
    root = Path(os.environ["QSA_FORK_ROOT"]) / "python"
    for _, _, path, qualname, fork_sha256, _ in COPIES:
        assert _sha256(_segment((root / path).read_text(), qualname)) == fork_sha256
    assert _sha256((root / MARLIN_OP_FORK).read_text()) == MARLIN_OP_SHA256
    folder = root / "sglang/kernels/jit/csrc/gemm/marlin_moe"
    for name, blob in MARLIN_HEADERS.items():
        assert _blob_id((folder / name).read_bytes()) == blob, name


# Copy globals (inventory 6, G3) ---------------------------------------------


def _global_names(code: types.CodeType) -> set[str]:
    names = {i.argval for i in dis.get_instructions(code) if i.opname == "LOAD_GLOBAL"}
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= _global_names(const)
    return names


def _functions(module):
    """Every function W5 defines in ``module``, including Fork* namespace members."""
    found = []
    for value in vars(module).values():
        value = getattr(value, "fn", value)  # Triton JITFunction
        if isinstance(value, type) and value.__module__ == module.__name__:
            found += [v for v in vars(value).values() if inspect.isfunction(v)]
        elif inspect.isfunction(value):
            value = inspect.unwrap(value)
            if value.__module__ == module.__name__:
                found.append(value)
    return found


# Bound only when CUDA is present, exactly as in the fork module.
CUDA_ONLY = {"moe_sum_reduce", "silu_and_mul", "moe_wna16_marlin_gemm"}


@pytest.mark.parametrize(
    "module",
    [marlin, quantization, hyperconnection, qwen4_exp, hc_mix, marlin_moe, ple_gather],
    ids=lambda m: m.__name__.rsplit(".", 1)[-1],
)
def test_every_global_a_copy_reads_is_bound(module):
    functions = _functions(module)
    assert functions
    missing = set()
    for function in functions:
        missing |= {
            name
            for name in _global_names(function.__code__)
            if name not in function.__globals__ and name not in vars(builtins)
        }
    expected = CUDA_ONLY if module is marlin and not marlin._is_cuda else set()
    assert missing == expected


def test_marlin_copy_imports_the_plugin_kernels():
    """Mechanical edits of J03: the stable helper and the GEMM are the plugin's."""
    from sglang_qsa_hisparse.kernels import stable_align

    assert marlin.moe_align_block_size_stable is stable_align.moe_align_block_size_stable
    tree = ast.parse(Path(marlin.__file__).read_text())
    (cuda_block,) = [
        node
        for node in tree.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "_is_cuda"
    ]
    imported = {
        alias.name: node.module
        for node in cuda_block.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert imported == {
        "moe_sum_reduce": "sgl_kernel",
        "silu_and_mul": "sglang.kernels.ops.activation.activation",
        "moe_wna16_marlin_gemm": "sglang_qsa_hisparse.kernels.marlin_moe",
    }


def test_gather_copy_launches_the_plugin_kernel():
    """Mechanical edit of E07: the int8/row-scale kernel (E04) replaces the pinned one."""
    from sglang.srt.models import qwen4_exp as pinned

    kernel = qwen4_exp._gather_ple_embedding_from_pinned_kernel
    assert kernel is ple_gather._gather_ple_embedding_from_pinned_kernel
    assert kernel is not pinned._gather_ple_embedding_from_pinned_kernel


def test_fused_marlin_moe_op_has_the_upstream_schema_under_its_own_name():
    """Mechanical edit of J03 (G4): same signature, distinct custom-op name."""
    plugin = torch.ops.sglang.qsa_hisparse_fused_marlin_moe
    upstream = torch.ops.sglang.fused_marlin_moe
    assert marlin.fused_marlin_moe_op is plugin
    assert marlin.OP_NAME == "qsa_hisparse_fused_marlin_moe"
    plugin_schema = str(plugin.default._schema).replace(marlin.OP_NAME, "fused_marlin_moe")
    assert plugin_schema == str(upstream.default._schema)


# Marlin JIT identity (inventory 3), resolved without compiling or loading -----


class _Resolved(Exception):
    pass


@pytest.fixture
def jit_resolver(monkeypatch, tmp_path):
    """Run the real ``load_jit`` up to its cache lookup, then stop."""
    from sglang.kernels.jit.utils import arch
    from sglang.kernels.jit.utils.compile import cache, loader, ninja

    def forbidden(*args, **kwargs):
        pytest.fail("compiling, loading or initializing CUDA is forbidden here")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(ninja, "build", forbidden)
    monkeypatch.setattr(loader, "_load", forbidden)
    # A fixed target instead of probing the GPU; the same for both modules.
    monkeypatch.setattr(arch, "_init_jit_cuda_arch_once", lambda: None)
    monkeypatch.setattr(arch, "_CUDA_ARCH", arch.ArchInfo(8, 9, ""), raising=False)
    monkeypatch.setattr(cache, "_target_tag", lambda: "sm89")
    monkeypatch.setattr(cache, "_environment_fingerprint", lambda: "environment")
    monkeypatch.setenv("SGLANG_JIT_CACHE_DIR", str(tmp_path))
    specs = []
    generate = ninja.generate
    monkeypatch.setattr(ninja, "generate", lambda spec: specs.append(spec) or generate(spec))

    def stop(*, scope, module_name):
        raise _Resolved(scope, module_name)

    monkeypatch.setattr(cache, "find_prebuilt", stop)

    def resolve(select, *args):
        with pytest.raises(_Resolved) as resolved:
            select(*args)
        scope, module_name = resolved.value.args
        assert module_name == specs[-1].module_name
        return specs[-1], scope

    return resolve


def _quoted_includes(path: Path) -> list[str]:
    return [
        line.split('"')[1]
        for line in path.read_text().splitlines()
        if line.startswith("#include \"")
    ]


def test_marlin_jit_module_name_cache_key_and_export_are_distinct(jit_resolver):
    from sglang.kernels.jit.utils import KERNEL_PATH
    from sglang.kernels.jit.utils.compile import toolchain
    from sglang.kernels.ops.moe import moe_wna16_marlin as in_tree

    tree, tree_scope = jit_resolver(
        in_tree._jit_moe_wna16_marlin_module, torch.bfloat16, False, False
    )
    plugin_csrc = KERNELS / "csrc" / "marlin_moe"
    marlin_headers = KERNEL_PATH / "csrc" / "gemm" / "marlin"
    for deterministic in (False, True):
        plugin, plugin_scope = jit_resolver(
            marlin_moe._jit_moe_wna16_marlin_module,
            torch.bfloat16,
            False,
            False,
            deterministic,
        )
        flag = "true" if deterministic else "false"
        assert plugin.module_name == (
            f"sgl_kernel_jit_qsa_hisparse_moe_wna16_marlin_bf16_t_false_false_{flag}"
        )
        assert tree.module_name == "sgl_kernel_jit_moe_wna16_marlin_bf16_t_false_false"
        # <root>/<target>/<module_name>/build-<build_key>: no shared directory.
        assert plugin_scope.parent != tree_scope.parent
        assert plugin_scope.name != tree_scope.name
        assert plugin.cuda_wrappers == (
            (
                "qsa_moe_wna16_marlin_gemm",
                f"moe_wna16_marlin_gemm<bf16_t, false, false, {flag}>",
            ),
        )
        assert plugin.cuda_files == (str(plugin_csrc / "moe_wna16_marlin.cuh"),)
        assert plugin.include_paths[-1] == str(marlin_headers)
        (unit,) = plugin.translation_units()
        assert f'#include "{plugin_csrc / "moe_wna16_marlin.cuh"}"' in unit.source
        assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(qsa_moe_wna16_marlin_gemm," in unit.source
    assert tree.cuda_wrappers[0][0] == "moe_wna16_marlin_gemm"
    assert tree.cuda_files != plugin.cuda_files

    # Quoted includes search the including file's directory, then the -I list
    # (toolchain base paths first, as in the generated build file). Every
    # header the copy reaches resolves to a plugin copy or to the in-tree
    # dense-Marlin headers, never to an in-tree marlin_moe sibling.
    search = toolchain.base_include_paths() + list(plugin.include_paths)
    for header in plugin_csrc.iterdir():
        for include in _quoted_includes(header):
            local = header.parent / include
            if local.exists():
                assert local.resolve().parent == plugin_csrc
                continue
            found = [
                Path(path) / include
                for path in search
                if (Path(path) / include).exists()
            ]
            assert found, (header.name, include)
            assert found[0].resolve().parent == marlin_headers.resolve()
