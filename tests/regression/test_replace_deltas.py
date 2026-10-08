"""Every REPLACE copy makes the fork's change to the pinned definition.

The reference fork changed each REPLACE target relative to its base commit
(FORK_BASE_COMMIT). At any pin, the plugin's copy, with its mechanical edits
reverted, must differ from the pinned definition by exactly the edit script
that takes the fork base to the fork. At the fork base itself this means
"verbatim fork copy"; after a pin upgrade it means the fork's change carried
onto the new upstream body. A row whose fork change overlaps an upstream edit
lists its hand resolution in RESOLVED; the review checks those.

Sources are read as text (fork base and fork with ``git show`` in the pinned
checkout, which is a worktree of the fork repository); nothing is imported.
"""

import ast
import difflib
import json
import subprocess
import textwrap
from pathlib import Path

import pytest

from sglang_qsa_hisparse import FORK_BASE_COMMIT, REFERENCE_FORK_COMMIT, fingerprint

REPO = Path(__file__).resolve().parents[2]
PATCHES = REPO / "src" / "sglang_qsa_hisparse" / "patches"
MANIFEST = REPO / "src" / "sglang_qsa_hisparse" / "manifest.json"

# row: (patch module under patches/, qualname of the copy in it, mechanical
#       edits as (fork text, plugin text) including any rename of the copy
#       [, the target's module file at the fork if upstream moved it])
COPIES: dict[str, tuple] = {
    "A03": ("../kernels/qsa_sparse_attn.py", "sparse_gqa_fwd_interface_triton", []),
    "A05": ("../kernels/qsa_sparse_attn.py", "sparse_gqa_fwd_interface_triton_ck", []),
    "A09": ("../kernels/qsa_sparse_attn.py", "qwen_sparse_kv_extraction_compact_triton", []),
    "E03": (
        "model_compat/qwen4_exp.py",
        "ForkQwen4ExpNGramEmbedding.__init__",
        [("super().__init__()\n", "super(Qwen4ExpNGramEmbedding, self).__init__()\n")],
    ),
    "E06": ("model_compat/qwen4_exp.py", "ForkQwen4ExpPinnedHostEmbedding.__init__", []),
    "E07": ("model_compat/qwen4_exp.py", "ForkQwen4ExpPinnedHostEmbedding.gather", []),
    "E08": (
        "model_compat/qwen4_exp.py",
        "ForkQwen4ExpForConditionalGeneration.load_weights",
        [],
    ),
    "J03": (
        "model_compat/marlin.py",
        "fused_marlin_moe",
        [('@register_custom_op(out_shape="hidden_states")\ndef fused_marlin_moe(', "def fused_marlin_moe(")],
    ),
    "Q01": ("model_compat/qsa_attention.py", "_resolve_flash_attn_varlen_func", []),
    "Q08": ("model_compat/qsa_attention.py", "forward_extend", []),
    "Q10": ("model_compat/qsa_attention.py", "_forward_trtllm_sparse", []),
    "Q11": ("model_compat/qsa_attention.py", "forward_decode", []),
    "Q12": ("model_compat/qsa_attention.py", "_forward_paged_attention", []),
    "T03": ("model_compat/qsa_attention.py", "select_decode_tokens", []),
    "T04": ("model_compat/qsa_attention.py", "QSAIndexerMetadata", []),
    "Z01": (
        "model_compat/quantization.py",
        "ForkGPTQMarlinMoEKernel.process_weights_after_loading",
        [],
    ),
    "Z02": ("model_compat/quantization.py", "ForkGPTQMarlinMoEScheme.create_weights", []),
    "Z03": ("model_compat/quantization.py", "ForkAutoRoundConfig.apply_gptq_quant_layer", []),
    "K02": (
        "hisparse/pools.py",
        "_build_hybrid_linear_kv_pool",
        (
            (
                "            from sglang.srt.mem_cache.qsa_hisparse.slots import QSAHiSparseSlots",
                "            from sglang_qsa_hisparse.hisparse.slots import QSAHiSparseSlots",
            ),
        ),
    ),
    "R02": ("hisparse/graph.py", "_forward_raw", ()),
    "G01": ("hisparse/graph.py", "capture_one_shape", ()),
    "G02": ("hisparse/graph.py", "load_batch", ()),
    "G03": ("hisparse/graph.py", "execute", ()),
}

# row: why the copy differs from "pin + fork change" (an upstream edit overlaps
# the fork's change and was merged by hand).
RESOLVED: dict[str, str] = {
    "E03": (
        "v0.5.21 already builds the n-gram table on meta for ple_offload_embedding "
        "(the fork's change, in upstream's form via a local offload_embedding) and "
        "now wraps it in Qwen4ExpPinnedHostEmbedding inside this method, through a "
        "local ngram_embedding. Carried over: the int8 params_dtype branch, and "
        "ple_row_scale_mode plus the int8_row checks on the local, placed before "
        "the wrapping because the E06 copy reads ple_row_scale_mode when it wraps "
        "(the fork wrapped after this method returned, so its order is unchanged)."
    ),
    "Q12": (
        "v0.5.21 adds a ROCm branch after extraction (pinned "
        "sparse_gqa_packed_decode_triton, early return) and resolves "
        "flash-attention after it; the fork re-indents that whole region into "
        "NVTX ranges. Merged: the fork's ranges and changes, with the ROCm branch "
        "after the qsa.fa2_extract range and the resolver call moved from the "
        "qsa.fa2_metadata_scratch range to just before qsa.fa2_attention (still "
        "before the FA2 graph check, as in the fork). On ROCm the branch returns "
        "before capture_decode, i.e. HiSparse decode capture is not wired on ROCm."
    ),
    "T04": (
        "Class REPLACE by a frozen msgspec subclass: the copy holds only the "
        "fork's added decode_score_width field and its get_decode_mqa_inputs "
        "(the pinned method, unchanged at v0.5.21, plus the fork's change); the "
        "rest is inherited from the pinned class, and topk_transform's change is "
        "T02's around. The class-level edit script cannot match by construction."
    ),
}


def replace_rows() -> dict[str, str]:
    rows = json.loads(MANIFEST.read_text())["rows"]
    return {
        row: patch["target"]
        for row, spec in rows.items()
        for patch in spec["patches"]
        if patch["hook_type"] == "replace"
    }


def pin_root() -> Path:
    import sglang

    return Path(sglang.__file__).resolve().parents[2]


def _plugin_decorator(node: ast.expr) -> bool:
    func = node.func if isinstance(node, ast.Call) else node
    return isinstance(func, ast.Name) and func.id in {"patch", "attach"}


def segment(source: str, qualname: str) -> list[str]:
    """The definition with its decorators (except the plugin's own
    ``@patch``/``@attach`` declarations), dedented, as lines."""
    node, _ = fingerprint._locate(ast.parse(source), qualname)
    kept = [d for d in node.decorator_list if not _plugin_decorator(d)]
    first = min([node.lineno] + [d.lineno for d in kept])
    lines = source.splitlines(keepends=True)[first - 1 : node.end_lineno]
    return textwrap.dedent("".join(lines)).splitlines()


def git_show(commit: str, path: str) -> str:
    return subprocess.run(
        ["git", "--no-optional-locks", "-C", str(pin_root()), "show", f"{commit}:python/{path}"],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip


def changes(a: list[str], b: list[str]) -> list[tuple[list[str], list[str]]]:
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return [
        (a[i1:i2], b[j1:j2])
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    ]


def revert(text: str, edits) -> str:
    for fork_text, plugin_text in edits:
        assert text.count(plugin_text) == 1, plugin_text
        text = text.replace(plugin_text, fork_text)
    return text


def deltas(row: str) -> tuple[list, list]:
    """(fork base -> fork, pin -> plugin copy) edit scripts for a REPLACE row."""
    module, copy_qualname, edits, *moved = COPIES[row]
    path, qualname = fingerprint.resolve_file(pin_root() / "python", replace_rows()[row])
    fork_path = moved[0] if moved else path
    base = segment(git_show(FORK_BASE_COMMIT, fork_path), qualname)
    fork = segment(git_show(REFERENCE_FORK_COMMIT, fork_path), qualname)
    pinned = segment((pin_root() / "python" / path).read_text(), qualname)
    plugin_source = revert((PATCHES / module).read_text(), edits)
    plugin = segment(plugin_source, copy_qualname)
    return changes(base, fork), changes(pinned, plugin)


def test_every_replace_row_is_registered():
    assert sorted(COPIES) == sorted(replace_rows())
    assert set(RESOLVED) <= set(COPIES)


@pytest.mark.parametrize("row", sorted(set(COPIES) - set(RESOLVED)))
def test_copy_makes_the_fork_change_to_the_pin(row):
    fork_change, plugin_change = deltas(row)
    assert fork_change, f"{row}: the fork does not change its target"
    assert plugin_change == fork_change


@pytest.mark.parametrize("row", sorted(RESOLVED))
def test_resolved_rows_really_differ(row):
    """A RESOLVED entry is needed only while the mechanical check fails."""
    fork_change, plugin_change = deltas(row)
    assert plugin_change != fork_change, f"{row}: matches; drop its RESOLVED entry"


def test_changes_reports_an_edit_script():
    base = ["def f(x):", "    a = 1", "    return a"]
    fork = ["def f(x):", "    a = 1", "    hook(a)", "    return a"]
    pinned = ["def f(x):", "    a = 2", "    return a"]
    carried = ["def f(x):", "    a = 2", "    hook(a)", "    return a"]
    dropped = ["def f(x):", "    a = 1", "    hook(a)", "    return a"]
    assert changes(pinned, carried) == changes(base, fork) == [([], ["    hook(a)"])]
    assert changes(pinned, dropped) != changes(base, fork)
