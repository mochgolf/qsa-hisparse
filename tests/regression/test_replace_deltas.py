"""Every REPLACE copy makes the fork's change to the pinned definition.

The reference fork changed each REPLACE target relative to its base commit
(FORK_BASE_COMMIT). At any pin, the plugin's copy, with its mechanical edits
reverted, must equal the three-way merge (``git merge-file``) of the fork's
change (fork base -> fork) into the pinned definition, so every changed line
is checked in place. At the fork base itself this means "verbatim fork
copy"; after a pin upgrade it means the fork's change carried onto the new
upstream body. A row whose fork change conflicts with an upstream edit lists
its hand resolution in RESOLVED; the review checks those.

Sources are read as text (fork base and fork with ``git show`` in the pinned
checkout, which is a worktree of the fork repository); nothing is imported.
"""

import ast
import json
import subprocess
import tempfile
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
    "G01": ("hisparse/graph.py", "capture_one_shape", ()),
    "G02": ("hisparse/graph.py", "load_batch", ()),
    "B02": ("hisparse/scheduler.py", "process_batch_result_prebuilt", ()),
    "B03": ("hisparse/scheduler.py", "process_batch_result_prefill", ()),
    "B04": ("hisparse/scheduler.py", "_handle_sampling_mask_abort", ()),
    "B05": ("hisparse/scheduler.py", "_handle_finish_state_updated_req", ()),
    "M01": ("hisparse/lifecycle.py", "alloc_for_extend", ()),
    "M03": ("hisparse/lifecycle.py", "release_kv_cache", ()),
    "P02": ("hisparse/scheduler.py", "add_one_req", ()),
    "S04": ("hisparse/scheduler.py", "get_next_batch_to_run", ()),
    "S06": ("hisparse/scheduler.py", "_get_new_batch_prefill_raw", ()),
    "S07": ("hisparse/scheduler.py", "on_idle", ()),
}

# row: why the copy differs from "pin + fork change" (an upstream edit overlaps
# the fork's change and was merged by hand).
RESOLVED: dict[str, str] = {
    "M03": (
        "v0.5.21 replaced the fork base's cache_finished_req handoff (and the blank "
        "line the fork's first insertion follows) with claim_kv_row / insert_req / "
        "free_kv_row / unpin / on_release, so git merge-file conflicts. Merged: the "
        "fork's lease capture (before_release, qsa_hisparse.release) stays right "
        "after the not-holds_kv early return, before the handoff (now claim_kv_row) "
        "and the logical free; its after_release stays right after "
        "mark_kv_released, and a streaming session that keeps the row still "
        "returns before it (claim_kv_row, as cache_finished_req + holds_kv did)."
    ),
    "T04": (
        "Class REPLACE by a frozen msgspec subclass: the copy holds only the "
        "fork's added decode_score_width field and its get_decode_mqa_inputs "
        "(the reference's method); the rest is inherited from the pinned class, "
        "and topk_transform's change is T02's around. The class-level edit "
        "script cannot match by construction."
    ),
    "P02": (
        "The fork split add_one_req into a new wrapper and the renamed original "
        "body _add_one_req, so on the target qualname its change replaces the "
        "whole body; the deleted side is the pinned body, which v0.5.21 edited "
        "(per_req_token_overhead, has_chunked_req, KV-shard scratch). The "
        "wrapper is the fork's, unchanged; _add_one_req is the pinned "
        "add_one_req with the fork's ignore_eos edit, checked mechanically by "
        "tests/lifecycle/test_renamed_copies.py."
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


def carried(base: list[str], fork: list[str], pinned: list[str]) -> list[str] | None:
    """The fork's change (base -> fork) merged into ``pinned``; None on conflict."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for name, lines in (("pinned", pinned), ("base", base), ("fork", fork)):
            path = Path(tmp) / name
            path.write_text("".join(line + "\n" for line in lines))
            paths.append(str(path))
        result = subprocess.run(
            ["git", "merge-file", "-p", "-q", *paths], capture_output=True, text=True
        )
    if result.returncode > 127:  # git reports errors as negative exit codes
        raise RuntimeError(result.stderr)
    return result.stdout.splitlines() if result.returncode == 0 else None


def revert(text: str, edits) -> str:
    for fork_text, plugin_text in edits:
        assert text.count(plugin_text) == 1, plugin_text
        text = text.replace(plugin_text, fork_text)
    return text


def sources(row: str) -> tuple[list[str], list[str], list[str], list[str]]:
    """(fork base, fork, pinned, plugin copy) definitions of a REPLACE row."""
    module, copy_qualname, edits, *moved = COPIES[row]
    path, qualname = fingerprint.resolve_file(pin_root() / "python", replace_rows()[row])
    fork_path = moved[0] if moved else path
    base = segment(git_show(FORK_BASE_COMMIT, fork_path), qualname)
    fork = segment(git_show(REFERENCE_FORK_COMMIT, fork_path), qualname)
    pinned = segment((pin_root() / "python" / path).read_text(), qualname)
    plugin_source = revert((PATCHES / module).read_text(), edits)
    plugin = segment(plugin_source, copy_qualname)
    return base, fork, pinned, plugin


def test_every_replace_row_is_registered():
    assert sorted(COPIES) == sorted(replace_rows())
    assert set(RESOLVED) <= set(COPIES)


@pytest.mark.parametrize("row", sorted(set(COPIES) - set(RESOLVED)))
def test_copy_makes_the_fork_change_to_the_pin(row):
    base, fork, pinned, plugin = sources(row)
    assert base != fork, f"{row}: the fork does not change its target"
    assert plugin == carried(base, fork, pinned)


@pytest.mark.parametrize("row", sorted(RESOLVED))
def test_resolved_rows_really_differ(row):
    """A RESOLVED entry is needed only while the mechanical check fails."""
    base, fork, pinned, plugin = sources(row)
    assert plugin != carried(base, fork, pinned), f"{row}: matches; drop its RESOLVED entry"


def test_carried_checks_each_change_in_place():
    base = ["def f(x):", "    a = 1", "    b = a", "    release(a)", "    return b"]
    fork = ["def f(x):", "    a = 1", "    b = a", "    release(a)", "    hook(a)", "    return b"]
    pinned = ["def f(x):", "    a = 2", "    b = a", "    release(a)", "    return b"]
    ported = ["def f(x):", "    a = 2", "    b = a", "    release(a)", "    hook(a)", "    return b"]
    assert carried(base, fork, pinned) == ported
    # The fork's line without upstream's edit, or in the wrong place (here
    # before the release it must follow), is not the merge.
    dropped = ["def f(x):", "    a = 1", "    b = a", "    release(a)", "    hook(a)", "    return b"]
    moved = ["def f(x):", "    a = 2", "    b = a", "    hook(a)", "    release(a)", "    return b"]
    assert carried(base, fork, pinned) not in (dropped, moved)
    # An upstream edit adjacent to the fork's change conflicts: hand merge.
    adjacent = ["def f(x):", "    a = 1", "    b = a", "    release(a, now=True)", "    return b"]
    assert carried(base, fork, adjacent) is None
