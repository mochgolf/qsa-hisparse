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
    "M03": ("hisparse/lifecycle.py", "release_kv_cache", ()),
    "S04": ("hisparse/scheduler.py", "get_next_batch_to_run", ()),
    "S07": ("hisparse/scheduler.py", "on_idle", ()),
}

# row: why the copy differs from "pin + fork change" (an upstream edit overlaps
# the fork's change and was merged by hand).
RESOLVED: dict[str, str] = {
    "G01": (
        "Main removes PD multiplexing/stream_idx (#43898), wraps both setup "
        "and each retained run_once in forward_context(forward_batch=...) "
        "(#44994), and adds capture_head_hooks, UOCR prologue and forked "
        "metadata-prep joining. Keep the entire new upstream body; move its "
        "pure shape-key construction before the outer context with the new "
        "three-argument key API, enter QSA graph_capture after capture_prepare, "
        "and chain after_graph_warmup. The changed context header and moved "
        "key conflict with production's corresponding QSA insertion."
    ),
    "G02": (
        "Main removes PD stream keys (#43898), carries ragged layouts directly "
        "on ForwardBatch (#34031/#34038), and adds early verify metadata, "
        "input-restaging bookkeeping and _init_replay_attn_metadata. Keep "
        "the entire upstream body and its staging/skip_slots behavior; insert "
        "the QSA prepare immediately before fill_from after early-stage "
        "resolution, retain the external-preplanning rejection and guard "
        "num_real_reqs.fill_. Production's prepare insertion at this same "
        "fill_from boundary conflicts with the new staging statements."
    ),
    "A05": (
        "Main #45753 adds max_q to avoid a host sync in CP calls. The plugin "
        "accepts that optional host length while retaining the reference FP8 "
        "descale kernel and its KV_IS_FP8 parameter. Main's SLOT_INDICES=False "
        "belongs to its paged-prefill kernel, not the plugin-owned packed kernel."
    ),
    "E08": (
        "Selectively adopt #38900's loader lifetime fix: a nonlocal boolean "
        "replaces the nested shard loader's self-reference for warning state. "
        "The snapshot of pre-repack parameters is released on return without "
        "waiting for cyclic GC; weight loading and numerical operations are unchanged."
    ),
    "J03": (
        "Main adds no_combine for the MoE runner contract; return the existing "
        "router-weighted expert tensor before reduction when requested. Keep "
        "the reference GEMM alignment/workspace and activation implementations "
        "with the plugin-owned deterministic Marlin JIT. #45462's inactive-count "
        "prefix and extra GEMM flags require its newer native kernel ABI and "
        "cannot be passed to the reference kernel. The removed GPT-OSS torch "
        "activation helper is retained locally to preserve reference arithmetic."
    ),
    "Q10": (
        "#44114 keeps FP8 sparse scratch on SM100. Retain the reference BF16 "
        "scratch so plugin extraction applies non-unit KV descales once before "
        "TRTLLM; rounding descaled values back into FP8 changes that contract."
    ),
    "Q11": (
        "#46990 replaces the upstream store call with a typed KV layer view; "
        "the plugin retains _store_kv for physical HiSparse locations and FP8 "
        "descale handling, followed by its existing after_store lease callback."
    ),
    "M03": (
        "The merge applies cleanly but keeps production's mamba free for a "
        "tree without mamba support (M05's ChunkCache, admitted by M06). At "
        "v0.5.22 M05 builds QSAHostPrefixCache, a disabled UnifiedRadixCache "
        "that owns the slot (checkpoint frees a finished request's, on_release "
        "an unchecked one's), and create_tree_cache rejects every other "
        "hybrid-SSM tree without mamba support, so the block is dropped with "
        "its comment: it could only free the slot a second time. v0.5.22's "
        "checkpoint= keyword, unlock(req.lock) and on_release(checkpointed=) "
        "are kept as merged. At main 582168dd7c the early state-slot release "
        "and final request-row release retain upstream's kv_storage_of "
        "delegation (#46988/#46989); the QSA lease still drains between "
        "checkpoint and free_kv_row and finishes after mark_kv_released."
    ),
    "Q08": (
        "v0.5.22 (#41933) replaced the chunk-prefill index_select gather, where "
        "the reference reads raw slots from qsa_hisparse.prefill_slots, by "
        "pack_qsa_prefill_kv(..., output_dtype=q.dtype). Merge: without a "
        "runtime the copy keeps v0.5.22's pack but drops output_dtype, so FP8 "
        "K/V reach the reference's FP8-aware chunk kernel raw and its "
        "**_kv_descale_kwargs are applied exactly once (the cast would hand "
        "the kernel unscaled BF16 and its non-FP8 path ignores the descales); "
        "the packed bytes equal the reference's gather. With a runtime the copy "
        "keeps the reference's prefill_slots gather (req_pool_indices.tolist(); "
        "the runtime requires the prefill graph disabled, so the reference's "
        "breakable-graph slot table, removed by #41933, never applies there). "
        "tests/qsa/test_prefill_pack_descale.py. Main #46990's typed KV layer "
        "reads are used for cache buffers; _store_kv retains physical location "
        "and FP8 scaling semantics."
    ),
    "T03": (
        "v0.5.22 (#43463) bounds the decode MQA by the context length when "
        "fast_topk selects (use_fast_topk, min_scored_len), editing the line of "
        "the reference's deterministic predicate. Merge: the deterministic flag "
        "is read before qsa_mqa_decode and folded into use_fast_topk, so the "
        "bound (unscored short rows, unset logits past each length) applies "
        "only with fast_topk; the stable top-k reads every score, so "
        "deterministic decode scores all rows as in the reference. "
        "Main #42087's batch-invariant predicate also disables fast_topk; the "
        "reference deterministic predicate still selects its stable kernel. "
        "tests/qsa/test_qsa_patches.py::"
        "test_deterministic_decode_skips_native_topk_only_for_the_target_model"
    ),
    "T04": (
        "Class REPLACE by a frozen msgspec subclass: the copy holds only the "
        "fork's added decode_score_width field and its get_decode_mqa_inputs "
        "(the reference's method); the rest is inherited from the pinned class, "
        "and topk_transform's change is T02's around. The class-level edit "
        "script cannot match by construction."
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
