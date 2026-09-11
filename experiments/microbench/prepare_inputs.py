#!/usr/bin/env python3
"""Build two exact-length, local-only HiSparse test requests.

The corpus is a deterministic prefix of real files.  It is never repeated to
reach the target length; the final source file may be character-truncated and
the task instruction is always kept after that prefix.  Token counts are
checked after applying the model chat template.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
import re
import subprocess
from typing import Any


TARGET_TOKENS = 261_120
OUTPUT_TOKENS = 768
SOURCE_COMMIT = "6f25e04479152cfe68e76a16b5c00236a6f360f8"
DEFAULT_SOURCE = Path(__file__).resolve().parents[3] / "sources/sglang-hisparse-tests-20260907"
CODE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hh", ".hpp", ".py", ".pyi",
    ".rs", ".sh", ".sql", ".toml",
}
DOC_SUFFIXES = {".adoc", ".md", ".mdx", ".rst", ".txt"}
SKIP_PARTS = {".git", "__pycache__", "build", "node_modules", "target"}
CODE_PRIORITY = (
    "python/sglang/srt/arg_groups/hisparse_hook.py",
    "python/sglang/srt/managers/hisparse_coordinator.py",
    "python/sglang/srt/mem_cache/allocator/hisparse.py",
    "python/sglang/srt/mem_cache/hisparse_memory_pool.py",
    "python/sglang/srt/mem_cache/pool_host/hisparse.py",
    "python/sglang/srt/mem_cache/qsa_kv_pool.py",
    "python/sglang/srt/layers/attention/qsa/qsa_indexer.py",
    "python/sglang/kernels/ops/kvcache/hisparse.py",
    "python/sglang/kernels/ops/attention/qsa_indexer.py",
    "python/sglang/kernels/jit/csrc/kvcacheio/hisparse.cuh",
    "python/sglang/kernels/jit/csrc/kvcacheio/hisparse_spec.cuh",
    "python/sglang/kernels/jit/csrc/attention/qsa_indexer.cuh",
)
DOC_PRIORITY = (
    "docs/docs/advanced_features/hisparse_guide.mdx",
    "docs/docs/advanced_features/server_arguments.mdx",
    "docs/docs/advanced_features/speculative_decoding.mdx",
    "docs/docs/references/environment_variables.mdx",
    "docs/docs/advanced_features/sgl_model_gateway.mdx",
)

TASKS = {
    "A": (
        "Task instruction: read this local SGLang source context as a real code "
        "review corpus. Identify the control flow and data structures relevant "
        "to HiSparse, QSA selection, KV-cache allocation, and host/device "
        "movement. Answer with a concise implementation risk list grounded in "
        "the supplied files; do not invent files or claim that CPU accounting "
        "is a GPU benchmark."
    ),
    "B": (
        "Task instruction: read this local technical documentation context as a "
        "real documentation corpus. Explain the operational assumptions, limits, "
        "configuration boundaries, and validation steps relevant to HiSparse and "
        "long-context KV caching. Answer with a concise deployment checklist "
        "grounded in the supplied documents; distinguish documented behavior "
        "from hypotheses and do not invent measurements."
    ),
}


def load_tokenizer(path: Path):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=True
    )


def source_files(root: Path, kind: str, commit: str | None = None) -> list[Path]:
    suffixes = CODE_SUFFIXES if kind == "A" else DOC_SUFFIXES
    root_names = ("python/sglang", "test/registered", "rust") if kind == "A" else (
        "docs/docs", "docs/cookbook"
    )
    files: list[Path] = []
    if commit:
        names = subprocess.check_output(
            ["git", "-C", str(root), "ls-tree", "-r", "--name-only", commit, "--", *root_names],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
        files = [root / name for name in names]
    else:
        roots = [root / name for name in root_names]
        for base in roots:
            if not base.is_dir():
                continue
            for path in base.rglob("*"):
                if path.is_file() and not path.is_symlink():
                    files.append(path)
    files = [
        path
        for path in files
        if path.suffix.lower() in suffixes
        and not (set(path.relative_to(root).parts) & SKIP_PARTS)
    ]
    unique = sorted(set(files), key=lambda path: path.relative_to(root).as_posix())
    priority = CODE_PRIORITY if kind == "A" else DOC_PRIORITY
    by_relative = {path.relative_to(root).as_posix(): path for path in unique}
    prioritized = [by_relative[value] for value in priority if value in by_relative]
    return prioritized + [path for path in unique if path not in prioritized]


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def worktree_diff(root: Path) -> list[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "diff", "--name-only"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except (OSError, subprocess.SubprocessError):
        return []


def read_corpus(
    root: Path, kind: str, commit: str | None
) -> tuple[str, list[dict[str, Any]], list[str]]:
    """Read enough unique source text to make an exact target possible."""

    pieces: list[str] = []
    records: list[dict[str, Any]] = []
    diff_paths = set(worktree_diff(root))
    total_chars = 0
    # A source-to-token ratio below 8:1 is ample for both code and prose.  If
    # a tokenizer revision is unusually sparse, the loop simply reads on.
    minimum_chars = TARGET_TOKENS * 8
    for path in source_files(root, kind, commit):
        relative = path.relative_to(root).as_posix()
        raw: bytes
        if commit:
            try:
                raw = subprocess.check_output(
                    ["git", "-C", str(root), "show", f"{commit}:{relative}"],
                    stderr=subprocess.DEVNULL,
                )
            except (OSError, subprocess.SubprocessError):
                raw = path.read_bytes()
        else:
            raw = path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        marker = f"\n\n===== LOCAL FILE: {relative} =====\n"
        pieces.append(marker + text)
        records.append(
            {
                "path": relative,
                "bytes": len(raw),
                "chars": len(text),
                "marker_chars": len(marker),
            }
        )
        total_chars += len(marker) + len(text)
        if total_chars >= minimum_chars:
            break
    corpus = "".join(pieces)
    if len(corpus) < minimum_chars:
        raise RuntimeError(
            f"{kind}: only {len(corpus)} source characters; need at least {minimum_chars}"
        )
    return corpus, records, sorted(diff_paths.intersection(record["path"] for record in records))


def render(tokenizer, content: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        tokenize=True,
        truncation=False,
        enable_thinking=False,
    )
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise RuntimeError("chat template returned a batch for one prompt")
        ids = ids[0]
    if not isinstance(ids, list) or not all(isinstance(value, int) for value in ids):
        raise RuntimeError("chat template did not return a flat integer list")
    return ids


def _suffix_start(values: list[int], suffix: list[int]) -> int:
    if not suffix or len(suffix) > len(values):
        raise RuntimeError("cannot locate an empty or oversized token suffix")
    start = len(values) - len(suffix)
    if values[start:] != suffix:
        raise RuntimeError("chat template suffix did not remain stable")
    return start


def token_cut(
    tokenizer, body: str, task_marker: str, task_text: str
) -> tuple[list[int], dict[str, int]]:
    """Render once, then splice a unique token prefix around the task suffix.

    Keeping the actual rendered chat tail avoids a second template guess.  The
    resulting IDs are authoritative; the decoded prompt is a review aid and is
    intentionally not re-encoded as a correctness check.
    """

    full = render(tokenizer, body)
    empty = render(tokenizer, "")
    tail_len = 0
    for left, right in zip(reversed(full), reversed(empty)):
        if left != right:
            break
        tail_len += 1
    if tail_len == 0:
        raise RuntimeError("could not identify the fixed chat-template tail")
    content_ids = full[:-tail_len]
    tail_ids = full[-tail_len:]
    task_ids = tokenizer.encode(task_marker, add_special_tokens=False)
    task_start = -1
    for candidate in range(len(content_ids) - len(task_ids), -1, -1):
        if content_ids[candidate : candidate + len(task_ids)] == task_ids:
            task_start = candidate
            break
    # A BPE merge can absorb the marker's leading newline into the preceding
    # source token.  The instruction text itself is stable and is sufficient
    # as the retained suffix in that case.
    if task_start < 0:
        task_ids = tokenizer.encode(task_text, add_special_tokens=False)
        task_start = -1
        for candidate in range(len(content_ids) - len(task_ids), -1, -1):
            if content_ids[candidate : candidate + len(task_ids)] == task_ids:
                task_start = candidate
                break
    if task_start < 0:
        raise RuntimeError("could not locate the task marker in rendered IDs")
    if TARGET_TOKENS <= len(tail_ids) + len(task_ids):
        raise RuntimeError("target is too short for chat tail and task instruction")
    keep_content = TARGET_TOKENS - len(tail_ids)
    keep_prefix = keep_content - len(task_ids)
    if keep_prefix > task_start:
        raise RuntimeError("source corpus is too short before the task suffix")
    ids = content_ids[:keep_prefix] + content_ids[task_start:] + tail_ids
    if len(ids) != TARGET_TOKENS:
        raise AssertionError("token splice did not produce the target length")
    return ids, {
        "rendered_full_tokens": len(full),
        "chat_tail_tokens": len(tail_ids),
        "task_suffix_tokens": len(task_ids),
        "source_prefix_tokens_kept": keep_prefix,
        "task_start_token": task_start,
    }


def request(label: str, ids: list[int]) -> dict[str, Any]:
    return {
        "rid": f"hisparse-long-{label}",
        "input_ids": ids,
        "sampling_params": {
            "temperature": 0,
            "top_p": 1.0,
            "top_k": 1,
            "max_new_tokens": OUTPUT_TOKENS,
            "ignore_eos": True,
        },
        "stream": False,
        "log_metrics": True,
        "return_logprob": False,
        "logprob_start_len": -1,
    }


def build_one(
    tokenizer, tokenizer_path: Path, root: Path, label: str, commit: str | None
) -> tuple[dict[str, Any], dict[str, Any], str]:
    corpus, records, changed_candidates = read_corpus(root, label, commit)
    context_header = (
        f"Local {label} context. The following material is a deterministic, "
        "non-repeated prefix of real files from the stated local checkout.\n"
    )
    task_marker = f"\n\n===== TASK INSTRUCTION =====\n{TASKS[label]}\n"
    body = context_header + corpus + task_marker
    ids, token_info = token_cut(tokenizer, body, task_marker, TASKS[label])
    content = tokenizer.decode(ids, skip_special_tokens=False)
    input_paths = re.findall(r"===== LOCAL FILE: (.*?) =====", content)
    if not input_paths:
        raise RuntimeError(f"{label}: decoded input IDs contain no source file marker")
    metadata = {
        "label": label,
        "kind": "real_sglang_code" if label == "A" else "local_technical_docs",
        "target_input_tokens": TARGET_TOKENS,
        "actual_input_tokens": len(ids),
        "output_tokens": OUTPUT_TOKENS,
        "sampling": "greedy: temperature=0, top_p=1, top_k=1, ignore_eos=true",
        "tokenizer_path": str(tokenizer_path),
        "chat_template": "apply_chat_template(add_generation_prompt=true, enable_thinking=false)",
        "source_path": str(root.resolve()),
        "source_commit": commit,
        "source_head_observed": git_commit(root),
        "source_head_used_for_content": False,
        "selection_rule": (
            "use the fixed HiSparse/QSA priority list, then sort remaining "
            "relative paths bytewise; include eligible UTF-8 files once with a "
            "file marker; render once and take a token prefix of the concatenated "
            "corpus; keep the complete task instruction and chat generation tail; "
            "no repetition"
        ),
        "source_files_read": records,
        "source_files_candidate_count": len(records),
        "source_files_in_input_ids": input_paths,
        "source_files_changed_in_worktree": changed_candidates,
        "source_read_mode": "git_show_at_source_commit" if commit else "worktree",
        "corpus_chars_available": len(corpus),
        "corpus_tokens_truncated_at": token_info["source_prefix_tokens_kept"],
        "task_marker": task_marker,
        "task_instruction": TASKS[label],
        "task_instruction_chars": len(TASKS[label]),
        "prompt_chars": len(content),
        "token_splice": token_info,
        "prompt_text_is_decode_of_input_ids": True,
        "prompt_reencode_required": False,
    }
    return request(label, ids), metadata, content


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    if not args.source_root.is_dir():
        raise SystemExit(f"source root is missing: {args.source_root}")

    tokenizer = load_tokenizer(args.tokenizer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # The shared checkout may move while the observer agent develops its
    # runner.  Read this fixed clean base through ``git show`` so inputs are
    # reproducible and never inherit concurrent worktree/branch changes.
    source_commit = SOURCE_COMMIT
    manifest = {
        "schema_version": "qwen38-hisparse-test-inputs-v1",
        "offline_only": True,
        "target_input_tokens": TARGET_TOKENS,
        "output_tokens": OUTPUT_TOKENS,
        "source_root": str(args.source_root.resolve()),
        "tokenizer_path": str(args.tokenizer.resolve()),
        "inputs": {},
    }
    for label in ("A", "B"):
        built, metadata, content = build_one(
            tokenizer, args.tokenizer, args.source_root, label, source_commit
        )
        if len(built["input_ids"]) != TARGET_TOKENS:
            raise AssertionError(f"{label}: exact token count check failed")
        (args.output_dir / f"request-{label}.json").write_text(
            json.dumps(built, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / f"prompt-{label}.txt").write_text(content, encoding="utf-8")
        (args.output_dir / f"input-{label}-metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        manifest["inputs"][label] = {
            **metadata,
            "request_path": str((args.output_dir / f"request-{label}.json").resolve()),
            "prompt_path": str((args.output_dir / f"prompt-{label}.txt").resolve()),
            "metadata_path": str((args.output_dir / f"input-{label}-metadata.json").resolve()),
        }
        print(f"{label}: {len(built['input_ids'])} tokens, {len(metadata['source_files_read'])} candidate files")
    (args.output_dir / "inputs-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
