#!/usr/bin/env python3
"""Offline F-vs-P comparator for Phase 2 evidence (docs/baseline.md step 15).

    compare.py <fork-arm-dir> <plugin-arm-dir>

Both arm directories use the same layout; an item present in one arm must be
present in the other. Every value is compared exactly (canonical JSON, so
floats compare bit for bit). For each item the first divergence is printed;
the exit status is 1 if any item differs.

- ``*.json`` (top level, except server_info.json): HTTP harness reports. The
  REPORT_KEYS values are compared wherever they occur. The lifecycle
  harness's deliberate abort (SKIPPED_KEYS) produces a timing-dependent
  number of tokens and is not compared.
- ``server_info.json``: SERVER_INFO paths (computed sizes; measured memory is
  printed for information only).
- ``server.log``: LOG_FIGURES per TP rank, in sorted order.
- ``events/rank-N.jsonl`` (fork runtime ledger): the first event's
  FIXED_POOL_FIELDS, every ``graph_capture_complete`` record without times
  and pointers, and the prefix event sequence with the fork ledger harness's
  fields (PREFIX_FIELDS).
- ``observer/rank-N.jsonl`` (observer.py): every record. Within each arm,
  every restore must also equal the capture of the checkpoint it restored.

Request IDs and salts embed ``time_ns`` and differ between arms: rids are
renamed by order of first appearance in each file, and salts are never
compared.
"""

import json
import re
import sys
from pathlib import Path

REPORT_KEYS = frozenset(
    {
        "output_ids",
        "cached_tokens",
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens_details",
        "finish_reason",
        "input_token_logprobs",
        "output_token_logprobs",
        "input_top_logprobs",
        "output_top_logprobs",
        "passed",
    }
)
SKIPPED_KEYS = frozenset({"aborted_response", "abort_result"})
# Fork test/manual/qsa_hisparse_prefix_ledger.py: FIXED_POOL_FIELDS and the
# fields of its TP prefix sequence (plus "tokens").
FIXED_POOL_FIELDS = (
    "staging_capacity_tokens",
    "raw_pool_size_tokens",
    "raw_backing_size_tokens",
    "ring_reserved_tokens",
    "lease_capacity",
    "logical_capacity",
    "raw_bytes",
    "index_bytes",
    "host_reserved_bytes",
    "hot_reserved_bytes",
    "workspace_bytes",
    "mamba_bytes",
)
PREFIX_FIELDS = (
    "event",
    "prefix_cache_host_bytes",
    "prefix_cache_entries",
    "prefix_cache_evictions",
    "prefix_cache_epoch",
    "rid",
    "generation",
    "req_pool_idx",
    "lease_slot",
    "forward_id",
)
VOLATILE_KEYS = frozenset(
    {"time_ns", "cuda_allocated", "cuda_reserved", "host_alloc_wall_ms", "graph_pool"}
)
SERVER_INFO = ("max_total_num_tokens", "max_req_input_len")
MEMORY_USAGE = ("kvcache", "token_capacity")
MEASURED_MEMORY = ("weight", "graph", "startup_available")
LOG_FIGURES = re.compile(
    r"\b(fixed_bytes|logical_bytes_per_token|max_total_num_tokens)=(\d+)"
)
TP_RANK = re.compile(r"\bTP(\d+)\b")


def canonical(value):
    return json.dumps(value, sort_keys=True)


def first_divergence(fork, plugin):
    """First differing (label, value) item of two sequences, or None."""
    for i, (f, p) in enumerate(zip(fork, plugin)):
        if f != p:
            return f"item {i}: F {short(f)} != P {short(p)}"
    if len(fork) != len(plugin):
        longer, name = (fork, "F") if len(fork) > len(plugin) else (plugin, "P")
        return (
            f"F has {len(fork)} items, P has {len(plugin)}; "
            f"first unmatched ({name}): {short(longer[min(len(fork), len(plugin))])}"
        )
    return None


def short(item, limit=300):
    text = item if isinstance(item, str) else canonical(item)
    return text if len(text) <= limit else text[:limit] + "..."


class Renamer:
    """Rename distinct values in order of first appearance (None stays None)."""

    def __init__(self, prefix):
        self.prefix, self.names = prefix, {}

    def __call__(self, value):
        if value is None:
            return None
        return self.names.setdefault(value, f"{self.prefix}{len(self.names)}")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def report_items(path):
    items = []

    def walk(value, where):
        if isinstance(value, dict):
            for key in sorted(value):
                if key in SKIPPED_KEYS:
                    continue
                if key in REPORT_KEYS:
                    items.append((f"{where}.{key}", canonical(value[key])))
                else:
                    walk(value[key], f"{where}.{key}")
        elif isinstance(value, list):
            for i, item in enumerate(value):
                walk(item, f"{where}[{i}]")

    walk(json.loads(path.read_text()), "")
    return items


def server_info_items(path, info):
    data = json.loads(path.read_text())
    items = [(key, canonical(data.get(key))) for key in SERVER_INFO]
    for i, state in enumerate(data.get("internal_states") or []):
        usage = state.get("memory_usage") or {}
        items += [
            (f"internal_states[{i}].memory_usage.{key}", canonical(usage.get(key)))
            for key in MEMORY_USAGE
        ]
        info.append(
            f"{path.parent.name}: measured memory "
            + canonical({key: usage.get(key) for key in MEASURED_MEMORY})
        )
    return items


def log_items(path):
    found = []
    for line in path.read_text(errors="replace").splitlines():
        rank = TP_RANK.search(line)
        for key, value in LOG_FIGURES.findall(line):
            found.append((rank.group(1) if rank else "-", key, int(value)))
    return [(f"TP{rank} {key}", value) for rank, key, value in sorted(found)]


def stripped(value):
    if isinstance(value, dict):
        return {
            key: stripped(item)
            for key, item in value.items()
            if key not in VOLATILE_KEYS
            and key != "ptr"
            and not key.endswith(("_ptr", "_ptrs"))
        }
    if isinstance(value, list):
        return [stripped(item) for item in value]
    return value


def ledger_items(path):
    rows = read_jsonl(path)
    items = [
        ("fixed_pools", {key: rows[0].get(key) for key in FIXED_POOL_FIELDS})
    ] if rows else []
    rid = Renamer("r")
    prefix = 0
    for row in rows:
        if row["event"] == "graph_capture_complete":
            items.append(("graph_capture_complete", stripped(row)))
        elif row["event"].startswith("prefix_"):
            record = {key: row.get(key) for key in PREFIX_FIELDS}
            record["rid"] = rid(record["rid"])
            record["tokens"] = row.get("checkpoint_tokens", row.get("reused_tokens"))
            items.append((f"prefix event {prefix}", record))
            prefix += 1
    return items


def observer_items(path):
    rid = Renamer("r")
    items = []
    for i, row in enumerate(read_jsonl(path)):
        row["rid"] = rid(row["rid"])
        items.append((f"record {i} {row['event']}", row))
    return items


STATE = ("tokens", "token_sha256", "segments", "pending", "rope", "mamba")


def observer_consistency(path):
    """Within one arm, each restore must equal its checkpoint's capture."""
    captures = {}
    for i, row in enumerate(read_jsonl(path)):
        state = {key: row[key] for key in STATE}
        if row["event"] == "capture":
            captures[row["checkpoint"]] = state
            continue
        captured = captures.get(row["checkpoint"])
        if captured is None:
            return f"record {i}: restore of unobserved checkpoint {row['checkpoint']}"
        if captured != state:
            key = next(k for k in STATE if captured[k] != state[k])
            return (
                f"record {i}: restore of checkpoint {row['checkpoint']} differs from "
                f"its capture in {key}: {short(captured[key])} != {short(state[key])}"
            )
    return None


def items_of(name, path, info):
    if name == "server_info.json":
        return server_info_items(path, info)
    if name == "server.log":
        return log_items(path)
    if name.startswith("events/"):
        return ledger_items(path)
    if name.startswith("observer/"):
        return observer_items(path)
    return report_items(path)


def arm_files(arm):
    names = {p.name for p in arm.glob("*.json")} | {
        p.name for p in arm.glob("server.log")
    }
    for sub in ("events", "observer"):
        names |= {f"{sub}/{p.name}" for p in arm.glob(f"{sub}/rank-*.jsonl")}
    return names


def compare(fork, plugin):
    """Return (lines, number of differing items)."""
    lines, info, failed = [], [], 0
    names = sorted(arm_files(fork) | arm_files(plugin))
    if not names:
        raise SystemExit(f"nothing to compare in {fork} and {plugin}")
    for name in names:
        f, p = fork / name, plugin / name
        if not (f.exists() and p.exists()):
            failed += 1
            lines.append(f"DIFF {name}: missing in {'P' if f.exists() else 'F'}")
            continue
        f_items, p_items = items_of(name, f, info), items_of(name, p, info)
        divergence = first_divergence(f_items, p_items)
        if divergence is None:
            lines.append(f"OK   {name} ({len(f_items)} items)")
        else:
            failed += 1
            lines.append(f"DIFF {name}: {divergence}")
        if name.startswith("observer/"):
            for arm, path in (("F", f), ("P", p)):
                message = observer_consistency(path)
                if message is not None:
                    failed += 1
                    lines.append(f"DIFF {name} restore vs capture in {arm}: {message}")
    lines += [f"INFO {line}" for line in info]
    lines.append(
        f"FAIL: {failed} difference(s)" if failed else f"PASS: {len(names)} items equal"
    )
    return lines, failed


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    lines, failed = compare(Path(args[0]), Path(args[1]))
    print("\n".join(lines))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
