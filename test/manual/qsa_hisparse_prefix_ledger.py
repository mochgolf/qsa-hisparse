#!/usr/bin/env python3
"""Check emitted host-cache budgets, static device pools and TP agreement."""

import argparse
import json
from pathlib import Path


def inspect(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows, f"Empty event log: {path}"
    prefix = [row for row in rows if row["event"].startswith("prefix_")]
    assert prefix, f"No prefix events: {path}"
    raw_pointers = {tuple(row["raw_ptrs"]) for row in rows}
    index_pointers = {row["index_ptr"] for row in rows}
    assert len(raw_pointers) == len(index_pointers) == 1, "Device pool changed"
    for row in rows:
        assert row["raw_storage_unchanged"] and row["index_storage_unchanged"]
        assert 0 <= row["lease_active"] <= row["lease_capacity"]
        assert row["pending_release_count"] >= 0
    for row in prefix:
        assert (
            row["prefix_cache_host_bytes"] + row["prefix_cache_pending_bytes"]
            <= row["prefix_cache_budget_bytes"]
        ), "Host cache exceeded its reserved byte budget"
    released = [
        row for row in rows
        if row["event"] == "logical_release_complete" and row["lease_active"] == 0
    ]
    assert released and released[-1]["pending_release_count"] == 0
    sequence = [
        {key: row[key] for key in (
            "event", "prefix_cache_host_bytes", "prefix_cache_entries",
            "prefix_cache_evictions", "prefix_cache_epoch",
        )} | {"tokens": row.get("checkpoint_tokens", row.get("reused_tokens"))}
        for row in prefix
    ]
    return {
        "file": str(path),
        "events": len(rows),
        "restores": sum(row["event"] == "prefix_restore_complete" for row in rows),
        "max_restored_tokens": max(row.get("reused_tokens", 0) for row in prefix),
        "host_budget_bytes": prefix[0]["prefix_cache_budget_bytes"],
        "max_host_bytes": max(row["prefix_cache_host_bytes"] for row in prefix),
        "max_entries": max(row["prefix_cache_entries"] for row in prefix),
        "evictions": max(row["prefix_cache_evictions"] for row in prefix),
        "max_active_leases": max(row["lease_active"] for row in rows),
        "raw_bytes": rows[0]["raw_bytes"],
        "index_bytes": rows[0]["index_bytes"],
        "idle_cuda_allocated_min": min(row["cuda_allocated"] for row in released),
        "idle_cuda_allocated_max": max(row["cuda_allocated"] for row in released),
        "idle_cuda_reserved_max": max(row["cuda_reserved"] for row in released),
        "last_idle_logical_available": released[-1]["logical_available"],
        "logical_capacity": released[-1]["logical_capacity"],
        "static_pools_unchanged": True,
    }, sequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"passed": False, "ranks": []}
    try:
        sequences = []
        for path in sorted(args.events.glob("rank-*.jsonl")):
            rank, sequence = inspect(path)
            report["ranks"].append(rank)
            sequences.append(sequence)
        assert len(sequences) == 2, "Expected TP2 event logs"
        assert sequences[0] == sequences[1], "TP prefix publication/restore differs"
        for rank in report["ranks"]:
            assert rank["last_idle_logical_available"] == rank["logical_capacity"]
        report["tp_prefix_sequence_equal"] = True
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
