#!/usr/bin/env python3
"""Check emitted host-cache budgets, static device pools and TP agreement."""

import argparse
import json
from pathlib import Path


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


def inspect(path):
    contents = path.read_text()
    assert contents.endswith("\n"), f"Incomplete final event record: {path}"
    rows = [json.loads(line) for line in contents.splitlines()]
    assert rows, f"Empty event log: {path}"
    prefix = [row for row in rows if row["event"].startswith("prefix_")]
    assert prefix, f"No prefix events: {path}"
    raw_pointers = {tuple(row["raw_ptrs"]) for row in rows}
    index_pointers = {row["index_ptr"] for row in rows}
    assert len(raw_pointers) == len(index_pointers) == 1, "Device pool changed"
    fixed_pools = {key: rows[0][key] for key in FIXED_POOL_FIELDS}
    for row in rows:
        for key, value in fixed_pools.items():
            assert row[key] == value, f"Fixed pool field changed: {key} in {path}"
        assert row["raw_storage_unchanged"] and row["index_storage_unchanged"]
        assert 0 <= row["lease_active"] <= row["lease_capacity"]
        assert row["pending_release_count"] >= 0
    for row in prefix:
        assert (
            row["prefix_cache_host_bytes"] + row["prefix_cache_pending_bytes"]
            <= row["prefix_cache_budget_bytes"]
        ), "Host cache exceeded its reserved byte budget"
    released = [
        row
        for row in rows
        if row["event"] == "logical_release_complete" and row["lease_active"] == 0
    ]
    assert released and released[-1]["pending_release_count"] == 0
    final = rows[-1]
    assert final["lease_active"] == 0, f"Final event retains live leases: {path}"
    assert final["pending_release_count"] == 0, f"Final event retains releases: {path}"
    assert final["logical_available"] == final["logical_capacity"], (
        f"Final event retains logical pages: {path}"
    )
    sequence = [
        {
            key: row[key]
            for key in (
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
        }
        | {"tokens": row.get("checkpoint_tokens", row.get("reused_tokens"))}
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
        "final_event": final["event"],
        "final_logical_available": final["logical_available"],
        "fixed_pools": fixed_pools,
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
        # Request rows, generations, lease slots and forward IDs describe the
        # shared TP schedule. Device addresses belong to each rank's allocator;
        # their stability is checked within inspect(), never across ranks.
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
