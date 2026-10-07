"""F-vs-P comparator on synthetic arm directories.

The plugin arm differs from the fork arm only in what legitimately varies
between runs (times, pointers, time-based rids, measured memory, log order,
the deliberately aborted request) unless a test injects a real difference.
"""

import hashlib
import json
import shutil

import pytest
import torch

RAW = torch.arange(2 * 2 * 64 * 256).remainder(251).to(torch.uint8).reshape(2, 2, 64, 1, 256)
INDEX = torch.arange(2 * 16 * 16).reshape(2, 16, 1, 16).to(torch.bfloat16)


def generation(ids, cached, rid, latency):
    return {
        "output_ids": ids,
        "text": "READY",
        "prompt_tokens": 76,
        "cached_tokens": cached,
        "wall_seconds": latency,
        "meta_info": {
            "id": rid,
            "prompt_tokens": 76,
            "cached_tokens": cached,
            "finish_reason": {"type": "length", "length": len(ids)},
            "output_token_logprobs": [[-0.015103152021765709, ids[0], None]],
            "e2e_latency": latency,
            "request_received_ts": 1789570621.9 + latency,
        },
    }


def report(tag, latency):
    return {
        "phase": "qualify",
        "started_at": 1789570000 + latency,
        "base_url": "http://127.0.0.1:8082",
        "cases": [
            {
                "name": "prefix-64-copy",
                "cold_0": generation([271, 248068, 198], 0, f"cold-{tag}", latency),
                "warm": generation([271, 248068, 198], 64, f"warm-{tag}", latency),
                "passed": True,
            }
        ],
        "aborted_response": {"output_ids": list(range(int(latency * 10)))},
        "output_mismatches": [],
        "passed": True,
    }


def ledger_rows(tag, pointer):
    fixed = dict(
        staging_capacity_tokens=262144,
        raw_pool_size_tokens=262184,
        raw_backing_size_tokens=262248,
        ring_reserved_tokens=40,
        lease_capacity=8,
        logical_capacity=2097152,
        raw_bytes=1611251712,
        index_bytes=1610661888,
        host_reserved_bytes=12884901888,
        hot_reserved_bytes=415236096,
        workspace_bytes=1000,
        mamba_bytes=2000,
    )

    def row(event, **extra):
        return {
            "event": event,
            "time_ns": pointer * 7,
            "rank": 0,
            "forward_id": extra.pop("forward_id", 0),
            "raw_ptrs": [pointer, pointer + 1],
            "index_ptr": pointer + 2,
            "cuda_allocated": pointer + 3,
            **fixed,
            **extra,
        }

    rows = [row("init", host_alloc_wall_ms=pointer / 1000)]
    for bs in (1, 2):
        rows.append(
            row(
                "graph_capture_complete",
                graph_key={"bs": bs},
                graph_pool=[0, pointer],
                graph_buffers={"real": {"ptr": pointer + bs, "bytes": 4, "shape": [1], "dtype": "torch.int32"}},
            )
        )
    for i, (event, tokens) in enumerate(
        [("prefix_publish_complete", 64), ("prefix_publish_complete", 128), ("prefix_restore_complete", 128)]
    ):
        key = "checkpoint_tokens" if event == "prefix_publish_complete" else "reused_tokens"
        rid = f"seed-{tag}" if i < 2 else f"warm-{tag}"
        rows.append(
            row(
                event,
                forward_id=3 + i,
                rid=rid,
                generation=1 + (i == 2),
                req_pool_idx=1,
                lease_slot=0,
                prefix_cache_host_bytes=1000 * (i + 1),
                prefix_cache_entries=1 + (i > 0),
                prefix_cache_evictions=0,
                prefix_cache_epoch=0,
                leases=[{"rid": rid, "host_ptr": pointer + 9}],
                **{key: tokens},
            )
        )
    return rows


def observer_rows(digest, tag, raw=RAW, restored_raw=None):
    state = {
        "tokens": 64,
        "token_sha256": "0" * 64,
        "pending": [digest(torch.full((4, 1, 16), 200, dtype=torch.bfloat16))],
        "rope": digest(torch.arange(12).reshape(4, 3)),
        "mamba": [[digest(torch.ones(3, 1, 2, 3))], digest(torch.zeros(3, 1, 2, 2))],
    }
    capture = {"segments": [[0, 64, digest(raw), digest(INDEX)]], **state}
    restored = restored_raw if restored_raw is not None else raw
    restore = {"segments": [[0, 64, digest(restored), digest(INDEX)]], **state}
    return [
        {"event": "capture", "rank": 0, "checkpoint": 0, "rid": f"seed-{tag}", **capture},
        {"event": "restore", "rank": 0, "checkpoint": 0, "rid": f"warm-{tag}", **restore},
    ]


def ledger_report(tag, latency):
    """Fork qsa_hisparse_prefix_ledger.py output; path, counts and CUDA figures vary."""
    rank = {
        "file": f"/runs/{tag}/events/rank-0.jsonl",
        "events": 40 + int(latency),
        "restores": 1,
        "max_restored_tokens": 128,
        "host_budget_bytes": 8589934592,
        "max_host_bytes": 3000,
        "max_entries": 2,
        "evictions": 0,
        "max_active_leases": 1 + int(latency),
        "raw_bytes": 1611251712,
        "index_bytes": 1610661888,
        "idle_cuda_allocated_min": int(latency * 1e6),
        "idle_cuda_reserved_max": int(latency * 2e6),
        "last_idle_logical_available": 2097152,
        "logical_capacity": 2097152,
        "final_event": "logical_release_complete",
        "final_logical_available": 2097152,
        "fixed_pools": {"lease_capacity": 8, "mamba_bytes": 2000},
        "static_pools_unchanged": True,
    }
    return {"passed": True, "ranks": [rank, dict(rank)], "tp_prefix_sequence_equal": True}


def marlin_report(tag):
    """Fork marlin_batch_invariance.py output; source path and helper hash vary."""

    def stages(ordinal):
        return {
            name: {
                "sha256": hashlib.sha256(f"{ordinal}-{name}".encode()).hexdigest(),
                "bitwise_equal": True,
                "distinct_repeated_outputs": 1,
            }
            for name in ("gate_up", "pipeline_sum")
        }

    return {
        "source": f"/src/{tag}/stable_align.py",
        "helper_sha256": hashlib.sha256(tag.encode()).hexdigest(),
        "results": [
            {"ordinal": i, "m": m, "row": 0, "pattern": "identical", "stages": stages(i)}
            for i, m in enumerate((1, 8))
        ],
        "cases": 2,
        "differing_cases": 0,
        "fixed_case_repeatable": True,
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def write_arm(root, digest, *, tag, pointer, latency, log_order=(0, 1), **observer):
    root.mkdir()
    (root / "qualification.json").write_text(json.dumps(report(tag, latency)))
    (root / "ledger.json").write_text(json.dumps(ledger_report(tag, latency)))
    (root / "marlin-whole-k.json").write_text(json.dumps(marlin_report(tag)))
    (root / "run.json").write_text(json.dumps({"server": [tag]}))
    (root / "server_info.json").write_text(
        json.dumps(
            {
                "max_total_num_tokens": 2097152,
                "max_req_input_len": 262138,
                "port": 8082,
                "internal_states": [
                    {
                        "memory_usage": {
                            "weight": 36.871 + latency / 1000,
                            "kvcache": 3.001,
                            "token_capacity": 2097152,
                            "startup_available": 3.11 + latency / 1000,
                        }
                    }
                ],
            }
        )
    )
    lines = []
    for rank in log_order:
        lines += [
            f"[2026-10-07 10:00:{latency:02.0f} TP{rank}] QSA P2 offload pool budget: "
            "fixed_bytes=1611399936, logical_bytes_per_token=768",
            f"[2026-10-07 10:00:{latency:02.0f} TP{rank}] max_total_num_tokens=2097152, "
            "chunked_prefill_size=2048",
        ]
    (root / "server.log").write_text("\n".join(lines) + "\n")
    for rank in (0, 1):
        write_jsonl(root / "events" / f"rank-{rank}.jsonl", ledger_rows(tag, pointer))
        rows = observer_rows(digest, tag, **observer)
        write_jsonl(root / "observer" / f"rank-{rank}.jsonl", rows)


@pytest.fixture
def arms(evidence, tmp_path):
    digest = evidence("observer").digest
    fork, plugin = tmp_path / "F", tmp_path / "P"
    write_arm(fork, digest, tag="actual-batch-1791000000000000000", pointer=1000, latency=1.5)
    write_arm(
        plugin,
        digest,
        tag="actual-batch-1791000099999999999",
        pointer=5000,
        latency=2.7,
        log_order=(1, 0),
    )
    return evidence("compare"), fork, plugin, digest


def run(compare, fork, plugin, capsys, *options):
    status = compare.main([str(fork), str(plugin), *options])
    return status, capsys.readouterr().out


def test_equal_behavior_passes(arms, capsys):
    compare, fork, plugin, _ = arms
    status, out = run(compare, fork, plugin, capsys, "--require-observer", "2")
    assert status == 0, out
    assert "PASS: 9 items equal" in out
    assert "OK   marlin-whole-k.json (6 items)" in out
    assert "OK   ledger.json (30 items)" in out
    assert "INFO P internal_states[0] measured memory" in out


def test_single_flipped_cached_byte_fails(arms, capsys):
    compare, fork, plugin, digest = arms
    flipped = RAW.clone()
    flipped[1, 1, 63, 0, 255] ^= 1
    write_jsonl(plugin / "observer" / "rank-0.jsonl", observer_rows(digest, "x", raw=flipped))
    status, out = run(compare, fork, plugin, capsys)
    assert status == 1
    assert "DIFF observer/rank-0.jsonl: item 0:" in out
    assert "restore vs capture" not in out  # P is self-consistent; only F vs P differs.


def test_reordered_ledger_fails(arms, capsys):
    compare, fork, plugin, _ = arms
    path = plugin / "events" / "rank-1.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-2], rows[-1] = rows[-1], rows[-2]
    write_jsonl(path, rows)
    status, out = run(compare, fork, plugin, capsys)
    assert status == 1
    assert "DIFF events/rank-1.jsonl: item 4: F [\"prefix event 1\"" in out
    assert "DIFF events/rank-0.jsonl" not in out


def edit_json(path, change):
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


@pytest.mark.parametrize(
    "mutate, item",
    [
        (
            lambda p: edit_json(
                p / "qualification.json",
                lambda d: d["cases"][0]["warm"]["output_ids"].__setitem__(2, 199),
            ),
            "F [\".cases[0].warm.output_ids\"",
        ),
        (
            lambda p: edit_json(
                p / "qualification.json",
                lambda d: d["cases"][0]["warm"]["meta_info"].__setitem__("cached_tokens", 0),
            ),
            "F [\".cases[0].warm.meta_info.cached_tokens\"",
        ),
        (
            lambda p: (p / "server.log").write_text(
                (p / "server.log").read_text().replace("fixed_bytes=1611399936", "fixed_bytes=1611399937", 1)
            ),
            "DIFF server.log",
        ),
        (
            lambda p: edit_json(
                p / "server_info.json",
                lambda d: d["internal_states"][0]["memory_usage"].__setitem__("kvcache", 3.002),
            ),
            "DIFF server_info.json",
        ),
        (
            lambda p: write_jsonl(
                p / "events" / "rank-0.jsonl",
                [{**r, "raw_bytes": r["raw_bytes"] + 1} for r in ledger_rows("y", 7)],
            ),
            "DIFF events/rank-0.jsonl: item 0: F [\"fixed_pools\"",
        ),
        (
            lambda p: edit_json(p / "ledger.json", lambda d: d["ranks"][1].__setitem__("restores", 2)),
            "F [\".ranks[1].restores\", \"1\"] != P [\".ranks[1].restores\", \"2\"]",
        ),
        (
            lambda p: edit_json(p / "ledger.json", lambda d: d["ranks"][0].__setitem__("max_entries", 3)),
            "F [\".ranks[0].max_entries\", \"2\"] != P [\".ranks[0].max_entries\", \"3\"]",
        ),
        (
            lambda p: edit_json(
                p / "marlin-whole-k.json",
                lambda d: d["results"][1]["stages"]["pipeline_sum"].__setitem__("sha256", "0" * 64),
            ),
            "F [\".results[1].stages.pipeline_sum.sha256\"",
        ),
        (lambda p: (p / "events" / "rank-1.jsonl").unlink(), "DIFF events/rank-1.jsonl: missing in P"),
        (lambda p: shutil.rmtree(p / "observer"), "DIFF observer/rank-0.jsonl: missing in P"),
    ],
)
def test_real_differences_fail(arms, capsys, mutate, item):
    compare, fork, plugin, _ = arms
    mutate(plugin)
    status, out = run(compare, fork, plugin, capsys)
    assert status == 1
    assert item in out, out


def test_restore_that_differs_from_its_capture_fails_in_each_arm(arms, capsys):
    compare, fork, plugin, digest = arms
    corrupted = RAW.clone()
    corrupted[0, 0, 0, 0, 0] ^= 0x80
    for arm, tag in ((fork, "a"), (plugin, "b")):
        rows = observer_rows(digest, tag, restored_raw=corrupted)
        write_jsonl(arm / "observer" / "rank-0.jsonl", rows)
    status, out = run(compare, fork, plugin, capsys)
    assert status == 1
    assert "OK   observer/rank-0.jsonl" in out
    for arm in "FP":
        assert (
            f"restore vs capture in {arm}: record 1: restore of checkpoint 0 "
            "differs from its capture in segments" in out
        )


def test_report_without_compared_values_fails(arms, capsys):
    compare, fork, plugin, _ = arms
    for arm in (fork, plugin):
        (arm / "probe.json").write_text(json.dumps({"device": "cuda:0", "results": []}))
    status, out = run(compare, fork, plugin, capsys)
    assert status == 1
    assert "DIFF probe.json: no compared value in either arm" in out


def keep_captures(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    write_jsonl(path, [r for r in rows if r["event"] == "capture"])


@pytest.mark.parametrize(
    "mutate, problem",
    [
        (lambda o: (o / "rank-1.jsonl").unlink(), "rank-1.jsonl"),
        (lambda o: (o / "rank-0.jsonl").write_text(""), "rank-0.jsonl"),
        (lambda o: keep_captures(o / "rank-1.jsonl"), "rank-1.jsonl"),
    ],
)
def test_required_observer_evidence_must_exist_in_both_arms(arms, capsys, mutate, problem):
    """Missing byte evidence fails even when it is missing identically in both arms."""
    compare, fork, plugin, _ = arms
    for arm in (fork, plugin):
        mutate(arm / "observer")
    status, out = run(compare, fork, plugin, capsys, "--require-observer", "2")
    assert status == 1
    for arm in "FP":
        expected = f"DIFF required evidence in {arm}: observer/{problem} lacks a capture or a restore"
        assert expected in out, out


def test_observer_files_are_optional_without_the_requirement(arms, capsys):
    compare, fork, plugin, _ = arms
    for arm in (fork, plugin):
        shutil.rmtree(arm / "observer")
    status, out = run(compare, fork, plugin, capsys)
    assert status == 0, out  # The compat-only arm has no HiSparse runtime to observe.
    status, out = run(compare, fork, plugin, capsys, "--require-observer", "2")
    assert status == 1
    assert out.count("lacks a capture or a restore") == 4
