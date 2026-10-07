#!/usr/bin/env python3
"""Generate the activation manifest from docs/patch-inventory.md.

The manifest lists, per inventory row, the patches and attachments a feature
must declare. Activation requires the declared set for each requested
feature to equal the manifest exactly, so a missing patch module, a missing
row, or an extra undeclared hook fails closed. Rows that need several hook
types or name new members are completed by OVERRIDES below.

    tools/manifest.py          # write src/sglang_qsa_hisparse/manifest.json
    tools/manifest.py --check  # fail if the file is stale
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INVENTORY = REPO / "docs" / "patch-inventory.md"
OUT = REPO / "src" / "sglang_qsa_hisparse" / "manifest.json"
HOOK_TYPES = ("before", "after", "around", "replace")
BACKEND = "sglang.srt.layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend"

OVERRIDES = {
    "K03": {
        "patches": [
            {"target": "sglang.srt.mem_cache.qsa_kv_pool.QSATokenToKVPool.__init__", "hook_type": "around"},
            {"target": "sglang.srt.mem_cache.memory_pool.HybridLinearKVPool.__init__", "hook_type": "before"},
        ]
    },
    "F01": {
        "patches": [
            {"target": "sglang.srt.model_executor.forward_batch_info.ForwardBatch.init_new", "hook_type": "after"},
            {"target": "sglang.srt.model_executor.runner.eager_runner.EagerRunner.load_batch", "hook_type": "after"},
        ]
    },
    "Q04": {"attach": [{"owner": BACKEND, "name": "_kv_descales"}, {"owner": BACKEND, "name": "_store_kv"}]},
    "Q09": {
        "attach": [
            {"owner": BACKEND, "name": "_qsa_local_head_shape"},
            {"owner": BACKEND, "name": "_ensure_fa2_graph_wrapper"},
            {"owner": BACKEND, "name": "_can_run_fa2_graph"},
        ]
    },
}

FRAMEWORK_ROWS = {
    "FW1": {
        "feature": "framework",
        "patches": [
            {
                "target": "sglang.srt.managers.scheduler.configure_scheduler_process",
                "hook_type": "before",
            }
        ],
        "attach": [],
    }
}


def build() -> dict:
    rows: dict = {}
    for line in INVENTORY.read_text().splitlines():
        match = re.match(r"^\| ([A-Z]+\d+[a-z]?) \|", line)
        if not match:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 9 or cells[0] in rows or cells[0] in FRAMEWORK_ROWS:
            continue  # Appendix tables repeat IDs with fewer columns.
        row, feature_cell, target_cell, hook_cell = cells[0], cells[3], cells[4], cells[5]
        feature = feature_cell.split(":")[0].split(" ")[0].strip("*")
        if feature not in ("model_compat", "hisparse"):
            continue
        entry = {"feature": feature, "patches": [], "attach": []}
        if row in OVERRIDES:
            entry.update(OVERRIDES[row])
        else:
            kind = hook_cell.split(":")[0].split(" ")[0].strip(".,(")
            targets = re.findall(r"`(sglang\.[A-Za-z0-9_.]+)`", target_cell)
            if kind in HOOK_TYPES:
                if len(targets) != 1:
                    raise SystemExit(f"{row}: expected one target, found {targets}")
                entry["patches"] = [{"target": targets[0], "hook_type": kind}]
            elif kind not in ("moved/owned", "moved", "none"):
                raise SystemExit(f"{row}: unknown hook kind {kind!r}")
        if entry["patches"] or entry["attach"]:
            rows[row] = entry
    rows.update(FRAMEWORK_ROWS)
    return {"source": "docs/patch-inventory.md", "rows": dict(sorted(rows.items()))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = json.dumps(build(), indent=2) + "\n"
    if args.check:
        if not OUT.exists() or OUT.read_text() != text:
            print("manifest.json is stale; run tools/manifest.py", file=sys.stderr)
            return 1
        return 0
    OUT.write_text(text)
    rows = json.loads(text)["rows"]
    print(f"wrote {len(rows)} rows to {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
