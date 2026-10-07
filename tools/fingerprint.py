#!/usr/bin/env python3
"""Generate or check pinned fingerprints for patch targets.

    tools/fingerprint.py write <name> <target>...   # fingerprints/<name>.json
    tools/fingerprint.py refresh                    # add current-mode chains
    tools/fingerprint.py check                      # every pinned record

Fingerprints are always taken from the pinned checkout (``--source-root``,
default ``$QSA_PIN_ROOT/python``), never from the reference fork. Records
import their targets, so run this with the validation interpreter. Binding
chains are recorded for the current mode: CPU when CUDA is hidden
(``CUDA_VISIBLE_DEVICES=99``), CUDA otherwise; run ``refresh`` in each mode
the plugin will be activated in.
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = os.environ.get(
    "QSA_PIN_ROOT",
    str(REPO.parent / ".worktrees" / "sglang-pin-76e06febab"),
)
OUT = REPO / "src" / "sglang_qsa_hisparse" / "fingerprints"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=str(Path(DEFAULT_ROOT) / "python"))
    sub = parser.add_subparsers(dest="command", required=True)
    write = sub.add_parser("write")
    write.add_argument("name")
    write.add_argument("targets", nargs="+")
    sub.add_parser("refresh")
    sub.add_parser("check")
    args = parser.parse_args()

    source_root = str(Path(args.source_root).resolve())
    sys.path[:0] = [source_root, str(REPO / "src")]
    from sglang_qsa_hisparse import fingerprint

    if args.command in ("write", "refresh"):
        files = (
            [OUT / f"{args.name}.json"]
            if args.command == "write"
            else sorted(OUT.glob("*.json"))
        )
        for path in files:
            records = json.loads(path.read_text()) if path.exists() else {}
            targets = args.targets if args.command == "write" else list(records)
            for target in targets:
                records[target] = fingerprint.record(
                    source_root, target, previous=records.get(target)
                )
            path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
            print(
                f"{path.relative_to(REPO)}: {len(targets)} target(s), "
                f"mode {fingerprint.binding_mode()}"
            )
        return 0

    pinned = fingerprint.load_pinned()
    fingerprint.verify(sorted(pinned), source_root=source_root, pinned=pinned)
    print(
        f"{len(pinned)} pinned fingerprint(s) match {source_root} "
        f"(mode {fingerprint.binding_mode()})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
