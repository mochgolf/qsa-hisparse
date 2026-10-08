#!/usr/bin/env python3
"""Generate or check pinned fingerprints for patch targets.

    tools/fingerprint.py write <name> <target>...   # fingerprints/<name>.json
    tools/fingerprint.py check                      # every pinned record

Fingerprints are always taken from the pinned checkout (``--source-root``,
default ``$QSA_PIN_ROOT/python``), never from the reference fork. Records are
computed from source files only; any Python 3.10+ interpreter works.
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = os.environ.get(
    "QSA_PIN_ROOT",
    str(REPO.parent / ".worktrees" / "sglang-main-35f3c96ff4"),
)
OUT = REPO / "src" / "sglang_qsa_hisparse" / "fingerprints"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=str(Path(DEFAULT_ROOT) / "python"))
    sub = parser.add_subparsers(dest="command", required=True)
    write = sub.add_parser("write")
    write.add_argument("name")
    write.add_argument("targets", nargs="+")
    sub.add_parser("check")
    args = parser.parse_args()

    source_root = str(Path(args.source_root).resolve())
    sys.path.insert(0, str(REPO / "src"))
    from sglang_qsa_hisparse import fingerprint

    if args.command == "write":
        path = OUT / f"{args.name}.json"
        records = json.loads(path.read_text()) if path.exists() else {}
        for target in args.targets:
            records[target] = fingerprint.record(source_root, target)
        path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
        print(f"{path.relative_to(REPO)}: {len(args.targets)} target(s)")
        return 0

    pinned = fingerprint.load_pinned()
    fingerprint.verify(sorted(pinned), source_root=source_root, pinned=pinned)
    print(f"{len(pinned)} pinned fingerprint(s) match {source_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
