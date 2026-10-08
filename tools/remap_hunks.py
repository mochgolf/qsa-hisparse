#!/usr/bin/env python3
"""Carry Appendix A's hunk -> row map to a new fork reference.

    tools/remap_hunks.py --repo <git repo> OLD_BASE OLD_FORK NEW_BASE NEW_FORK

Reads both fork diffs (``git diff -U0 base fork -- python/sglang``) and the
current Appendix A (which maps OLD_BASE..OLD_FORK). A hunk of the new diff
takes the row of the old hunk with the same removed and added lines in the
same file (renamed files via RENAMES); a new file takes its old row. Prints
the new Appendix A rows and, on stderr, every new hunk without a match: those
are changes the old reference did not have, to be mapped by hand.
"""

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INVENTORY = REPO / "docs" / "patch-inventory.md"
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
# Files upstream moved between the references (old path -> new path).
RENAMES = {"srt/layers/hc_mix_triton.py": "kernels/ops/gemm/hc_mix.py"}


def hunks(repo, base, fork):
    """[(file, key, removed, added)] with key '+start,count' or 'new'."""
    diff = subprocess.run(
        ["git", "-C", repo, "diff", "-U0", "--no-color", "--no-ext-diff", "--no-renames",
         "--diff-algorithm=myers", base, fork, "--", "python/sglang"],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    out, path, current, new_file = [], None, None, False
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            path, current, new_file = line.split(" b/python/sglang/", 1)[1], None, False
        elif line.startswith("new file mode"):
            new_file = True
            out.append((path, "new", (), ()))
        elif new_file:
            continue
        elif match := HUNK.match(line):
            current = [path, f"+{match[1]},{match[2] or 1}", [], []]
            out.append(current)
        elif current is not None and line[:1] in "-+" and not line.startswith(("---", "+++")):
            current[2 if line[0] == "-" else 3].append(line[1:])
    return [(f, k, tuple(r), tuple(a)) for f, k, r, a in out]


def appendix_a():
    text = INVENTORY.read_text()
    start = text.index("## Appendix A.")
    end = text.find("\n## ", start + 1)
    rows = {}
    for line in text[start : end if end != -1 else None].splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 3 and cells[1].startswith(("+", "(new file)")):
            key = "new" if cells[1] == "(new file)" else cells[1].split(" ", 1)[0]
            rows[(cells[0], key)] = cells[2]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", required=True)
    parser.add_argument("refs", nargs=4, metavar=("OLD_BASE", "OLD_FORK", "NEW_BASE", "NEW_FORK"))
    args = parser.parse_args()
    old_base, old_fork, new_base, new_fork = args.refs
    old_rows = appendix_a()
    by_content = defaultdict(list)
    for file, key, removed, added in hunks(args.repo, old_base, old_fork):
        file = RENAMES.get(file, file)
        by_content[(file, removed, added)].append(old_rows[(RENAMES_INV.get(file, file), key)])
    mapped, unmatched = [], []
    for file, key, removed, added in hunks(args.repo, new_base, new_fork):
        candidates = by_content.get((file, removed, added))
        if candidates:
            mapped.append((file, key, candidates.pop(0)))
        else:
            unmatched.append((file, key, removed, added))
    for file, key, row in mapped:
        print(f"| {file} | {'(new file)' if key == 'new' else key} | {row} |")
    for file, key, removed, added in unmatched:
        first = (added or removed or ("",))[0].strip()[:90]
        print(f"UNMATCHED {file} {key} -{len(removed)} +{len(added)}: {first}", file=sys.stderr)
    print(f"{len(mapped)} mapped, {len(unmatched)} unmatched", file=sys.stderr)


RENAMES_INV = {new: old for old, new in RENAMES.items()}

if __name__ == "__main__":
    main()
