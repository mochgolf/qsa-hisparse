#!/usr/bin/env python3
"""Compare per-test pytest outcomes of two arms (G2-1 kernel tests).

    pytest_outcomes.py <fork log> <plugin log> [--known TEST ...]

Reads ``-rA`` short-summary lines (``PASSED``/``FAILED``/``XFAIL``/``XPASS``/
``ERROR`` with a node id) and ``SKIPPED [n] path:line: reason`` lines. Tests
are keyed by ``<file name>::<test name and parameters>`` so that ported files
in other directories line up. ``XFAIL`` counts as a failure; every failure
on either arm must be listed with ``--known`` (inherited fork failures), and
the two arms must agree on every key. Exits 1 on any disagreement.
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

OUTCOME = re.compile(r"^(PASSED|FAILED|XFAIL|XPASS|ERROR) (\S+)")
SKIPPED = re.compile(r"^SKIPPED \[(\d+)\] (\S+?):\d+: (.*)$")


def outcomes(path):
    tests, skips = {}, Counter()
    for line in Path(path).read_text(errors="replace").splitlines():
        if match := OUTCOME.match(line):
            status, node = match.groups()
            file, _, name = node.partition("::")
            key = f"{Path(file).name}::{name}"
            tests[key] = "failed" if status in ("FAILED", "XFAIL", "ERROR") else "passed"
        elif match := SKIPPED.match(line):
            count, file, reason = match.groups()
            skips[(Path(file).name, reason)] += int(count)
    return tests, skips


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("fork")
    parser.add_argument("plugin")
    parser.add_argument("--known", nargs="*", default=[], help="file::test of declared known failures")
    args = parser.parse_args(argv)
    fork, fork_skips = outcomes(args.fork)
    plugin, plugin_skips = outcomes(args.plugin)
    problems = []
    for key in sorted(set(fork) | set(plugin)):
        f, p = fork.get(key, "missing"), plugin.get(key, "missing")
        if f != p:
            problems.append(f"{key}: F {f}, P {p}")
        elif f == "failed" and key not in args.known:
            problems.append(f"{key}: fails on both arms but is not a declared known failure")
    if fork_skips != plugin_skips:
        problems.append(f"skips differ: F {dict(fork_skips)}, P {dict(plugin_skips)}")
    failed = sorted(k for k, v in fork.items() if v == "failed")
    print(f"{len(fork)} tests per arm; {sum(fork_skips.values())} skipped; known failures on both arms: {failed}")
    for problem in problems:
        print("DIFF", problem)
    print("PASS" if not problems else f"FAIL: {len(problems)} problem(s)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
