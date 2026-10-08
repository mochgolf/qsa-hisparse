#!/usr/bin/env python3
"""Compare per-test pytest outcomes of two arms (G2-1 kernel tests).

    pytest_outcomes.py --fork LOG [LOG ...] --plugin LOG [LOG ...]
                       --inventory FILE [--supplement LOG ...]
                       [--known FILE::TEST=SIGNATURE ...]

Each log holds exactly one pytest run (one final summary line).

Reads ``-rA`` short-summary lines (``PASSED``/``FAILED``/``XFAIL``/``XPASS``/
``ERROR`` with a node id), ``SKIPPED [n] path:line: reason`` lines and the
final ``N passed, M failed ... in Xs`` line of each log. Tests are keyed by
``<file name>::<test name and parameters>`` so that ported files in other
directories line up.

Fails unless, for each arm: the final summary exists and its counts equal the
parsed lines; the outcome IDs and the skips (file and reason) equal the frozen
``--inventory`` exactly; no ID has conflicting outcomes. Across arms: every
key has the same outcome; every failure is a declared known failure whose
exception line (from the FAILURES section) equals the declared SIGNATURE;
an ``XFAIL`` is accepted only if a ``--supplement`` log (the same test run
with ``--runxfail``) shows it FAILED with that signature; ``ERROR``,
``XPASS`` and ``[XPASS(strict)]`` are never accepted.

Inventory lines: ``FILE::TEST`` for a test with an outcome, and
``skip FILE::TEST <reason>`` for a skip; logs must then come from ``-v``
runs, whose ``<node id> SKIPPED`` lines identify the skipped tests.
``skip FILE <reason>`` (no test name) is accepted only for evidence recorded
without ``-v`` (run2), where skips are reported by file and reason.

The reference fork arm is Phase 2's (pin 76e06febab). SGLang v0.5.21 moved
the registered kernel tests to ``test/registered/kernels/ops/attention/qsa/``
(``test_qsa_indexer.py``, ``test_qsa_strided_zero_fill.py``) and
``test/registered/kernels/ops/gemm/`` and renamed ``test_hc_mix_triton.py`` to
``test_hc_mix.py`` (same test names); ``RENAMED`` keys the renamed file under
its reference name.
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

OUTCOME = re.compile(r"^(PASSED|FAILED|XFAIL|XPASS|ERROR) (\S+)(?: - (.*))?$")
SKIPPED = re.compile(r"^SKIPPED \[(\d+)\] (\S+?):\d+: (.*)$")
VERBOSE_SKIP = re.compile(r"^(\S+?)::(\S+) SKIPPED")
SECTION = re.compile(r"^_{3,} (\S+) _{3,}$")
EXCEPTION = re.compile(r"^E\s+((?:\w+\.)*\w*(?:Error|Exception|Exit|Interrupt)\b.*)$")
SUMMARY = re.compile(r"^=*\s*((?:\d+ \w+(?:, )?)+) in [\d.]+s")
SUMMARY_ITEM = re.compile(r"(\d+) (passed|failed|skipped|xfailed|xpassed|errors?|deselected|warnings?|subtests passed)")
# Upstream test file renamed after the reference run -> its reference name.
RENAMED = {"test_hc_mix.py": "test_hc_mix_triton.py"}


def file_key(path):
    name = Path(path).name
    return RENAMED.get(name, name)


def parse(path, tests, skips):
    """Add one pytest run's outcomes to ``tests``/``skips``; return problems."""
    problems = []
    counted, run_skips = Counter(), Counter()
    summaries = []
    raised, section = {}, None  # test name -> last exception type in its FAILURES section
    lines = Path(path).read_text(errors="replace").splitlines()
    for line in lines:
        if match := SECTION.match(line):
            section = match.group(1)
        elif section and (match := EXCEPTION.match(line)):
            raised[section] = match.group(1)
    for line in lines:
        if match := OUTCOME.match(line):
            status, node, message = match.groups()
            file, _, name = node.partition("::")
            key = f"{file_key(file)}::{name}"
            if status == "FAILED" and not message:
                message = raised.get(name, "")
            outcome = (status, message or "")
            if key in tests and tests[key] != outcome:
                problems.append(f"{key}: conflicting outcomes {tests[key][0]} and {status}")
            tests[key] = outcome
            counted[status] += 1
        elif match := SKIPPED.match(line):
            count, file, reason = match.groups()
            skips[(file_key(file), reason)] += int(count)
            run_skips[(file_key(file), reason)] += int(count)
        elif match := VERBOSE_SKIP.match(line):
            skips[("id", f"{file_key(match.group(1))}::{match.group(2)}")] += 1
        elif match := SUMMARY.match(line.strip("= ")):
            summary = Counter()
            for number, kind in SUMMARY_ITEM.findall(match.group(1)):
                summary[kind.rstrip("s") if kind.startswith("error") else kind] += int(number)
            summaries.append(summary)
    if len(summaries) != 1:
        problems.append(f"{path}: {len(summaries)} final pytest summaries, expected exactly one")
    else:
        summary = summaries[0]
        expected = {
            "passed": counted["PASSED"], "failed": counted["FAILED"], "xfailed": counted["XFAIL"],
            "xpassed": counted["XPASS"], "error": counted["ERROR"], "skipped": sum(run_skips.values()),
        }
        for kind, number in expected.items():
            if summary[kind] != number:
                problems.append(f"{path}: summary says {summary[kind]} {kind}, parsed {number}")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fork", nargs="+", required=True)
    parser.add_argument("--plugin", nargs="+", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--supplement", nargs="*", default=[])
    parser.add_argument("--known", nargs="*", default=[], help="FILE::TEST=SIGNATURE")
    args = parser.parse_args(argv)
    known = dict(item.rsplit("=", 1) for item in args.known)
    arms, problems = {}, []
    for arm, logs in (("F", args.fork), ("P", args.plugin)):
        tests, skips = {}, Counter()
        for log in logs:
            problems += [f"{arm}: {p}" for p in parse(log, tests, skips)]
        arms[arm] = (tests, skips, None)
    ids, skip_inventory = set(), Counter()
    for line in Path(args.inventory).read_text().splitlines():
        if line.startswith("skip "):
            _, file, reason = line.split(" ", 2)
            if "::" in file:
                skip_inventory[("id", file)] += 1
                file = file.split("::", 1)[0]
            skip_inventory[(file, reason)] += 1
        elif line.strip() and not line.startswith("#"):
            ids.add(line.strip())
    for arm, (tests, skips, _) in arms.items():
        if set(tests) != ids:
            problems.append(f"{arm}: IDs differ from inventory: missing {sorted(ids - set(tests))}, "
                            f"extra {sorted(set(tests) - ids)}")
        if skips != skip_inventory:
            problems.append(f"{arm}: skips {dict(skips)} differ from inventory {dict(skip_inventory)}")
    supplement, supplement_skips = {}, Counter()
    for log in args.supplement:
        problems += [f"supplement: {p}" for p in parse(log, supplement, supplement_skips)]

    def verdict(key, outcome):
        status, message = outcome
        if status == "PASSED":
            return "passed"
        if status in ("ERROR", "XPASS") or "XPASS" in message:
            return f"unacceptable {status} {message}".strip()
        if key not in known:
            return f"undeclared {status} {message}".strip()
        if status == "XFAIL":
            status, message = supplement.get(key, ("XFAIL", ""))
            if status != "FAILED":
                return "xfail without --runxfail supplement evidence"
        if known[key] != message.strip():
            return f"failed with {message!r}, declared {known[key]!r}"
        return "known failure"

    fork, plugin = arms["F"][0], arms["P"][0]
    for key in sorted(set(fork) | set(plugin)):
        f = verdict(key, fork[key]) if key in fork else "missing"
        p = verdict(key, plugin[key]) if key in plugin else "missing"
        if f != p or f not in ("passed", "known failure"):
            problems.append(f"{key}: F {f}; P {p}")
    if arms["F"][1] != arms["P"][1]:
        problems.append(f"skips differ: F {dict(arms['F'][1])}, P {dict(arms['P'][1])}")
    print(f"{len(fork)} outcomes + {sum(arms['F'][1].values())} skipped per arm (F); "
          f"declared known failures: {sorted(known)}")
    for problem in problems:
        print("DIFF", problem)
    print("PASS" if not problems else f"FAIL: {len(problems)} problem(s)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
