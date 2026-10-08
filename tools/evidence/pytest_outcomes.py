#!/usr/bin/env python3
"""Compare per-test pytest outcomes of two arms (G2-1 kernel tests).

    pytest_outcomes.py --fork LOG [LOG ...] --plugin LOG [LOG ...] --expect-tests N
                       [--known FILE::TEST=ExceptionType ...]

Each log holds exactly one pytest run (one final summary line).

Reads ``-rA`` short-summary lines (``PASSED``/``FAILED``/``XFAIL``/``XPASS``/
``ERROR`` with a node id), ``SKIPPED [n] path:line: reason`` lines and the
final ``N passed, M failed ... in Xs`` line of each log. Tests are keyed by
``<file name>::<test name and parameters>`` so that ported files in other
directories line up.

Fails unless, for each arm: the final summary exists and its counts equal the
parsed lines; the number of tests (outcomes plus skips) equals
``--expect-tests``; no ID has conflicting outcomes. Across arms: every key
has the same outcome; every failure is a declared known failure whose
``FAILED`` message starts with the declared exception type (``XFAIL`` is
accepted for it: the port's strict xfail pins ``raises=``); ``ERROR``,
``XPASS`` and ``[XPASS(strict)]`` are never accepted.
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

OUTCOME = re.compile(r"^(PASSED|FAILED|XFAIL|XPASS|ERROR) (\S+)(?: - (.*))?$")
SKIPPED = re.compile(r"^SKIPPED \[(\d+)\] (\S+?):\d+: (.*)$")
SECTION = re.compile(r"^_{3,} (\S+) _{3,}$")
EXCEPTION = re.compile(r"^E\s+((?:\w+\.)*\w*(?:Error|Exception|Exit|Interrupt))\b")
SUMMARY = re.compile(r"^=*\s*((?:\d+ \w+(?:, )?)+) in [\d.]+s")
SUMMARY_ITEM = re.compile(r"(\d+) (passed|failed|skipped|xfailed|xpassed|errors?|deselected|warnings?|subtests passed)")


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
            key = f"{Path(file).name}::{name}"
            if status == "FAILED" and not message:
                message = raised.get(name, "")
            outcome = (status, message or "")
            if key in tests and tests[key] != outcome:
                problems.append(f"{key}: conflicting outcomes {tests[key][0]} and {status}")
            tests[key] = outcome
            counted[status] += 1
        elif match := SKIPPED.match(line):
            count, file, reason = match.groups()
            skips[(Path(file).name, reason)] += int(count)
            run_skips[(Path(file).name, reason)] += int(count)
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
    parser.add_argument("--expect-tests", type=int, required=True)
    parser.add_argument("--known", nargs="*", default=[], help="FILE::TEST=ExceptionType")
    args = parser.parse_args(argv)
    known = dict(item.rsplit("=", 1) for item in args.known)
    arms, problems = {}, []
    for arm, logs in (("F", args.fork), ("P", args.plugin)):
        tests, skips = {}, Counter()
        for log in logs:
            problems += [f"{arm}: {p}" for p in parse(log, tests, skips)]
        arms[arm] = (tests, skips, None)
    for arm, (tests, skips, _) in arms.items():
        total = len(tests) + sum(skips.values())
        if total != args.expect_tests:
            problems.append(f"{arm}: {total} tests, expected {args.expect_tests}")

    def verdict(key, outcome):
        status, message = outcome
        if status == "PASSED":
            return "passed"
        if status in ("ERROR", "XPASS") or "XPASS" in message:
            return f"unacceptable {status} {message}".strip()
        if key not in known:
            return f"undeclared {status} {message}".strip()
        if status == "FAILED" and not message.startswith(known[key]):
            return f"failed with {message!r}, declared {known[key]}"
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
