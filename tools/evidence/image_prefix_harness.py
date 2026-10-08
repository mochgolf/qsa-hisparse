#!/usr/bin/env python3
"""Track I GPU evidence harness: exact image host-prefix reuse over HTTP (I-C).

    image_prefix_harness.py --url URL --fixtures <image fixtures> --output <report>
        [--vit-log <server's QSA_EVIDENCE_VIT_DIR> [--vit-cache-off]]
        [--observer-log <server's QSA_EVIDENCE_OBSERVER_DIR>]
        [--text-fixtures <G2 fixtures.json> --text-reference <G2 fork qualification.json>
         [--text-max-prefix 8192]]

Image cases (image_fixtures.py) run one request at a time through
``/generate``, each case in fresh salts: cold (the case prompt, unique salt),
seed (the case's seed prompt in the warm salt), warm (the case prompt in the
warm salt). Greedy, the fixture's max_new_tokens, output logprobs with top 5
(input logprobs from the case's logprob_start_len), prompt token IDs (image
spans are the runs of at least 64 equal IDs). A case passes when:

- cold and seed reuse 0 tokens; warm output IDs equal cold output IDs;
- warm cached_tokens equals the frozen ``cached`` and is below prompt_tokens
  (the logits tail is computed);
- ``hit_inside`` k: start < cached < stop of the warm prompt's k-th image;
- ``aligned``: prompt_tokens is a multiple of 64;
- ``logprob_start_len`` s: cached <= s, warm returns prompt_tokens - s input
  logprobs, and warm input logprobs (and top) equal the cold control's;
- with ``--vit-log``, for each request with reuse length L (0 for controls,
  seeds and misses): no image ending at or before L is encoded on any rank,
  and with ``--vit-cache-off`` every other image is encoded on every rank.
  Requests are sequential, so the ViT records written between a request's
  send and its response belong to it;
- with ``--observer-log`` (observer.py), on every TP rank the warm request
  (by its ``meta_info.id``, the scheduler rid) has exactly one restore, at
  ``cached`` tokens, for a hit and none for a miss; cold and seed have none.
  run_image.py checks that every restore equals its capture.

With ``--vit-cache-off`` every image must also have one embedding digest per
rank across all its encodes, alone or batched with other images, and at
least one image must have been encoded both ways (ViT batch invariance).

Text control: the G2-2 qualification cases with prefix_length <=
``--text-max-prefix``, replayed like the fork harness (cold_0, cold_1, seed,
warm, repeated_warm; same payloads and salt grouping); output IDs and
cached_tokens must equal the reference (the Phase 2 fork arm). Logprob
equality is recorded, not judged.

Exit 0 only if everything passes. The report is rewritten after every case.
"""

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

TOP_LOGPROBS = 5
MIN_IMAGE_RUN = 64
PAGE = 64
TEXT_MAX_PREFIX = 8192  # G2-2 qualification cases replayed as the text control.
TEXT_REQUESTS = ("cold_0", "cold_1", "seed", "warm", "repeated_warm")
OUTPUT_LOGPROBS = ("output_token_logprobs", "output_top_logprobs")
INPUT_LOGPROBS = ("input_token_logprobs", "input_top_logprobs")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def generate(url, payload):
    """POST /generate; returns the summarized result and the raw response."""
    request = urllib.request.Request(
        url.rstrip("/") + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    sent = time.time()
    with urllib.request.urlopen(request, timeout=1800) as response:
        data = json.loads(response.read())
    received = time.time()
    info = data["meta_info"]
    if (info.get("finish_reason") or {}).get("type") in ("abort", "error"):
        raise RuntimeError(f"generation failed: {info.get('finish_reason')}")
    result = {
        "id": info["id"],
        "output_ids": data["output_ids"],
        "cached_tokens": info["cached_tokens"],
        "prompt_tokens": info["prompt_tokens"],
        "finish_reason": info.get("finish_reason"),
        **{key: info.get(key) for key in OUTPUT_LOGPROBS + INPUT_LOGPROBS},
        "sent": sent,
        "received": received,
    }
    return result, data


def image_spans(ids):
    """[start, stop) of every run of at least 64 equal token IDs (image pads)."""
    spans, start = [], 0
    for i in range(1, len(ids) + 1):
        if i == len(ids) or ids[i] != ids[start]:
            if i - start >= MIN_IMAGE_RUN:
                spans.append([start, i])
            start = i
    return spans


def image_request(url, fixtures, prompt, salt, logprob_start_len):
    data = []
    for item in prompt["images"]:
        image = "data:image/png;base64," + fixtures["images"][item["name"]]["png_base64"]
        data.append(image if "detail" not in item else {"url": image, "detail": item["detail"]})
    result, raw = generate(
        url,
        {
            "text": prompt["text"],
            "image_data": data,
            "cache_salt": salt,
            "sampling_params": {"temperature": 0, "max_new_tokens": fixtures["max_new_tokens"]},
            "return_logprob": True,
            "logprob_start_len": logprob_start_len,
            "top_logprobs_num": TOP_LOGPROBS,
            "return_prompt_token_ids": True,
        },
    )
    ids = raw["prompt_token_ids"]
    result["prompt_sha256"] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    result["image_spans"] = image_spans(ids)
    result["images"] = [
        item["name"] + (f":{item['detail']}" if "detail" in item else "")
        for item in prompt["images"]
    ]
    return result


def log_rows(directory):
    """Records of every rank-<r>.jsonl in an observer directory."""
    rows = []
    for path in sorted(Path(directory).glob("rank-*.jsonl")):
        rows += [json.loads(line) for line in path.read_text().splitlines()]
    return rows


def attach_vit(record, rows):
    """Per rank: [image index (None if unknown), batch size, digest] per encode."""
    starts = {start: k for k, (start, _) in enumerate(record["image_spans"])}
    record["vit"] = {}
    for row in rows:
        if record["sent"] <= row["time"] <= record["received"]:
            for item in row["items"]:
                record["vit"].setdefault(str(row["rank"]), []).append(
                    [starts.get(item["offsets"][0][0]), len(row["items"]), item["sha256"]]
                )


def judge_vit(kind, record, ranks, cache_off):
    reused = record["cached_tokens"]
    spans = record["image_spans"]
    done = {k for k, (_, stop) in enumerate(spans) if stop <= reused}
    failures = []
    for rank in sorted(ranks):
        encoded = [k for k, _, _ in record["vit"].get(rank, [])]
        if None in encoded:
            failures.append(f"{kind}: rank {rank} encoded an image at no image span")
        early = sorted(done & set(encoded))
        missing = sorted(set(range(len(spans))) - done - set(encoded))
        if early:
            failures.append(f"{kind}: rank {rank} encoded image(s) {early} ending at or before {reused}")
        if cache_off and missing:
            failures.append(f"{kind}: rank {rank} did not encode image(s) {missing}")
    return failures


def judge_restores(row, cached, rows):
    """Per rank, the restore lengths of each request's rid: [cached] for a warm
    hit, [] otherwise (a restore of another request never counts)."""
    ranks = sorted({r["rank"] for r in rows})
    if not ranks:
        return ["the checkpoint observer log holds no records"]
    failures = []
    for kind, length in (("cold", 0), ("seed", 0), ("warm", cached)):
        record, expected = row[kind], [length] if length else []
        record["restores"] = {
            str(rank): [
                r["tokens"]
                for r in rows
                if r["event"] == "restore" and r["rank"] == rank and r["rid"] == record["id"]
            ]
            for rank in ranks
        }
        for rank, lengths in record["restores"].items():
            if lengths != expected:
                failures.append(f"{kind}: rank {rank} restored {lengths}, expected {expected}")
    return failures


def judge(case, row):
    cold, warm = row["cold"], row["warm"]
    failures = [
        f"{kind} reused {row[kind]['cached_tokens']} tokens in a fresh salt"
        for kind in ("cold", "seed")
        if row[kind]["cached_tokens"] != 0
    ]
    if warm["output_ids"] != cold["output_ids"]:
        failures.append("warm output IDs differ from the cold control")
    reused, length = warm["cached_tokens"], warm["prompt_tokens"]
    if reused != case["cached"]:
        failures.append(f"warm reused {reused} tokens, expected {case['cached']}")
    if reused >= length:
        failures.append(f"warm reused {reused} of {length} prompt tokens: no logits tail")
    k = case["hit_inside"]
    if k is not None:
        spans = warm["image_spans"]
        span = spans[k] if k < len(spans) else None
        if span is None or not span[0] < reused < span[1]:
            failures.append(f"reuse {reused} is not inside image {k} (span {span})")
    if case.get("aligned") and length % PAGE:
        failures.append(f"prompt length {length} is not a multiple of {PAGE}")
    start = case.get("logprob_start_len")
    if start is not None:
        if reused > start:
            failures.append(f"warm reused {reused} tokens past logprob_start_len {start}")
        count = len(warm["input_token_logprobs"] or ())
        if count != length - start:
            failures.append(f"warm returned {count} input logprobs, expected {length - start}")
        failures += [
            f"warm {key} differ from the cold control"
            for key in INPUT_LOGPROBS
            if warm[key] != cold[key]
        ]
    return failures


def run_case(url, fixtures, case, salt, vit_log, cache_off, observer_log):
    row = {"name": case["name"], "cached": case["cached"], "hit_inside": case["hit_inside"]}
    start = case.get("logprob_start_len", -1)
    try:
        row["cold"] = image_request(url, fixtures, case["prompt"], f"{salt}-cold", start)
        row["seed"] = image_request(url, fixtures, case["seed"], f"{salt}-warm", -1)
        row["warm"] = image_request(url, fixtures, case["prompt"], f"{salt}-warm", start)
        failures = judge(case, row)
        if observer_log is not None:
            failures += judge_restores(row, case["cached"], log_rows(observer_log))
        if vit_log is not None:
            rows = log_rows(vit_log)
            ranks = {str(r["rank"]) for r in rows}
            if not ranks:
                failures.append("the ViT log holds no records")
            for kind in ("cold", "seed", "warm"):
                attach_vit(row[kind], rows)
                failures += judge_vit(kind, row[kind], ranks, cache_off)
        row["logprobs_equal"] = all(row["warm"][k] == row["cold"][k] for k in OUTPUT_LOGPROBS)
    except Exception as exc:  # Recorded as this case's failure; later cases still run.
        failures = [f"{type(exc).__name__}: {exc}"]
    row["failures"] = failures
    row["passed"] = not failures
    return row


def batch_invariance(rows):
    """Per image and rank: digests of every encode, alone or batched."""
    groups = {}
    for row in rows:
        for kind in ("cold", "seed", "warm"):
            record = row.get(kind) or {}
            for rank, encodes in (record.get("vit") or {}).items():
                for k, size, digest in encodes:
                    if k is not None:
                        group = groups.setdefault((record["images"][k], rank), ([], []))
                        group[size > 1].append(digest)
    images, failures = [], []
    for (image, rank), (alone, batched) in sorted(groups.items()):
        digests = sorted(set(alone + batched))
        images.append(
            {"image": image, "rank": rank, "alone": len(alone), "batched": len(batched), "digests": digests}
        )
        if len(digests) > 1:
            failures.append(f"image {image} rank {rank}: {len(digests)} different embeddings")
    if not any(entry["alone"] and entry["batched"] for entry in images):
        failures.append("no image was encoded both alone and batched")
    return {"images": images, "failures": failures}


def load_text(fixtures_path, reference_path, max_prefix):
    data = fixtures_path.read_bytes()
    reference = json.loads(reference_path.read_text())
    if hashlib.sha256(data).hexdigest() != reference["fixtures_sha256"]:
        raise SystemExit("--text-fixtures are not the fixtures of --text-reference")
    expected = {case["name"]: case for case in reference["cases"]}
    cases = [c for c in json.loads(data)["cases"] if c["prefix_length"] <= max_prefix]
    missing = [c["name"] for c in cases if c["name"] not in expected]
    if not cases or missing:
        raise SystemExit(f"no text cases selected or missing from the reference: {missing}")
    return [(case, expected[case["name"]]) for case in cases]


def text_request(url, ids, salt, count):
    """The fork harness's request (qsa_hisparse_prefix_acceptance.generate)."""
    result, _ = generate(
        url,
        {
            "input_ids": ids,
            "cache_salt": salt,
            "sampling_params": {"temperature": 0, "max_new_tokens": count},
            "return_logprob": True,
            "logprob_start_len": -1,
            "top_logprobs_num": TOP_LOGPROBS,
        },
    )
    return result


def run_text(url, pairs, namespace):
    rows = []
    for index, (case, reference) in enumerate(pairs):
        row = {"name": f"text {case['name']}"}
        try:
            for repeat in (0, 1):
                row[f"cold_{repeat}"] = text_request(
                    url, case["input_ids"], f"{namespace}-cold-{index}-{repeat}", case["max_new_tokens"]
                )
            salt = f"{namespace}-warm-{case['group']}"
            row["seed"] = text_request(url, case["prefix_ids"], salt, 1)
            for kind in ("warm", "repeated_warm"):
                row[kind] = text_request(url, case["input_ids"], salt, case["max_new_tokens"])
            failures = [
                f"{kind} {field} differ from the reference"
                for kind in TEXT_REQUESTS
                for field in ("output_ids", "cached_tokens")
                if row[kind][field] != reference[kind][field]
            ]
            row["logprobs_equal_reference"] = all(
                row[kind][key] == reference[kind]["meta_info"].get(key)
                for kind in TEXT_REQUESTS
                for key in OUTPUT_LOGPROBS
            )
        except Exception as exc:
            failures = [f"{type(exc).__name__}: {exc}"]
        row["failures"] = failures
        row["passed"] = not failures
        rows.append(row)
        print(f"{'Passed' if row['passed'] else 'FAILED'} {row['name']}", flush=True)
    return rows


def parse(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vit-log", type=Path)
    parser.add_argument("--vit-cache-off", action="store_true")
    parser.add_argument("--observer-log", type=Path)
    parser.add_argument("--text-fixtures", type=Path)
    parser.add_argument("--text-reference", type=Path)
    parser.add_argument("--text-max-prefix", type=int, default=TEXT_MAX_PREFIX)
    args = parser.parse_args(argv)
    if (args.text_fixtures is None) != (args.text_reference is None):
        parser.error("--text-fixtures and --text-reference go together")
    if args.vit_cache_off and args.vit_log is None:
        parser.error("--vit-cache-off needs --vit-log")
    return args


def main(argv=None):
    args = parse(argv)
    data = args.fixtures.read_bytes()
    fixtures = json.loads(data)
    text = args.text_fixtures and load_text(args.text_fixtures, args.text_reference, args.text_max_prefix)
    namespace = f"qsa-image-{time.time_ns()}"
    report = {
        "fixtures_sha256": hashlib.sha256(data).hexdigest(),
        "base_url": args.url,
        "vit_log": None if args.vit_log is None else str(args.vit_log),
        "vit_cache_off": args.vit_cache_off,
        "observer_log": None if args.observer_log is None else str(args.observer_log),
        "started_at": time.time(),
        "cases": [],
    }
    for index, case in enumerate(fixtures["cases"]):
        row = run_case(
            args.url, fixtures, case, f"{namespace}-{index}",
            args.vit_log, args.vit_cache_off, args.observer_log,
        )  # fmt: skip
        report["cases"].append(row)
        write_json(args.output, report)
        print(f"{'Passed' if row['passed'] else 'FAILED'} {case['name']}: {row['failures']}", flush=True)
    sections = [row["failures"] for row in report["cases"]]
    names = [row["name"] for row in report["cases"]]
    if args.vit_cache_off:
        report["vit_batch_invariance"] = batch_invariance(report["cases"])
        sections.append(report["vit_batch_invariance"]["failures"])
        names.append("ViT batch invariance")
    if text:
        report["text_control"] = run_text(args.url, text, f"{namespace}-text")
        sections += [row["failures"] for row in report["text_control"]]
        names += [row["name"] for row in report["text_control"]]
    report["failures"] = [f"{name}: {f}" for name, fs in zip(names, sections) for f in fs]
    report["passed"] = not report["failures"]
    report["finished_at"] = time.time()
    write_json(args.output, report)
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {len(report['failures'])} failure(s)")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
