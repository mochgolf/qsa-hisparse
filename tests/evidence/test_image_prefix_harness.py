"""Track I evidence harness (I-C) against a fake SGLang ``/generate`` server.

The fake tokenizes like a byte-level BPE pre-tokenizer (words, digits,
punctuation, whitespace), lays images out as Qwen-VL does (vision start,
(H/32)(W/32) pad tokens, vision end; pad values from the image bytes and
``detail``), publishes host checkpoints at chunk ends and at the last page64
boundary, reuses the longest equal checkpoint within the logits-tail and
input-logprob limits, and writes ViT encode records like vit_observer.py
(per chunk: the images it computes; with the per-image cache on, only images
never encoded before). Each fault breaks one behavior the harness judges.
"""

import base64
import contextlib
import hashlib
import http.server
import json
import re
import struct
import threading
import time

import pytest

CHUNK, PAGE = 2048, 64
VISION_START, VISION_END, IMAGE_PAD = 11, 12, 13
TEXT_LENGTHS = (64, 128, 2048, 16384)  # 16384 lies beyond the text control's 8192.


class FakeModel:
    def __init__(self, fault=None, digits=r"\d", cache_off=False, vit_dir=None):
        self.fault, self.cache_off, self.vit_dir = fault, cache_off, vit_dir
        self.pieces = re.compile(rf" ?[A-Za-z]+|{digits}|[^\w\s]+|\s+")
        self.published = {}  # salt -> [(tokens, checkpoint lengths)]
        self.encoded = set()  # per-image ViT cache

    def token(self, piece):
        return 1000 + int.from_bytes(hashlib.sha256(piece.encode()).digest()[:3], "little")

    def layout(self, payload):
        """Padded tokens (pad values), prompt token IDs, image (start, stop, key)."""
        if "input_ids" in payload:
            return payload["input_ids"], payload["input_ids"], []
        texts = payload["text"].split("<image>")
        padded, plain, items = [], [], []
        for index, text in enumerate(texts):
            ids = [self.token(p) for p in self.pieces.findall(text)]
            padded, plain = padded + ids, plain + ids
            if index == len(texts) - 1:
                break
            image = payload["image_data"][index]
            url, detail = (image, "") if isinstance(image, str) else (image["url"], image["detail"])
            png = base64.b64decode(url.split(",", 1)[1])
            width, height = struct.unpack(">II", png[16:24])
            key = hashlib.sha256(png + detail.encode()).hexdigest()
            start, count = len(padded) + 1, (width // 32) * (height // 32)
            padded = padded + [VISION_START] + [100 + int(key[:6], 16)] * count + [VISION_END]
            plain = plain + [VISION_START] + [IMAGE_PAD] * count + [VISION_END]
            items.append((start, start + count, key))
        return padded, plain, items

    def handle(self, payload):
        image = "text" in payload
        fault = self.fault if (self.fault or "").startswith("text_") != image else None
        padded, plain, items = self.layout(payload)
        if fault == "bos":  # A tokenizer that prepends a token: nothing is page-aligned.
            padded, plain, items = [1, *padded], [1, *plain], [(s + 1, e + 1, k) for s, e, k in items]
        n, start = len(padded), payload["logprob_start_len"]
        matched = plain if fault == "pad_collision" else padded
        salt = "shared" if fault == "shared_salt" else payload["cache_salt"]
        limit = n if fault == "full_hit" else n - 1
        if start >= 0 and fault != "ignore_logprob_limit":
            limit = min(limit, start)
        reuse = max(
            (
                length
                for tokens, lengths in self.published.get(salt, [])
                for length in lengths
                if length <= limit and tokens[:length] == matched[:length]
            ),
            default=0,
        )
        if fault == "no_hit":
            reuse = 0
        boundary = n // PAGE * PAGE if n % PAGE else n
        lengths = [*range(reuse + CHUNK, boundary, CHUNK), *([boundary] if boundary > reuse else [])]
        self.published.setdefault(salt, []).append((matched, lengths))
        self.encode(items, reuse, [*[x for x in lengths if x < n], n], fault)

        count = payload["sampling_params"]["max_new_tokens"]
        out = list(hashlib.sha256(json.dumps(padded).encode()).digest())[:count]
        if reuse and fault in ("warm_ids", "text_ids"):
            out[0] ^= 1
        inputs = [[-0.25, t, None] for t in plain[start:]] if start >= 0 else []
        if reuse and fault == "input_logprobs" and inputs:
            inputs[0] = [-0.5, *inputs[0][1:]]
        if reuse and fault == "input_logprobs_short":
            inputs = inputs[1:]
        reported = reuse + {"late_hit": 128, "text_cached": 64}.get(fault, 0) * bool(reuse)
        response = {
            "output_ids": out,
            "meta_info": {
                "cached_tokens": reported,
                "prompt_tokens": n,
                "finish_reason": {"type": "length", "length": count},
                "output_token_logprobs": [[-0.5, t, None] for t in out],
                "output_top_logprobs": [[[-0.5, t, None]] for t in out],
                "input_token_logprobs": inputs,
                "input_top_logprobs": [[entry] for entry in inputs],
            },
        }
        if payload.get("return_prompt_token_ids"):
            response["prompt_token_ids"] = plain
        return response

    def encode(self, items, reuse, chunk_ends, fault):
        if self.vit_dir is None or fault == "no_vit_log" or (fault == "vit_stale" and reuse):
            return
        begin = reuse
        for end in chunk_ends:
            # vit_reencode: a hit re-encodes every image of its request per chunk.
            batch = [i for i in items if (fault == "vit_reencode" and reuse) or (i[1] > begin and i[0] < end)]
            if not self.cache_off:
                batch = [i for i in batch if i[2] not in self.encoded]
                self.encoded.update(i[2] for i in batch)
            begin = end
            if not batch:
                continue
            for rank in (0, 1):
                row = {"time": time.time(), "rank": rank, "items": [
                    {"offsets": [[s, e - 1]], "hash": int(key[:12], 16),
                     "sha256": hashlib.sha256((key + (str(len(batch)) if fault == "vit_batch" else "")).encode()).hexdigest()}
                    for s, e, key in batch
                ]}  # fmt: skip
                with (self.vit_dir / f"rank-{rank}.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")


@contextlib.contextmanager
def serve(model):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == "/generate"
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body = json.dumps(model.handle(payload)).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def text_fixtures(path):
    cases = []
    for length in TEXT_LENGTHS:
        prefix = [5000 + i % 97 for i in range(length)]
        for name, suffix in (("copy", [7, 8, 9]), ("arithmetic", [7, 6, 5, 4])):
            cases.append({"name": f"prefix-{length}-{name}", "group": f"prefix-{length}",
                          "prefix_length": length, "prefix_ids": prefix,
                          "input_ids": prefix + suffix, "max_new_tokens": 32})  # fmt: skip
    path.write_text(json.dumps({"format": 1, "cases": cases}))
    return path


def reference(text):
    """The fork harness's request sequence (cold_0, cold_1, seed, warm,
    repeated_warm; one warm salt per group) replayed directly on a correct fake."""
    model, cases = FakeModel(), []
    selected = [c for c in json.loads(text.read_text())["cases"] if c["prefix_length"] <= 8192]
    for index, case in enumerate(selected):

        def call(ids, salt, count):
            response = model.handle({"input_ids": ids, "cache_salt": salt, "logprob_start_len": -1,
                                     "sampling_params": {"max_new_tokens": count}})  # fmt: skip
            info = response["meta_info"]
            return {"output_ids": response["output_ids"], "cached_tokens": info["cached_tokens"], "meta_info": info}

        entry = {"name": case["name"]}
        for repeat in (0, 1):
            entry[f"cold_{repeat}"] = call(case["input_ids"], f"cold-{index}-{repeat}", case["max_new_tokens"])
        entry["seed"] = call(case["prefix_ids"], f"warm-{case['group']}", 1)
        for kind in ("warm", "repeated_warm"):
            entry[kind] = call(case["input_ids"], f"warm-{case['group']}", case["max_new_tokens"])
        cases.append(entry)
    return {"fixtures_sha256": hashlib.sha256(text.read_bytes()).hexdigest(), "cases": cases}


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    """Image fixtures from the real freezer, text fixtures and their reference."""
    import importlib.util
    from pathlib import Path

    evidence = Path(__file__).resolve().parents[2] / "tools" / "evidence"
    spec = importlib.util.spec_from_file_location("qsa_image_fixtures_test", evidence / "image_fixtures.py")
    freezer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(freezer)
    root = tmp_path_factory.mktemp("inputs")
    images = root / "images.json"
    images.write_text(json.dumps(freezer.build()))
    text = text_fixtures(root / "text.json")
    expected = root / "reference.json"
    expected.write_text(json.dumps(reference(text)))
    return images, text, expected


def run(evidence, inputs, tmp_path, model, *extra):
    images, text, reference = inputs
    output = tmp_path / "report.json"
    vit = tmp_path / "vit"
    vit.mkdir()
    model.vit_dir = vit
    args = ["--fixtures", str(images), "--output", str(output), "--vit-log", str(vit),
            "--text-fixtures", str(text), "--text-reference", str(reference), *extra]  # fmt: skip
    if model.cache_off:
        args.append("--vit-cache-off")
    with serve(model) as url:
        status = evidence("image_prefix_harness").main(["--url", url, *args])
    report = json.loads(output.read_text())
    assert status == (0 if report["passed"] else 1)
    return report


def failing(report):
    rows = report["cases"] + report.get("text_control", [])
    names = {row["name"] for row in rows if row["failures"]}
    if (report.get("vit_batch_invariance") or {}).get("failures"):
        names.add("ViT batch invariance")
    return names


@pytest.mark.parametrize("cache_off", [False, True])
@pytest.mark.parametrize("digits", [r"\d", r"\d+"])
def test_correct_server_passes_with_the_frozen_reuse_lengths(evidence, inputs, tmp_path, cache_off, digits):
    report = run(evidence, inputs, tmp_path, FakeModel(digits=digits, cache_off=cache_off))
    assert report["passed"], report["failures"]
    reused = {row["name"]: row["warm"]["cached_tokens"] for row in report["cases"]}
    assert reused == {
        "divergent-suffix": 6144,
        "different-image-after-boundary": 2048,
        "boundary-inside-b": 4096,
        "page-aligned": 4096,
        "input-logprob": 4096,
        "different-image-a": 0,
        "swapped": 0,
        "different-preprocessing": 0,
    }
    warm = {row["name"]: row["warm"] for row in report["cases"]}
    assert warm["page-aligned"]["prompt_tokens"] == 4288
    assert len(warm["input-logprob"]["input_token_logprobs"]) == warm["input-logprob"]["prompt_tokens"] - 5000
    # ViT work: no image is encoded for a hit after both; only the straddling
    # and later images otherwise (ViT cache off); nothing again with it on.
    encoded = {name: sorted({k for k, _, _ in record["vit"].get("1", [])}) for name, record in warm.items()}
    if cache_off:
        assert encoded["divergent-suffix"] == []
        assert encoded["different-image-after-boundary"] == [0, 1]
        assert encoded["boundary-inside-b"] == encoded["page-aligned"] == [1]
        invariance = report["vit_batch_invariance"]["images"]
        assert {e["image"] for e in invariance if e["alone"] and e["batched"]} >= {"A", "B"}
    else:
        assert encoded["divergent-suffix"] == encoded["boundary-inside-b"] == []
        assert "vit_batch_invariance" not in report
    assert len(report["text_control"]) == 6 and all(row["passed"] for row in report["text_control"])


IMAGE_HITS = {"divergent-suffix", "different-image-after-boundary", "boundary-inside-b", "page-aligned", "input-logprob"}
ALL_IMAGE = IMAGE_HITS | {"different-image-a", "swapped", "different-preprocessing"}
TEXT = {f"text prefix-{n}-{s}" for n in TEXT_LENGTHS[:3] for s in ("copy", "arithmetic")}


@pytest.mark.parametrize(
    "fault, cache_off, names, message",
    [
        ("warm_ids", False, IMAGE_HITS, "warm output IDs differ"),
        ("no_hit", False, IMAGE_HITS, "warm reused 0 tokens, expected"),
        ("shared_salt", False, ALL_IMAGE, "in a fresh salt"),
        ("pad_collision", False, {"different-image-a", "swapped", "different-preprocessing",
                                  "different-image-after-boundary"}, "expected"),
        ("late_hit", False, IMAGE_HITS, "warm reused"),
        ("full_hit", False, {"page-aligned"}, "no logits tail"),
        ("ignore_logprob_limit", False, {"input-logprob"}, "past logprob_start_len"),
        ("input_logprobs", False, {"input-logprob"}, "input_token_logprobs differ"),
        ("input_logprobs_short", False, {"input-logprob"}, "input logprobs, expected 2361"),
        ("bos", False, {"page-aligned"}, "4289 is not a multiple of 64"),
        ("no_vit_log", False, ALL_IMAGE, "the ViT log holds no records"),
        ("vit_reencode", True, IMAGE_HITS - {"different-image-after-boundary"}, "ending at or before"),
        ("vit_stale", True, IMAGE_HITS - {"divergent-suffix"}, "did not encode"),
        ("vit_batch", True, {"ViT batch invariance"}, "different embeddings"),
        ("text_ids", False, TEXT, "warm output_ids differ from the reference"),
        ("text_cached", False, TEXT, "cached_tokens differ from the reference"),
    ],
)  # fmt: skip
def test_each_criterion_has_a_failing_counterexample(
    evidence, inputs, tmp_path, fault, cache_off, names, message
):
    report = run(evidence, inputs, tmp_path, FakeModel(fault, cache_off=cache_off))
    assert not report["passed"]
    assert failing(report) == names
    for name in names:
        assert any(message in f for f in report["failures"] if f.startswith(name + ": ")), name


def test_hit_inside_is_judged_from_the_observed_image_offsets(evidence, inputs):
    harness = evidence("image_prefix_harness")
    case = next(c for c in json.loads(inputs[0].read_text())["cases"] if c["name"] == "boundary-inside-b")
    record = {"cached_tokens": 4096, "prompt_tokens": 4385, "output_ids": [1], "image_spans": [[8, 2312], [2323, 4371]],
              **{k: [] for k in harness.INPUT_LOGPROBS}}  # fmt: skip
    row = {"cold": dict(record, cached_tokens=0), "seed": dict(record, cached_tokens=0), "warm": record}
    assert harness.judge(case, row) == []
    record["image_spans"] = [[8, 2312], [4100, 6148]]  # The hit lies before B: not an in-image restore.
    assert harness.judge(case, row) == ["reuse 4096 is not inside image 1 (span [4100, 6148])"]
    record["image_spans"] = [[8, 2312]]
    assert harness.judge(case, row) == ["reuse 4096 is not inside image 1 (span None)"]


def test_text_inputs_must_be_the_references_fixtures(evidence, inputs, tmp_path):
    harness = evidence("image_prefix_harness")
    _, text, reference = inputs
    other = tmp_path / "other.json"
    other.write_bytes(text.read_bytes() + b"\n")
    with pytest.raises(SystemExit, match="not the fixtures"):
        harness.load_text(other, reference, 8192)
    with pytest.raises(SystemExit, match="missing from the reference"):
        harness.load_text(text, reference, 16384)
    assert len(harness.load_text(text, reference, 8192)) == 6
