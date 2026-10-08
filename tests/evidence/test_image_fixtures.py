"""Track I image fixture freezer (I-C): determinism, images and the frozen cases."""

import base64
import hashlib
import io
import json

import pytest
from PIL import Image

# sha256 of the frozen fixture file. A change to images, prompts or expected
# reuse lengths changes it: fixtures are frozen before any GPU run.
FROZEN_SHA256 = "09de4b3a0149d3d54cd7499d0c64f22fe26905d18806fb46dbbe00b01469946b"


def test_freeze_is_deterministic_and_refuses_to_overwrite(evidence, tmp_path):
    freezer = evidence("image_fixtures")
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    assert freezer.main(["freeze", "--output", str(first)]) == 0
    assert evidence("image_fixtures").main(["freeze", "--output", str(second)]) == 0
    assert first.read_bytes() == second.read_bytes()
    assert hashlib.sha256(first.read_bytes()).hexdigest() == FROZEN_SHA256
    first.write_text("frozen")
    with pytest.raises(FileExistsError):
        freezer.main(["freeze", "--output", str(first)])
    assert first.read_text() == "frozen"


def test_images_decode_to_their_declared_token_grids(evidence):
    images = evidence("image_fixtures").build()["images"]
    pixels = {}
    for name, image in images.items():
        data = base64.b64decode(image["png_base64"])
        assert hashlib.sha256(data).hexdigest() == image["sha256"]
        decoded = Image.open(io.BytesIO(data))
        assert decoded.mode == "RGB" and decoded.size == (image["width"], image["height"])
        assert image["width"] % 32 == image["height"] % 32 == 0
        assert image["tokens"] == image["width"] * image["height"] // 32**2 > 64
        pixels[name] = decoded.tobytes()
    assert {name: image["tokens"] for name, image in images.items()} == {
        "A": 2304, "B": 2048, "A2": 2304, "B2": 2048, "C": 1980,
    }  # fmt: skip
    assert len(set(pixels.values())) == len(pixels)  # Same sizes, different content.


def test_frozen_cases_and_expected_reuse(evidence):
    fixtures = evidence("image_fixtures").build()
    images = lambda prompt: [(i["name"], i.get("detail")) for i in prompt["images"]]  # noqa: E731
    cases = {c["name"]: c for c in fixtures["cases"]}
    assert {
        name: (c["cached"], c["hit_inside"], c.get("aligned"), c.get("logprob_start_len"), images(c["prompt"]))
        for name, c in cases.items()
    } == {
        "divergent-suffix": (6144, None, None, None, [("A", None), ("B", None)]),
        "different-image-after-boundary": (2048, 0, None, None, [("A", None), ("B2", None)]),
        "boundary-inside-b": (4096, 1, None, None, [("A", None), ("B", None)]),
        "page-aligned": (4096, 1, True, None, [("A", None), ("C", None)]),
        "page-boundary-inside-b": (4352, 1, None, None, [("A", None), ("B", None)]),
        "input-logprob": (4096, 1, None, 5000, [("A", None), ("B", None)]),
        "different-image-a": (0, None, None, None, [("A2", None), ("B", None)]),
        "swapped": (0, None, None, None, [("B", None), ("A", None)]),
        "different-preprocessing": (0, None, None, None, [("A", "high"), ("B", None)]),
    }  # fmt: skip
    base = cases["input-logprob"]["prompt"]
    images_only = {"page-aligned", "page-boundary-inside-b"}
    for name, case in cases.items():
        assert case["seed"] == (case["prompt"] if name in images_only else base)
        assert case["prompt"]["text"].count("<image>") == len(case["prompt"]["images"]) == 2
    # One expected hit is a page64 boundary that is no chunk end (2048 multiple).
    assert {c["cached"] % 2048 for c in cases.values()} == {0, 4352 % 2048}
    # Prompts that keep the base question differ from it only in their images.
    same_text = {"different-image-after-boundary", "different-image-a", "swapped", "different-preprocessing"}
    assert {n for n, c in cases.items() if c["prompt"]["text"] == base["text"]} == same_text | {"input-logprob"}
    # Image-only prompts: A + C is 2 + 2304 + 2 + 1980 = 4288 tokens (aligned);
    # A + B is 4356 tokens, last page64 boundary 4352 inside B (2307, 4355).
    assert all(cases[name]["prompt"]["text"] == "<image><image>" for name in images_only)
    assert (2 + 2304 + 2 + 1980) % 64 == 0
    assert (2 + 2304 + 2 + 2048) // 64 * 64 == 4352 and 2307 < 4352 < 2307 + 2048
    assert json.loads(json.dumps(fixtures)) == fixtures
