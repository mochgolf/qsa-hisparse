#!/usr/bin/env python3
"""Freeze the Track I image prefix fixtures (I-C).

    image_fixtures.py freeze --output <new file>

Offline and deterministic: synthetic PNG images from fixed seeds, embedded in
the file, and native ``/generate`` prompts: text with SGLang's ``<image>``
placeholder plus an ordered image list (name, optional ``detail``). No
tokenizer is used. Every case freezes its expected restore length
(``cached``; 0 is a miss).

Why each length holds for any reasonable tokenizer. With patch 16 and spatial
merge 2, an H x W image (sides multiples of 32, within the processor's pixel
range) spans (H/32)(W/32) tokens between its vision start and end tokens.
Host checkpoints land at chunk ends (multiples of the base profile's
``--chunked-prefill-size 2048``) and at a prompt's last page64 boundary; a
hit leaves at least the last prompt token and stays within
``logprob_start_len``. In ``intro <A> between <B> question`` the texts before
B are short, so A (2304 tokens) contains token 2048 and B (2048 tokens)
starts after A and contains 4096. The notes after B are long enough that the
common prefix of the base and divergent questions lies in (6144, 8192), and
the base ending is over 64 tokens, so the base prompt's last page64
checkpoint lies after that common prefix. The seed of every case except
page-aligned is the base prompt: checkpoints 2048 (in A), 4096 (in B), 6144
(after B) and its last page64 boundary. The page-aligned prompt has no text:
2 + 2304 + 2 + 1980 = 4288 = 67 x 64 tokens (no BOS; the harness checks).
"""

import argparse
import base64
import hashlib
import json
import random
import struct
import sys
import zlib
from pathlib import Path

PIXELS_PER_TOKEN_SIDE = 32  # patch 16 x spatial merge 2
IMAGES = {  # name: (seed, height, width)
    "A": (1, 1536, 1536),  # 2304 tokens
    "B": (2, 1024, 2048),  # 2048 tokens
    "A2": (3, 1536, 1536),
    "B2": (4, 1024, 2048),
    "C": (5, 1152, 1760),  # 1980 tokens
}
MAX_TOKENS = 32
LOGPROB_START = 5000  # Between checkpoints 4096 and 6144.
INTRO = "Look at the first picture: <image>"
BETWEEN = "\nNow look at the second picture: <image>"
NOTES = "".join(f"\nNote {k}: keep this line unchanged." for k in range(1, 251))
END = (
    "\nQuestion: describe the colors of both pictures, compare their overall "
    "brightness and the contrast between neighboring blocks, name at least three "
    "colors together with their positions in each picture, estimate how many "
    "clearly distinct colors each picture contains, point out any row or column "
    "that stands out, and suggest a short title for each picture that reflects "
    "its palette and mood. Answer in one paragraph of about five sentences, "
    "without lists, headings or any other formatting, and do not repeat these "
    "instructions or the notes."
)
OTHER_END = "\nTask: list the three most common colors in the second picture, one per line."
OTHER_QUESTION = "\nWhich picture contains more blue blocks? Reply with one word."
ACCEPTANCE = (
    "Per case: a salted cold control, the seed in the case's warm salt, then the "
    "warm request (greedy, max_new_tokens fixed). Warm output IDs equal cold "
    "output IDs; cold controls and seeds reuse 0 tokens; the warm request reuses "
    "exactly `cached` tokens and leaves the logits tail; with hit_inside k the "
    "hit lies strictly inside the warm prompt's k-th image span; page-aligned: "
    "the prompt length is a multiple of 64; input-logprob: cached <= "
    "logprob_start_len and input logprobs equal the cold control's. ViT: no "
    "image ending at or before the hit is encoded; with the per-image ViT cache "
    "off every other image is encoded on every rank, and each image's "
    "embedding bytes are equal alone and batched."
)


def png(seed, height, width):
    """8-bit RGB PNG of random 32 x 32 pixel blocks (one color per image token)."""
    side = PIXELS_PER_TOKEN_SIDE
    rng = random.Random(seed)
    rows = [
        b"".join(bytes(rng.randrange(256) for _ in range(3)) * side for _ in range(width // side))
        for _ in range(height // side)
    ]
    raw = b"".join(b"\x00" + rows[y // side] for y in range(height))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def prompt(first, second, question, detail=None):
    image = {"name": first} if detail is None else {"name": first, "detail": detail}
    return {"text": INTRO + BETWEEN + question, "images": [image, {"name": second}]}


BASE = prompt("A", "B", NOTES + END)
ALIGNED = {"text": "<image><image>", "images": [{"name": "A"}, {"name": "C"}]}

# name, seed, prompt, cached, hit_inside (image index), extra fields, what it shows
CASES = [
    (
        "divergent-suffix", BASE, prompt("A", "B", NOTES + OTHER_END), 6144, None, {},
        "Same through B and the notes, different ending: hit at 6144, after both "
        "images (no image is encoded for the warm request).",
    ),
    (
        "different-image-after-boundary", BASE, prompt("A", "B2", NOTES + END), 2048, 0, {},
        "B replaced by B2 (same size): hit at 2048, inside A; B2 starts after "
        "the hit, so it is not part of the key.",
    ),
    (
        "boundary-inside-b", BASE, prompt("A", "B", OTHER_QUESTION), 4096, 1, {},
        "Same through B, a different question: hit at 4096, inside B.",
    ),
    (
        "page-aligned", ALIGNED, ALIGNED, 4096, 1, {"aligned": True},
        "A 4288-token prompt (its own seed) published a checkpoint at 4288; the "
        "repeat reuses 4096 (inside C) and leaves the logits tail.",
    ),
    (
        "input-logprob", BASE, BASE, 4096, 1, {"logprob_start_len": LOGPROB_START},
        "The base prompt with logprob_start_len 5000: hit at 4096 (inside B), "
        "never past 5000; input logprobs equal the cold control's.",
    ),
    (
        "different-image-a", BASE, prompt("A2", "B", NOTES + END), 0, None, {},
        "A replaced by A2 (same size): every checkpoint holds A.",
    ),
    (
        "swapped", BASE, prompt("B", "A", NOTES + END), 0, None, {},
        "B first, A second: every checkpoint holds the first image.",
    ),
    (
        "different-preprocessing", BASE, prompt("A", "B", NOTES + END, "high"), 0, None, {},
        "A's bytes with detail 'high': a different artifact key (preprocess "
        "kwargs). At the pin Qwen-VL pixel limits come only from the server's "
        "--mm-process-config; detail changes the key, not the pixels.",
    ),
]  # fmt: skip


def build():
    images = {}
    for name, (seed, height, width) in IMAGES.items():
        data = png(seed, height, width)
        side = PIXELS_PER_TOKEN_SIDE
        images[name] = {
            "seed": seed,
            "height": height,
            "width": width,
            "tokens": (height // side) * (width // side),
            "sha256": hashlib.sha256(data).hexdigest(),
            "png_base64": base64.b64encode(data).decode(),
        }
    cases = [
        {"name": name, "cached": cached, "hit_inside": inside, **extra, "shows": shows,
         "seed": seed, "prompt": case_prompt}
        for name, seed, case_prompt, cached, inside, extra, shows in CASES
    ]  # fmt: skip
    return {
        "format": 1,
        "generator": "tools/evidence/image_fixtures.py freeze",
        "endpoint": "/generate",
        "acceptance": ACCEPTANCE,
        "layout": __doc__.split("\n\n", 3)[3].strip(),
        "max_new_tokens": MAX_TOKENS,
        "images": images,
        "cases": cases,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["freeze"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    data = (json.dumps(build(), indent=2) + "\n").encode()
    with args.output.open("xb") as out:  # Refuses to overwrite frozen fixtures.
        out.write(data)
    print(f"Frozen {len(CASES)} cases in {args.output} (sha256 {hashlib.sha256(data).hexdigest()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
