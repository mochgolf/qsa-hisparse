# GPU window 3 results (2026-10-08)

Production stopped 04:41 and restored 05:02 (ready, same profile hash,
`/health` 200, same model id and GPU memory). Raw evidence (private):
`qwen:results/plugin-window3-20261008/` (`i5/summary.json`,
`i5/<session>/image-prefix.json`, `d5/results.json`). Plugin main
`4476a37` (D5 included), image fixtures sha `59f72d85…` (10 cases), text
control against Phase 2's fork qualification (`plugin-g2-20261007/run2`).

## I5: Track I image evidence — passed

`run_image.py`: two fresh plugin servers (deterministic p2-offload profile,
TP2, `--mm-preprocess-cache-size-mb 512`), harness exit 0 and no observer
problems in both sessions.

| Case | Expected hit | Hit (on / off) | Restore length, rank 0 / 1 |
| --- | --- | --- | --- |
| divergent-suffix | 6144 | 6144 / 6144 | 6144 / 6144 |
| different-image-after-boundary | 2048 (inside A) | 2048 / 2048 | 2048 / 2048 |
| boundary-inside-b | 4096 (inside B) | 4096 / 4096 | 4096 / 4096 |
| page-aligned | 4096 | 4096 / 4096 | 4096 / 4096 |
| page-boundary-inside-b | 4352 (inside B) | 4352 / 4352 | 4352 / 4352 |
| odd-page-boundary-inside-c | 3904 (inside C) | 3904 / 3904 | 3904 / 3904 |
| input-logprob | 4096 | 4096 / 4096 | 4096 / 4096 |
| different-image-a, swapped, different-preprocessing | 0 | 0 / 0 | none |

- Criteria 1–3: hits at the expected page64 boundaries, including inside
  images (4352 is not a multiple of 2048); every negative case misses.
- Criterion 4: each image hit has its own restore at the expected length on
  both TP ranks, and every restore's observer hashes equal the capture it
  restored (compare.py), in both sessions.
- Criterion 5: hits leave the logits tail; the input-logprob case returns
  2361 input logprobs, equal to its uncached control in both sessions.
- Criterion 6: warm token IDs equal the cold controls in every case and in
  both sessions, and match across sessions. Most cases generate one token
  (fixture design); divergent-suffix 11, boundary-inside-b 7. With the ViT
  cache on, no warm request encodes an image; with it off, warm requests
  never encode an image wholly before the hit, and encode the straddling and
  later images. The pin re-encodes an image for every prefill chunk it
  overlaps when the ViT cache is off, so B appears both batched and alone.
  Per-image embedding digests are equal alone and batched (batch invariance).
- Criterion 7: the 10 text-control cases (prefix 64–8192, copy and
  arithmetic) pass against the fork reference in both sessions.

## D5 logprob check — passed

An image request with `return_logprob`, `top_logprobs_num` 5,
`logprob_start_len` -1 and 32 new tokens finishes normally on the plugin
(11 output tokens, 11 output logprobs, stop token); the server log has no
request-time error (only startup import notices for the unused inkling
model). In window 2 the same request crashed both arms (device-side index out
of bounds in `get_token_ids_logprobs`); that is the failing control.
