# GPU window 2 results (2026-10-08)

Production stopped 03:00 and restored 03:51 (ready, same profile hash,
`/health` 200, same model id and GPU memory). Raw evidence (private):
`qwen:results/plugin-window2-20261008/`.

## I5: Track I image evidence — not obtained

- The ViT-cache-on session's server crashed on the first image request (a
  salted cold control, no prefix hit): CUDA device-side assert "index out of
  bounds" in `sglang/srt/layers/logprob_processor.py`
  `get_token_ids_logprobs` (decode stage), located with
  `CUDA_LAUNCH_BLOCKING=1`.
- **Inherited, not a plugin regression:** the fork arm (`ee8fe158d6`, same
  deterministic p2-offload profile plus `--mm-preprocess-cache-size-mb 512`)
  crashes identically on the same request.
- The same image request without logprob options succeeds on the plugin.
- The ViT-cache-off session did not start: its preflight saw the previous
  session's GPU processes still exiting (`run_image.py` lacked a wait).
- Next: I-C root-causes the logprob crash from the pinned code and adjusts
  the harness to avoid it (or marks the input-logprob criterion as not
  verifiable on GPU because of the inherited bug); a short window 3 reruns I5.

## Upstream branch GPU checks (local only; no PRs, owner decision)

Interpreter `results/dsh-maintenance-20261004/upstream-runtime-env`
(torch 2.14.1, sglang-kernel 0.4.9, FA4 without FA2), SM89.

| Check | Expected | Result |
| --- | --- | --- |
| U6a unit, GPU, full `test_qsa.py` | pass | pass |
| U6a control (main resolver, new test) | fail | fail (FA4 selected) |
| main `test_qsa.py` baseline | recorded | exit 0 |
| U6b unit | pass | **1 failed**: `sglang::store_cache` has no CPU kernel in sglang-kernel 0.4.9 with CUDA visible (passes with 0.4.7); environment-sensitive test, fix before any PR |
| U6b (+U6a) GPU, full | pass | pass |
| U6b control (U6a without the scale fix) | fail | fail |
| U8 unit, GPU (680 subtests, 17 min incl. JIT), wna16 | pass | pass |
| U8 control (main code, new batch-invariance test) | fail | fail (bf16 and fp16) |
