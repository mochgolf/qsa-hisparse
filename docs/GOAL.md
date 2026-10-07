# Goal

Turn the QSA HiSparse SGLang fork into an independently installed SGLang
general plugin, then add exact multimodal (image) host prefix reuse, while
upstreaming interfaces that shrink the plugin's internal replacements.

Set 2026-10-07. Owner decisions are final unless the owner changes them.

## Decisions

1. Pin: SGLang `76e06febab732d28a61b75a61b7835284568cdfb`. The reference
   behavior is fork `ee8fe158d64186b47236b007a299696030c372e8`. Upgrading the
   pin is a separate later cycle, never mixed with migration.
2. Two independent switches: `SGLANG_QSA_MODEL_COMPAT=1` (fork shared-path
   behavior) and `SGLANG_QSA_HISPARSE_V3=<mode>` (offload runtime and host
   prefixes; requires model compat). No switch: zero patches.
3. Image prefixes are reused at every page64 boundary, including boundaries
   inside an image. No outside-image-only first version.
4. Upstream PRs are part of the work. Branches are prepared locally; pushing
   or opening a PR waits for explicit owner confirmation each time.
5. Work proceeds as parallel tasks by multiple agents. Every phase ends with a
   read-only review by `gpt-6.1-sol` at `xhigh` reasoning via the Codex CLI;
   findings are resolved or explicitly answered before the next phase starts.

## Done means

- Plugin on (`compat` and `compat+hisparse`) reproduces the fork: CPU suite,
  kernel and graph checks, and frozen deterministic HTTP fixtures with equal
  token IDs and exact cached bytes/state.
- Plugin installed but off is pristine upstream: no patch registered, upstream
  test subset passes. Non-target models are unaffected.
- Any upstream edit to a patched definition fails activation (fingerprints);
  activation failures stop the process (SGLang's loader swallows `Exception`).
- Lease ordering is preserved: restore, async copy, logical release,
  free-group flush, then physical slot reuse.
- Image prefix hits match tokens, image content/preprocessing identity
  (full artifact key, not the 30-bit pad value), order/offsets/grid, and
  prefix M-RoPE positions, without a whole-prompt digest; page64, complete
  recurrent/PLE/C4 state and the logits tail are preserved. Cold, warm and
  divergent-suffix outputs are identical under the deterministic profile.
- Upstream PRs for the interface and fix list in `PLAN.md` are prepared with
  tests; merged ones replace the matching plugin REPLACE patches.

## Constraints

- Do not start, stop, or replace the production service. GPU work needs an
  owner-approved test window and runs exclusively, one job at a time.
- Do not modify shared Python environments (`flash-next-env`, `quant-env`);
  tests select sources with `PYTHONPATH`.
- No publication (push, PR, issue, release) without explicit confirmation.
- No tolerance fitting or fixture filtering; failures are reported as failures.
