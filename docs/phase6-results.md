# Phase 6 results: v0.5.22, reference = production

Owner decisions and rules: PLAN.md "Phase 6"; survey `docs/phase6-survey.md`;
host prefix design `docs/phase6-prefix-design.md`. Tasks P6-A..D ran in
parallel worktrees and were merged into `phase6` (P6-B `586be8c`, P6-D
`80da09e`, P6-C `c618102`, P6-A `2559a47`, P6-A follow-up `b30818a`).

## CPU (G6-CPU input)

Interpreter `../results/phase6-20261018/runtime-env` (torch 2.14.1,
sglang-kernel 0.4.9, flashinfer-python 0.7.1, triton 3.8.0, transformers
5.19.0; `pip freeze` equals production's except those three bumps:
`../results/phase6-20261018/runtime-requirements.txt`).

| Check | Result |
|---|---|
| `tools/fingerprint.py check` | 402 pinned fingerprints match v0.5.22 |
| `tools/manifest.py --check` | current (64 rows) |
| `tools/run_cpu_tests.sh` | unit 723 passed, 54 skipped, 1 xfailed; subprocess integration 4 passed, 1 skipped; activated integration 11 passed; 0 failed |
| Previous state (`main`, pin 35f3c96ff4, production interpreter) | 685 + 3 + 26 passed |

The activated-integration pass shrank because the prefix tests now need only
P6-A's rows and run in the unit pass with those rows activated.

### REPLACE rows: 32 → 26

| Row | Phase 5 | Phase 6 | How |
|---|---|---|---|
| S06 | replace (332 lines) | after on `Scheduler.get_prefill_adder_class` | #43870 seam; `QSAPrefillAdder` (`prefill_max_requests = 1`) |
| P02 | replace + attach | none (in `QSAPrefillAdder.add_one_req`) | no copied body; production's budget charge and `ignore_eos` reuse kept |
| P01 | before | none (in `QSAPrefillAdder.add_chunked_req`) | |
| M01 | replace (133 lines) | around on `alloc_for_extend` | pages via `prefix_device_indices`, restore via `maybe_hand_to_session` (#42825, #42537) |
| M06 | replace | dropped | the disabled `UnifiedRadixCache` supports mamba (#43198–#43201) |
| B04, B05 | replace ×2 | one before on `_prepare_kv_cache_release` (B04) | `46498bc8c5` seam; equivalence for every caller in the reason |
| S01 | after | none | M05 builds `QSAHostPrefixCache` itself; rejections raise in the factory |
| M04 | around | dropped | #43439 (`e8fb52c526`) no longer zeroes `req_generation` |

Kept REPLACE (26): A03 A05 A09 B02 B03 E03 E06 E07 E08 G01 G02 J03 K02 M03
Q01 Q08 Q10 Q11 Q12 S04 S07 T03 T04 Z01 Z02 Z03. Hand merges (`RESOLVED`):
M03 (production's mamba free dropped: the cache owns the slot), Q08 (#41933
pack without `output_dtype`, so the FP8 descale is applied once), T03
(#43463's bound only with `fast_topk`), T04 (unchanged).

### Host prefix cache

`QSAHostPrefixCache` is a `UnifiedRadixCache` subclass built by M05 through
`create_unified_radix_cache(ctx, cache_class=...)`: a hit is reported as
`host_hit_length`, accepted in `init_load_back`, pages come from
`prefix_device_indices`, the restore runs in `maybe_hand_to_session`,
capture in `checkpoint` before `super()`. `cached_tokens`, its tier
breakdown and the admission tier statistics equal production's for a hit,
a miss and a chunked hit (`tests/prefix/test_prefix_cache.py`). Docs:
`docs/prefix-cache.md`.

### Upstream overlaps (R2)

- #41933 vs the FP8 descale (Q08): FP8 bytes reach the chunk kernel raw; the
  scale is applied once; bitwise equal to production's gather and far from
  "scale lost"/"scale doubled" (`tests/qsa/test_prefill_pack_descale.py`).
- #43463 vs T03: the context-length bound applies only with `fast_topk`;
  deterministic decode scores every row as in production.
- #42175 vs H08: new `mix` signature and 3-tuple residuals adapted; the
  stable path keeps production's combine.
- C02 vs #40102: the KV-shard scratch and production's bias are each
  subtracted once (scratch is 0 unless KV sharding is on).
- #37771: QSA coordinator gets a no-op `allocate_spec_scratch`; B03's new
  `request_finished` gate stays on `enable_hisparse` (production semantics).

Deviations D1–D3, D5–D7 still apply unchanged; D4 no longer arises (the
new seam affects only the scheduler's own adder).

### Differences from production (for owner review; none on the served path's results)

Error or unreachable paths:
1. Mamba slot release order: on the checkpointed release the disabled tree
   frees the slot in `checkpoint`, before the lease drain (production: after
   `on_release`); still freed exactly once. Only a failed drain (which kills
   the scheduler) is affected.
2. Rollback after an `alloc_aux_to_lengths` failure (impossible: QSA pools
   have no aux cache) or a `write_cache_indices` failure (process exits).
3. S01's rejections (radix cache/HiCache enabled, mamba extra buffer) raise
   while the tree cache is built, with new wording; still before serving.

Upstream v0.5.22 behavior now reaching the plugin (inert or unobservable for
the production profile; GPU confirms):
4. Per-request TP convergence and prefix page allocation happen at the start
   of `alloc_for_extend`, before the request row is allocated.
5. `_get_new_batch_prefill_raw` is upstream's again (e.g. `batch_is_full` is
   not forced on `NO_TOKEN`; S04 resets it every round under QSA).
6. Q08 without a runtime gathers with upstream's Triton pack (same bytes).
7. T03 non-deterministic decode inherits #43463 (`min_scored_len=512`) and
   the restructured TileLang MQA kernel (same selection contract).
8. The detokenizer process now loads plugins (v0.5.22 calls
   `load_plugins()` there): ~5 s of fingerprint checks at startup, no hooks
   reached, no activation record.
9. transformers 5.19.0 enters the image artifact key, so image item hashes
   and pad values differ from production's; keys stay exact within a server.

Other platforms or options only (not production): K02 on NPU/out-of-tree
platforms (p2-offload raises), C02 with KV sharding, Q10's SM100 fused
branch, H08 on SM100 in deterministic mode (keeps production's combine).

### For G6-GPU

FP8 prefix-hit prefill vs production; deterministic decode after the
TileLang MQA change; non-deterministic decode with `min_scored_len` through
the plugin's `fast_topk`; image prefix hits (I5) with the v0.5.22 image
keys; the Rust tree core (`UnifiedRadixCache`) is compiled on first use into
`~/.cache/sglang/rust_extensions` (already built by the CPU tests for this
user). Evidence tools: fork arm = production code on production's
interpreter, plugin arm = v0.5.22 on the Phase 6 interpreter
(`docs/baseline.md` "Phase 6").
