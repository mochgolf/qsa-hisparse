# P6-A scheduler, lifecycle and host prefix cache at v0.5.22

Read `P6-common.md` and `docs/phase6-prefix-design.md` (decisions 1–5 are
binding; the brief below them is verified research, re-check what you rely
on). Rows S01–S09, P01–P02, B02–B06, M01–M06 and the host prefix cache
(`hisparse/prefix_cache.py`), plus a no-op `allocate_spec_scratch` on the
QSA coordinator (`hisparse/coordinator.py`, #37771).

1. Host prefix cache: implement design (a1)+H (`QSAHostPrefixCache` as a
   `UnifiedRadixCache` subclass built through M05; hit via
   `host_hit_length`/`init_load_back`; pages via `prefix_device_indices`;
   restore in `maybe_hand_to_session`; capture in `checkpoint`). Remove S01's
   wrap and M06. `prefix_cache.py` becomes plugin-owned in
   `tests/runtime/test_moved_sources.py` (move it out of `MOVED`/`DIVERGED`,
   re-record `track_i.diff` for the other diverged modules if needed).
2. Adder: S06/P02 (P01 optional) as a `PrefillAdder` subclass via an AFTER
   hook on `Scheduler.get_prefill_adder_class`; keep production's
   `ignore_eos` reuse and budget charge (design 3.1.2–3.1.3).
3. Release: M03 decomposition (design 3.1.6) with the mamba slot freed
   exactly once; fallback one rebased REPLACE. M01: rollback as a small
   AROUND hook, no copy.
4. Rebase REPLACE rows B02–B05, S04, S07 (three-way merge, `RESOLVED` for
   hand merges: `checkpoint=` keyword, `mamba_lazy_checkpoint`,
   `reqs_to_exclude`, `extend_end`/`prefix_len`). B03's #37771
   `request_finished` gate: follow production's semantics for QSA leases.
   M04: drop if #43439 made it redundant.
5. Tests (`tests/lifecycle/`, `tests/prefix/`): port every existing test to
   the new API without loosening it; add the design's section 5 checks
   (factory-time runtime exists, hit path incl. `prefix_len`/
   `cache_protected_len`/row contents, restore once, release-order ledger,
   mamba freed once on both `checkpoint` paths, `ignore_eos` with a pending
   hit, chunk ends on page-64 boundaries incl. inside images, TP convergence
   points) and a test that the request's reported `cached_tokens` equals
   production's for a host hit, a miss and a chunked hit.
