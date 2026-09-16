# P0 component attribution — main acceptance

Candidate source `3d155d30c13a7ce4024a8f8a7482696f7d4a0dfb`.

Verdict: `PASS_P0_COMPONENT_ATTRIBUTION`. Together with the accepted P0 E2E report (`eb73ee1`), this closes P0 and permits measured P1 changes. Original execution-01 remains rc=1 / `FAIL_RESTORED`: the CPU/GPU annotation inventory double count was corrected offline at `cb64137`, and capture/recheck evidence was archived at `ea6db5b`. No additional GPU run was needed. The independent parser review and main CPU self-check pass.

## Findings

| Diagnostic quantity | rank 0 | rank 1 |
|---|---:|---:|
| First handoff CPU wall | 1357.309 ms | 1356.592 ms |
| allocate_validate CPU | 1111.116 ms | 1110.302 ms |
| gather_pack CPU | 76.487 ms | 76.185 ms |
| D2H submit CPU | 40.322 ms | 39.965 ms |
| wait_copy CPU | 51.247 ms | 51.516 ms |
| D2H actual memcpy union | 61.499 ms | 61.490 ms |
| Steady named QSA CPU union, resident | 9.321 ms/step | 9.318 ms/step |
| Steady named QSA CPU union, offload | 15.752 ms/step | 15.711 ms/step |
| Steady QSA-associated actual GPU union, resident | 4.295 ms/step | 4.344 ms/step |
| Steady QSA-associated actual GPU union, offload | 4.810 ms/step | 4.706 ms/step |

Rank 0 `cudaHostAlloc` accounts for 1092.540 ms inside allocation/validation. The first P1 change therefore reserves the existing pinned slab during adapter startup and reuses it across requests. This moves cost before readiness; it does not eliminate allocation or free the reserved slab at request release.

Steady `qsa.selected` has about 5.80 ms CPU inclusive and 1.13 ms exclusive per step. Compact mapping (~1.85 ms), resolve/refetch (~1.52 ms), hot gather (~0.84 ms), and indexed unpack (~0.47 ms) are the subsequent candidates. Indexer and FA2 CPU times are approximately unchanged. No new kernel is justified by these data alone.

## Interpretation limits

The four traces cover 64 decode steps; steady attribution excludes the first and uses steps 2–64. CPU annotations count only complete `user_annotation` events; GPU projections supply fallback association, never GPU busy duration. Actual kernel/copy/memset intervals are unioned; residuals are explicit interval differences and accounting errors are zero. Missing External-id device/runtime events are included as documented in `RESULT-RECHECK-REVIEW.md`. Nested inclusive durations, CPU wall, GPU work, and TP ranks must not be summed together. The slow-rank aggregate is a proxy, not an exact synchronized TP critical path.

Profiler wall (~129 ms resident / ~136 ms offload) is not service TPOT. The unprofiled P0 medians remain 93.820/97.036 ms; paired steady overhead is +3.216 ms median. Profile attribution identifies where to investigate, not the amount of unprofiled speedup to promise. Full device-union changes have opposing signs across ranks and cannot support a GPU speedup claim.

The restore evidence passed the full pre-test service identity checks. The baseline source remains separate from the candidate. This is B1 attribution, with no multi-request, B8, or deployment-performance qualification.
