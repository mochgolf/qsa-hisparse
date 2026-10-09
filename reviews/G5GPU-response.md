# G5-GPU response

Review: `reviews/G5GPU.md` (two reporting corrections; functional
comparisons, provenance and the cutover profile confirmed).

| # | Resolution |
| --- | --- |
| 1 latency bound overstated | Corrected in `docs/phase5-results.md` (G2-3 row): every per-sample and median difference is now listed, computed from `latency.json` (e.g. 8192 warm +2.04% median, +3.83% second sample; 65536 warm +0.59%; 262016 warm −0.34%; 8192 second cold +1.02%), without a summary bound. |
| 2 memory bound incorrect | Corrected: GPU usage snapshots differ by a few MiB (compat's initial GPU-0 usage 43,334 / 43,338 MiB); weight memory 2 MB as stated. |

Both were wording errors in the results document; no evidence or code changed.

**Re-check (`reviews/G5GPUr.md`):** the remaining rounding errors are fixed by listing exact figures; the cutover facts now cite `cutover-after.txt` and `cutover-evidence.txt` (start/readiness/launcher log lines and the smoke results).
