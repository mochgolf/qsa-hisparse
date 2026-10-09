# G5-GPU response

Review: `reviews/G5GPU.md` (two reporting corrections; functional
comparisons, provenance and the cutover profile confirmed).

| # | Resolution |
| --- | --- |
| 1 latency bound overstated | Corrected in `docs/phase5-results.md`: after the first request cold requests differ by at most 1.0%; 8192 warm (about 0.13 s) is 2.0% slower at the median (+2.6 ms) and 3.8% in the second sample; 65536 warm 1.0% slower; 262016 warm 0.2% faster; two samples per case. |
| 2 memory bound incorrect | Corrected: GPU usage snapshots differ by a few MiB (compat's initial GPU-0 usage 43,334 / 43,338 MiB); weight memory 2 MB as stated. |

Both were wording errors in the results document; no evidence or code changed.
