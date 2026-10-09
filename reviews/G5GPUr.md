The latency resolution still has incorrect figures in [G2-3](/docs/phase5-results.md:90) and the [response](/reviews/G5GPU-response.md:8):

- **65,536 warm median:** **0.594% slower** (rounds to 0.6%), rather than 1.0%. [F](qwen:results/plugin-g5-20261009/F/g23/latency.json:197), [P](qwen:results/plugin-g5-20261009/P/g23/latency.json:197).
- **262,016 warm median:** **0.338% faster** (rounds to 0.3%), rather than 0.2%. [F](qwen:results/plugin-g5-20261009/F/g23/latency.json:289), [P](qwen:results/plugin-g5-20261009/P/g23/latency.json:289).
- **Literal cold bound “at most 1.0%”:** the second 8,192-token cold request is **1.015% slower**, slightly exceeding that bound. “About 1.0%” would fit. [F](qwen:results/plugin-g5-20261009/F/g23/latency.json:63), [P](qwen:results/plugin-g5-20261009/P/g23/latency.json:63).

The memory correction is supported. Cutover’s profile, ready status, health, model ID and GPU figures match [the snapshot](qwen:results/plugin-g5-20261009/cutover-after.txt:1); it does not record the restart/readiness times, activation count or request measurements.

G5-GPU: not cleared