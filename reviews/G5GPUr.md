The latency resolution still has incorrect figures in [G2-3](/home/zyk/projects/interests/ai-video/qwen/qsa-hisparse-plugin/.claude/worktrees/phase5/docs/phase5-results.md:90) and the [response](/home/zyk/projects/interests/ai-video/qwen/qsa-hisparse-plugin/.claude/worktrees/phase5/reviews/G5GPU-response.md:8):

- **65,536 warm median:** **0.594% slower** (rounds to 0.6%), rather than 1.0%. [F](/home/zyk/projects/interests/ai-video/qwen/results/plugin-g5-20261009/F/g23/latency.json:197), [P](/home/zyk/projects/interests/ai-video/qwen/results/plugin-g5-20261009/P/g23/latency.json:197).
- **262,016 warm median:** **0.338% faster** (rounds to 0.3%), rather than 0.2%. [F](/home/zyk/projects/interests/ai-video/qwen/results/plugin-g5-20261009/F/g23/latency.json:289), [P](/home/zyk/projects/interests/ai-video/qwen/results/plugin-g5-20261009/P/g23/latency.json:289).
- **Literal cold bound “at most 1.0%”:** the second 8,192-token cold request is **1.015% slower**, slightly exceeding that bound. “About 1.0%” would fit. [F](/home/zyk/projects/interests/ai-video/qwen/results/plugin-g5-20261009/F/g23/latency.json:63), [P](/home/zyk/projects/interests/ai-video/qwen/results/plugin-g5-20261009/P/g23/latency.json:63).

The memory correction is supported. Cutover’s profile, ready status, health, model ID and GPU figures match [the snapshot](/home/zyk/projects/interests/ai-video/qwen/results/plugin-g5-20261009/cutover-after.txt:1); it does not record the restart/readiness times, activation count or request measurements.

G5-GPU: not cleared