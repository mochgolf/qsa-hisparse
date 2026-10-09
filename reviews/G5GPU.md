Two reporting claims need correction:

1. **Latency bound is overstated** ([claim](/home/zyk/projects/interests/ai-video/qwen/qsa-hisparse-plugin/.claude/worktrees/phase5/docs/phase5-results.md:90)). The 8,192-token warm median is **0.125879555 s F / 0.128450598 s P**, a **2.04%** increase. The second warm sample increases **3.83%** (0.126954823 / 0.131817589 s). Both occur after the first request, contradicting “No regression beyond about 1%.” The reported medians are correct. Evidence: [F samples](/home/zyk/projects/interests/ai-video/qwen/results/plugin-g5-20261009/F/g23/latency.json:83), [P samples](/home/zyk/projects/interests/ai-video/qwen/results/plugin-g5-20261009/P/g23/latency.json:83).

2. **The blanket memory bound is incorrect** ([claim](/home/zyk/projects/interests/ai-video/qwen/qsa-hisparse-plugin/.claude/worktrees/phase5/docs/phase5-results.md:88)). Compat’s initial GPU-0 usage is **43,334 MiB F / 43,338 MiB P**, a **4 MiB** difference, exceeding “no other memory figure differs beyond 2 MB.” Evidence: [F snapshot](/home/zyk/projects/interests/ai-video/qwen/results/plugin-g5-20261009/F/compat/compat-cold.json:9), [P snapshot](/home/zyk/projects/interests/ai-video/qwen/results/plugin-g5-20261009/P/compat/compat-cold.json:9).

The functional comparisons, source provenance, and remaining claims are supported. The prepared profile differs only as stated and resolves to clean release commit `6a1c401`.

G5-GPU: not cleared