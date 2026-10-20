One explicit G2-3 criterion is unmet. The remaining required gates and the stated arm/interpreter/commit provenance are supported.

1. **Medium — G2-3 is incorrectly marked fully passing.**  
   Location: [results table](/home/zyk/projects/interests/ai-video/qwen/qsa-hisparse-plugin/docs/phase6-results.md:136). [Baseline step 16](/home/zyk/projects/interests/ai-video/qwen/qsa-hisparse-plugin/docs/baseline.md:306) requires cold `prompt_tokens_details` to be `null`. F returns `null`; [P returns `{"cached_tokens": 0}`](/home/zyk/projects/interests/ai-video/qwen/results/plugin-g6-20261019/P/g23/smoke.json:47). The [checker](/home/zyk/projects/interests/ai-video/qwen/qsa-hisparse-plugin/tools/evidence/run_g23.py:171) normalizes both to zero and misses this difference. D8 does not cover it.  
   **Smallest fix:** report this unmet criterion and obtain an explicit owner acceptance of the upstream response change, or preserve the required response; make the checker enforce the resulting criterion.

2. **Medium — exclusive upstream latency attribution exceeds the evidence.**  
   Location: [diagnosis conclusion](/home/zyk/projects/interests/ai-video/qwen/qsa-hisparse-plugin/docs/phase6-results.md:184). The TP0 traces support comparable cache scheduling/restore costs and a roughly 12 ms longer model forward. Upstream stage scaffolding and `/generate` validation are credible contributors. However, the scheduler traces do not measure the HTTP contract check, its reported CPU timings include cached JSON parsing, and profiled forward durations cannot directly explain unprofiled latency. F/P also change dependencies and plugin implementation together. A smaller plugin contribution remains plausible; “nothing to fix in the plugin” is not established.  
   **Smallest fix:** describe upstream overhead as the likely explanation and limit the plugin conclusion to “no substantial regression observed in the sampled cache path/hooks.” No additional GPU qualification is required by the latency criterion.

3. **Low — the results table double-counts the SM121 skip.**  
   Location: [G2-1 row](/home/zyk/projects/interests/ai-video/qwen/qsa-hisparse-plugin/docs/phase6-results.md:129). `F/g21/step8.log` records **91 passed, 1 skipped**; P records **95 passed, 1 skipped**, including four plugin-only tests. The comparator counts the skip’s ID and reason entries separately in its printed total. Its parity verdict remains valid.  
   **Smallest fix:** change “2 skips” to “1 skip” and count only reason entries in the comparator’s summary.

4. **Low — profiling statistics mix sample selections.**  
   Location: [profiling summary](/home/zyk/projects/interests/ai-video/qwen/qsa-hisparse-plugin/docs/phase6-results.md:156). From `profile/{F,P}/timings.json`, pre-profile medians are **0.200248 / 0.221832 s**; post-profile medians are **0.218331 / 0.240226 s**. All ten unprofiled samples give **0.209457 / 0.231023 s**. Also, 142.8/140.6 ms scheduling spans are from the **first** profiled request, while 106.7/105.1 ms restore figures average both requests, despite “second … unless noted.”  
   **Smallest fix:** label the sample groups and aggregations accurately and correct the numbers.

The two stated irregularity explanations are sound: `descale-plugin` launched CPU-tensor interpreter tests without `TRITON_INTERPRET`; existing GPU descale tests passed. `steps.json` is runner metadata, and all recorded G2-1 steps exited zero. Both profiling commands separately exited one during final summary printing, **after** saving timings and stopping their servers; those footer errors do not invalidate the captures.

Production restoration is supported both times: all four before/after records retain profile SHA256 `99752c69771a91fd342ff401891a208052ee41a36792bfc0eaece755882bdf16`; both after-records show ready, health 200, and the same model ID.

G6-GPU: not cleared