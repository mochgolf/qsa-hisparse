Three branches are ready to offer to the owner; five need changes. I found no demonstrated production-code defect in the three ready candidates.

This review was read-only. AST and whitespace checks passed for all branches; available Ruff checks passed for U6/U8. Eight isolated U8 alignment cases passed against the committed Python oracle. Full test reruns were **blocked during imports** by temporary-directory and FlashInfer log writes prohibited by this sandbox; the recorded suite results were therefore not independently reproduced. No GPU validation ran.

Source locations below refer to the named branch’s Git tree.

1. **`qsa/U1-hisparse-coordinator-gating` — ready after owner confirmation.**

   `python/sglang/srt/managers/scheduler.py:5281`: including staging requests in `collect_inflight_reqs` fixes the missed-abort window and preserves weight-version tracking through the existing caller. Coordinator gating is contained and consistent with runner behavior. The abort regression exercises the actual scheduler path; the description accurately states that cancellation still permits the usual cleanup decode.

   Small test improvement: at `test_scheduler_chunked_req_gate.py:178`, explicitly set `enable_hisparse=False` in both new gate tests. Their committed pre-fix failures otherwise include missing-stub attributes; the documented scratch run supplies the stronger behavioral evidence.

   A separate small PR is reasonable despite [#35488](https://github.com/sgl-project/sglang/pull/35488). **No GPU validation is essential before offering this scheduler fix.** A staging-abort smoke is useful integration evidence.

2. **`qsa/U9-monotonic-req-generation` — ready after owner confirmation.**

   `python/sglang/srt/mem_cache/memory_pool.py:458` and `disaggregation/decode.py:254`: retaining generations across clear is correct and minimal. The CPU regression covers both pools and distinguishes the previous reset behavior. The DSpark explanation is supported by the surviving relay/carry rings and generation-equality check.

   The separate PR decision is sound. [#36093](https://github.com/sgl-project/sglang/pull/36093) changes when tracking occurs, without fixing resets. **No GPU validation is required.**

3. **`qsa/U9-hisparse-decode-mm-inputs` — needs changes, limited to test style.**

   `python/sglang/srt/managers/scheduler.py:3609`: the assignment correctly restores the per-request list bypassed by the staging batch builder. The CPU test guards meaningful field bookkeeping, and the M-RoPE failure described in the document follows from the actual consumer.

   `test/registered/unit/managers/test_schedule_batch_req_pool_indices.py:75`: the new class inherits `unittest.TestCase`, contrary to the stated upstream `CustomTestCase` rule. **Smallest fix:** import `CustomTestCase` and use it for the new class.

   The independent PR decision is sound. **No GPU prerequisite is needed for this metadata repair.** An image/text M-RoPE decode smoke would strengthen integration evidence.

4. **`qsa/U7-gptq-moe-w13-scale-k` — ready after owner confirmation as a contribution to #35955.**

   `python/sglang/srt/hardware_backend/gpu/quantization/gptq_kernels.py:286`: deriving K from the w13 scale table correctly selects grouped permutation at the expert-partition boundary. The test uses real scheme processing, stubs only CUDA repacking, and compares against the unchanged permutation helper with independently specified dimensions.

   Contributing the incremental commit to [#35955](https://github.com/sgl-project/sglang/pull/35955) is preferable. **Do not open this current stack as an independent PR against main:** its parent changes belong to that existing PR.

   CPU evidence is sufficient to offer the contribution transparently. Before claiming GPU accuracy in a standalone follow-up, run one affected-boundary Marlin comparison against dequantized weights and the upstream-required model accuracy evaluation. Speed benchmarking is unnecessary for this correctness fix.

5. **`qsa/U7-autoround-moe-marlin-group-split` — needs changes.**

   `python/sglang/srt/layers/quantization/gptq/gptq.py:620`: repeating group rows before TP sharding is mathematically sound. The selection test and real w2 loader test provide useful CPU evidence, including the split group on rank 1. The type guard improves on the fork’s broader retry.

   [U7.md:69](/docs/upstream/U7.md:69) and `:239`: the examples conflate full expert widths with per-rank widths. The listed 640/768/1536 values are divisible by 128. **Smallest fix:** explicitly distinguish full widths 640/768/1536 from partitions 320/192/192 at TP2/TP4/TP8.

   Before opening, validate the changed backend’s outputs against the existing MoeWNA16 path or dequantized reference on an affected shape, run the required model accuracy evaluation, and measure representative prefill/decode performance. These are material for a backend-selection change. Contribute first or rebase after the parent fixes merge.

6. **`qsa/U6-qsa-sm8x-varlen-fallback` — needs changes.**

   The resolver change at `python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py:114` is small and plausible. A new focused PR is reasonable relative to [#36968](https://github.com/sgl-project/sglang/pull/36968).

   `test/registered/kernels/ops/attention/qsa/test_qsa.py:173`: the new tests run on CPU but belong to a file registered only for B200 GPU CI at `:39`. **Smallest fix:** move the resolver regression into a CPU-registered unit file under `unit/layers/attention/`.

   [U6.md:217](/docs/upstream/U6.md:217): “Served on 2x RTX 4090” must identify the historical fork validation; this branch has not been GPU-tested.

   **Required before opening:** one real SM89 head-dimension-256 decode/reference smoke without classic FA2, confirming the selected implementation. SM86 can remain explicitly untested if unavailable; a multi-architecture model campaign is unnecessary.

7. **`qsa/U6-qsa-fp8-kv-scales` — needs changes.**

   The backend-only approach is coherent: scaled stores protect caller tensors, cached reads apply scales, and the fused unscaled path is excluded. It reasonably carries forward [#36644](https://github.com/sgl-project/sglang/pull/36644) and composes algebraically with [#41933](https://github.com/sgl-project/sglang/pull/41933). The CPU decode and CP expectations are useful.

   `test/registered/kernels/ops/attention/qsa/test_qsa.py:101`: move the CPU decode regression into a CPU-registered unit file. CP coverage already has CPU registration.

   `qwen_sparse_attn_backend.py:1432`: the fused-path rejection and GPU read-path scale propagation remain unvalidated. **Smallest evidence addition:** a non-unit-scale GPU regression covering cached-prefix prefill and decode, with distinct K/V scales and a dequantized-cache reference. Confirm fused decode is bypassed and BF16/unit-scale behavior remains unchanged.

   One available fallback architecture is enough before opening; TRTLLM/HIP validation can be assigned to upstream runners. The draft must retain explicit untested-path limitations.

8. **`qsa/U8-marlin-moe-batch-invariant` — needs changes.**

   The whole-K ownership argument, fixed configuration, and stable alignment are coherent. CPU alignment evidence does not validate CUDA compilation or the new reduction’s outputs.

   `test/registered/kernels/ops/moe/test_marlin_moe.py:382`: the new test compares candidate outputs across batch sizes. It establishes invariance, but an consistently incorrect result would also pass. Existing numerical tests run outside the newly enabled mode. **Smallest fix:** retain the dequantized reference weights and add a numerical-reference assertion while deterministic mode is enabled.

   Before opening, compile and run the new variant, demonstrate the invariance regression fails before the fix, and check numerical accuracy for representative BF16/FP16 cases. Include one graph-replay case and representative performance measurement. SM89 plus coverage of the SM90+ atomic-path difference is proportionate; the entire fork matrix and exact fork comparison belong primarily to later plugin-removal qualification.

The dropped/deferred decisions need these qualifications:

- **U4:** deferral remains reasonable, but [U4.md:29](/docs/upstream/U4.md:29) incorrectly treats [#35594](https://github.com/sgl-project/sglang/pull/35594) as open. It is merged in `b7b2975b57`. Reassess K02 using the existing pool-class selection and `full_kv_pool_class` seam before asserting a new prebuilt-pool factory is necessary.
- **U5:** deferring graph hooks and bundling the decode CPU mirror with its consumer is sound. [U5.md:27](/docs/upstream/U5.md:27) should distinguish U1’s gating change from #35488’s protocol: U1 does not introduce that protocol.
- **U23:** deferral around [#42923](https://github.com/sgl-project/sglang/pull/42923) and [#43023](https://github.com/sgl-project/sglang/pull/43023) is sound. However, [U23.md:184](/docs/upstream/U23.md:184) cannot treat `on_release` as the frozen post-row-free point: it runs before `req_to_token_pool.free`. Preserve an after-release function hook or supply an explicit equivalence argument, including failures, before claiming M03 disappears through cache overrides alone.
- **INT8-row PLE:** deferral is sound given [#41624](https://github.com/sgl-project/sglang/pull/41624), format coordination, and absent public checkpoint/evidence. That PR supplies related plumbing; it does not already support `int8_row`.
- **Stable HC:** avoiding a second kernel is reasonable for the target SM89 compiled path. [U8.md:83](/docs/upstream/U8.md:83) should scope the conclusion: main’s SM100 low-M CuTe HC path bypasses the compiled chain. Crossing its 24-row dispatch boundary remains a separate validation question.
- **Stable top-k:** avoiding duplication of [#42087](https://github.com/sgl-project/sglang/pull/42087) is sound. Its CUDA contract matches the fork’s stated ordering/tie rules. SM89 graph-replay equivalence and acceptance of changed prefill numerics remain prerequisites for removing plugin rows.
- **Shared reasoning correction:** “no in-tree consumer therefore fails unit-test admission” overstates the rules in U4/U5/U6/U23. Protocol properties and critical bookkeeping are admitted categories. Describe those deferrals as scope, evidence, and maintenance judgments.

All publication bodies still need the complete upstream template and required pre-commit results; partial lint checks do not establish that requirement.

G3-U: not cleared