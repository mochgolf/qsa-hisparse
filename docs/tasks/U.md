# Track U: upstream PR preparation (local only)

Goal: prepare upstream SGLang changes that let the plugin drop REPLACE
patches, aligned with upstream's current direction. Nothing is pushed,
opened or commented on; the owner confirms each publication separately.

Read: `docs/upstream-status.md`, PLAN.md "Track U", `docs/patch-inventory.md`
(the rows each task would remove), and upstream's contribution rules
(section 4 of upstream-status.md: no defensive getattr probing, msgspec
containers, model_runner.py orchestration only, tests under
`test/registered/unit/<module>` with `register_cpu_ci`).

For each U item in your group:
1. Re-check the related open upstream PRs (read-only `gh pr view`) and the
   current upstream code in your worktree. Decide: contribute to an
   existing PR (prepare a patch on top of it), new PR, or drop (already
   upstream / not worth it). Prefer the smallest change upstream would accept.
2. For items you pursue: a local branch `qsa/<item>-<slug>` in your upstream
   worktree with the change and upstream-style tests; run the tests that
   can run on CPU with the validation interpreter (do not install anything;
   report tests that cannot run in this environment).
3. Write `docs/upstream/<item>.md` in the plugin repository: decision and
   reasoning, related PRs, branch and commits, test results, the plugin
   rows it would remove, and a draft PR description (title, summary, test
   plan) for owner review.

Groups:
- UA (small fixes): U9 (monotonic `req_generation` across flush; HiSparse
  decode batch `multimodal_inputs`), U7 (GPTQ MoE scale sizing/dtype incl. w13
  `size_k`, aligned with open #35955; AutoRound g128→g64 TP split; INT8-row
  PLE).
- UB (scheduler and cache lifecycle): U1 (HiSparse coordinator protocol and
  scheduler gating, aligned with open #35488) and U23 (prefix-cache
  lifecycle hooks on the reworked API plus allocator free-group/release
  callbacks and KV pool runtime attachment); freeze the lifecycle contract
  (checkpoint, drain, logical release, free-group flush, physical reuse) in
  the doc before code. U23 rebases onto U1 if both are pursued.
- UC (backend, pools, graphs, determinism): U4 (pool fixed reservation and
  `full_kv_pool` factory), U5 (decode CUDA graph backend lifecycle hooks;
  widen `ForwardBatch.req_pool_indices_cpu`), U6 (QSA backend extension
  points, FP8 descales, SM86/SM89 flash-attention fallback,
  `decode_score_width`), U8 (deterministic Marlin whole-K, QSA stable top-k
  aligned with open #42087, stable HC).
