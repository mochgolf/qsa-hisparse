You are an independent reviewer for gate G3-I, CPU part (Track I: exact image
host-prefix reuse). Work read-only. Do not modify files. Be proportionate
(owner's instruction; PLAN.md rule 10).

Read `docs/GOAL.md` (image obligations), `docs/PLAN.md` ("Track I",
"Track I acceptance"), `docs/prefix-cache.md` §6, `docs/tasks/I-A.md`,
`docs/tasks/I-B.md`, `docs/tasks/I-C.md`, `docs/STATUS.md` (Track I entries).
Code: `src/sglang_qsa_hisparse/hisparse/{image_identity,image_request,prefix,prefix_cache,runtime}.py`,
`src/sglang_qsa_hisparse/patches/hisparse/image_identity.py`; tests:
`tests/image/`, `tests/image_boundaries/`, `tests/runtime/test_moved_sources.py`,
`tests/runtime/track_i.diff`. Pinned SGLang: `../.worktrees/sglang-pin-76e06febab`.
Merged into main at `8b92737`.

Check, against the seven acceptance criteria:
1. Correctness of the identity (artifact key transport, records intersecting
   [0, L), straddling image, order/offsets/grid, M-RoPE page digest) and of
   matching (acquire, ancestor lookup, TP signatures); any way two requests
   with different image-dependent KV/recurrent/PLE state at L can share a
   snapshot, or one with identical state is needlessly refused.
2. The I-B claim that the pinned SGLang already computes suffix embeddings,
   M-RoPE positions and PLE history correctly at in-image boundaries: verify
   in the pinned code paths cited.
3. Text-only behavior unchanged (Phase 2 parity must still hold).
4. Test adequacy: does each criterion have a test that would fail on a
   plausible defect?
5. Whether the I-C GPU plan (docs/tasks/I-C.md) would establish the GPU part
   of the criteria, including the ViT batch-invariance risk I-B raised.

Output: findings by severity (at most ten) with file:line, failure scenario
and smallest fix; final line "G3-I (CPU): cleared" or "G3-I (CPU): not cleared".
