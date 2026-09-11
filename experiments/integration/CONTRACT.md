# QSA–HiSparse integration spike contract

Frozen before running the candidate prototype.

## Authority and fixed environment

- Acceptance authority: `ANALYSIS-PLAN.md` at lab commit `e576563`.
- Candidate source base: SGLang `d0664112a8c81dd8019689bc2dd5e5b8fc4f851a`, isolated branch `codex/qsa-hisparse-spike-20260908`; upstream `afe90a8b` is observed but not merged.
- Independent references: the source QSA pool/backend contracts, `qsa_sparse_attention_reference`, and the previously reviewed execution-02 trace and resident bytes. Candidate cache state never supplies its own expected bytes, scale, selection, or output.
- Runtime: Python from the test environment, two RTX 4090 48 GiB, TP2/no-spec, BF16 Q, FP8 E4M3 K/V, FP32 GDN, page 64, C4, top 512 C4 (=2048 raw tokens), 12 QSA layers.

## Source interface and layouts

The existing `QSATokenToKVPool` owns two distinct contracts. Index-K is BF16 and compressed: its complete-group location is `full_slot // 4`, page size is `64 // 4 = 16` C4 records, and pending uncompressed index-K is a per-request ring of four `[1,128]` BF16 rows. Attention K/V remains the parent pool's separated tensors, per layer and TP rank:

- K: `[raw_slots, 1, 256]`, FP8 E4M3, contiguous head dimension.
- V: `[raw_slots, 1, 256]`, same dtype and layout.
- One raw token K+V: `1 * 256 * 1B * 2 = 512B`.
- Host adapter C4: one contiguous 2048B record `[K0,K1,K2,K3,V0,V1,V2,V3]`; unpacked hot views are `[4,1,256]` K followed by `[4,1,256]` V. One page is 16 C4 records.
- Scale: one non-unit scalar K descale and V descale per layer in the current backend; the adapter preserves them exactly. It does not invent per-record scale semantics.
- Selection is logical C4 IDs `[rows,512]`. Expansion is `4*block + {0,1,2,3}`, then clipped/masked by the row's logical raw-token length. The resulting raw logical IDs map through the request's `req_to_token` row to separated physical K/V slots. The host/hot adapter instead resolves C4 IDs to packed hot slots, then exposes the equivalent separated K/V rows to the same attention call.
- A tail of 0/1/2/3 raw tokens is device-resident and not host-fetchable as a complete C4. The newest complete C4 remains protected until its D2H completion event permits eviction.

Generic HiSparse resolve/LRU and pinned-host copies can be reused. A QSA adapter is required for separated K/V packing/unpacking, per-layer scales, logical-C4 to four raw-token expansion, the QSA pending ring, completion-event ownership, and request generation/slot identity. DSV4-specific pool types or compressed-slot arithmetic are not substituted for these contracts.

## Frozen test inputs and assertions

V0 uses CPU layout/address tests for C4 remainders 0/1/2/3, page boundary IDs 15/16, valid tails, and invalid physical mapping. Shapes, dtypes, strides, and byte counts above are exact assertions.

V1 reuses real execution-02 A/B selections at first/middle/last positions on 12 layers and both ranks. It fixes identical Q, selection order, K/V bytes, non-unit scales, and valid masks for resident and host→hot arms. Packed round-trip must be byte-identical. Both arms must call the same `qsa_sparse_attention` entry with identical operation order and produce bit-identical output. An independent FP32 reference must meet the pre-frozen `max_abs <= 0.03` and `max_rel <= 0.03` bound: FP8 E4M3 input quantization contributes at most 6.25% relative per element before softmax, but the tested finite payload is deliberately bounded and the tighter empirical-independent bound is checked against FP32 values generated before FP8 casting. A mutated mapping or scale must be rejected. CUDA graph capture/replay time and allocator workspace are recorded; failure blocks dependent performance claims.

V2 runs append→C4 close→asynchronous D2H→event-gated eviction→H2D refetch→V1 attention consumption. It forces hot exhaustion, release while a copy is pending, slot generation reuse, and two requests. Expected host bytes are built independently from append inputs. Every selected C4 must be resident before consumption; mappings are unique; request generation tags prevent cross-request reads. Early slot reuse and omitted-event mutations must be rejected. No global per-step synchronize is allowed; explicit event waits are timed and reported.

V3 sends one real A-long request with the existing chunk size 2048 and samples NVML frequently across cold prefill and first decode. Endpoint memory is not called a peak. Because the current allocator retains the full GPU pool and has no QSA host/hot handoff adapter, actual GPU-pool release is expected to be `BLOCKED/SERVICE_DEPENDENCY` unless the isolated prototype exposes a bounded existing hook. Process exit or deleting a temporary tensor never counts as release. No chunk scan or shortened input is permitted.

V4 runs component-only independent requests at B=1/4/8, each with its own host/hot pool, logical map, generation state, and staggered A/B real-trace rows; common-prefix sharing is forbidden. Logical capacities are 262144/1048576/2097152 raw tokens. Both GPUs execute 12 layers serially. Each B is replayed three times and records per-request and batch latency, slower-rank P50/P95/P99, D2H/H2D bytes, CUDA allocator peak, pinned bytes, and host NUMA observation. V2 writeback remains enabled. B=1 must stay within the A-long no-observer TPOT 10% budget of 2.427575 ms for the declared component interval. B=4/8 are scaling evidence only and cannot qualify service concurrency.

## Failure and reporting semantics

V1/V2 correctness failure stops dependent performance conclusions. Failed attempts remain immutable; fixes use a new attempt and source/script commit. `PASS`, `FAIL`, `BLOCKED/SERVICE_DEPENDENCY`, and `BLOCKED/RESTORE` retain their plan meanings. Tolerances and service acceptance are not changed after observing results. Highest possible conclusion is evidence for an integration design; no 8-request service GO is available in this spike.
