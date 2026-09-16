# QSA HiSparse prefix reuse and service control

Date: 2026-09-16. Status: implementation and target qualification complete.
See [the final validation record](PREFIX_CACHE_VALIDATION.md) for results and scope.

## Objective and ownership

Preserve the current TP2, FP8 E4M3, C4/page64, eight-request, 262,144-token
per-request QSA HiSparse offload service. Add reusable host-resident prefixes
and one-command start, stop, restart, status, and log access. Automatic GPU
sleep/unload is outside this task.

Prefix storage/runtime/scheduler integration and service management have
separate focused tests. Integration review and live qualification use
independent evidence and exclusive GPU scheduling so experiments cannot collide.

## Implementation contract

1. A reusable prefix has an owner independent of an active request lease.
   Request-slot reuse must never overwrite published cached state. Publishing
   requires completed writes on both TP ranks; eviction requires no readers
   or pending transfers. Cache identity includes tokens and applicable model,
   position, adapter, and multimodal distinctions.
2. Cache hits resume only at page-aligned boundaries with all required state:
   raw K/V bytes, compressed QSA index state, recurrent/Mamba state, PLE history
   and convolution state, and position metadata. Recompute an unmatched tail;
   do not approximate missing state or share mutable continuation state.
3. First implement bounded CPU snapshots and restoration into the existing
   single-context prefill staging arena and active request backing. Preserve
   bounded GPU hot caches and the accepted sparse decode math. A separate
   immutable shared-page optimization is deferred unless needed for correctness.
   Cached prefixes and their checkpoints must have explicit host and GPU
   budgets; historical prefixes must not consume the eight active leases.
4. Restore cache data before suffix computation; restore all necessary layer
   states at the same logical boundary. Preserve FP8 bytes and scales, C4 tail
   semantics, page mapping, TP agreement, and CUDA graph pointer/lifetime rules.
   Prefix caching must skip real model work and expose actual reused tokens.
5. Fail closed for unsupported combinations. Preserve the no-cache path for
   comparison and rollback. Avoid enabling upstream radix sharing without the
   matching host ownership integration. New mathematical or tolerance choices
   require explicit analysis before implementation, not an acceptance relaxation.
6. Service management preserves the current model, API name/address, quantized
   PLE mode, execution flags, and environment. Starts are idempotent; conflicting
   listeners and stale PID records cannot authorize killing unrelated processes.
   Stop covers worker processes; start waits for readiness and surfaces failures.
   Provide readable logs, a local profile, and a rollback profile. No automatic
   restart loop, memory release policy, model download, or external publication.

## Acceptance contract, frozen before candidate measurements

- Establish the existing CPU-suite baseline and save the original production
  command/environment and service health before changing live state.
- Use independent host/tensor reference construction for exact byte/state
  restoration tests. Include counterexamples for stale lease reuse, branch
  mutation, partial pages/C4 groups, incomplete publication, and eviction while
  referenced. Exercise misses, hits, eviction, cancellation, and concurrency.
- Live cold and warm runs use fixed token inputs and greedy generation. For
  the preselected fixtures require identical generated token IDs across repeat
  cold runs, warm reuse, and divergent suffixes versus their uncached controls.
  Cached bytes and checkpoint restoration require exact equality. Do not tune
  tolerances or select passing fixtures using candidate failures.
- Demonstrate nonzero reused-token counts and reduced actually computed
  prefill tokens. Measure first-token/prefill latency rather than promise a
  speedup from architecture alone. New-token attention and PCIe transfer remain.
- Cover repeated shared prefixes, branching continuations, page/chunk boundary
  lengths, B2/B8 concurrency, cache budget pressure, abort, flush, and request-slot
  recycling. Retain TP2/B8/256K configuration and qualify available long-context
  cases with explicit reporting of any untested scope.
- Compare GPU allocated/reserved memory and device usage before/after cache
  population at the same serving configuration. The host budget must bound
  cache growth; old prefixes must not create growing GPU-resident raw KV.
- Exercise service start/stop/restart/status and failure reporting with CPU
  fixtures before live operations. Finish with a healthy qualified service or
  the original rollback service if qualification fails, and report that status.

## Current source evidence

The source baseline is `e6e90a9c6663f378d031a29d20211079fedd984d`.
`qsa_hisparse/config.py` currently requires `disable_radix_cache=True`.
`runtime.py` binds host slabs to active leases and clears their request
ownership on release. `slots.py` serializes one position-addressed staging
arena. The current prefix-aware prefill attention path gathers full-context K/V
from that arena; a cache hit therefore needs restoration, not only token-index
reuse. Mamba prefix matching also requires a usable state checkpoint.

The original rollback source is preserved in a separate private checkout.
Machine-specific profiles and measurements stay outside this repository;
do not publish model paths, environment details or raw logs.

## Baseline observation before candidate GPU execution

The unchanged original production service produced different greedy token IDs
on two identical uncached `prefix-64-copy` requests: the first divergence was
at output token 27, and returned log probabilities differed earlier. The frozen
fixture and both complete responses are retained in `original-cold.json`.
This establishes that native/light execution cannot be used as an exact
repeatability oracle, independently of the new cache implementation.

Exact cold/warm token comparisons will therefore use the same deterministic
math configuration on both arms (`--enable-deterministic-inference`), with
the original frozen inputs. Native/light serving performance and smoke tests
remain a separate deployment check. Exact cache-byte/state comparisons remain
mandatory. No candidate-derived numerical tolerance or fixture filtering is
introduced by this clarification.

The clean fork also failed cold repeatability with deterministic inference
on the same first frozen fixture (eight further cold trials produced three
different reasoning continuations). This is retained as a failed obligation,
not a cache failure or a passing numerical test. Subsequent initial candidate
runs are exploratory runtime and exact state-transfer checks while the
baseline nondeterminism is diagnosed. No generated fixture is removed and
no candidate-based output tolerance is introduced.


## Deterministic comparison corrections

Independent layer traces of the clean baseline locate the first differing
values inside local quantized MoE experts: input activations, routing IDs,
routing weights, and router logits are byte-identical before that operation.
A separate W4/group128 GPU fixture observes six output variants in nine fresh
native-alignment trials, but one output with either frozen alignment or
canonical stable alignment. Marlin's expert placement can change split-K
partitioning even with FP32, non-atomic accumulation. The deterministic-only
correction uses canonical integer placement and non-atomic accumulation.
Native execution retains its existing policy. This fixes repeatability for
identical shapes; batch invariance is a separate test obligation.

Above 512 compressed blocks, QSA's native top-k collector also permits arbitrary
output order and threshold ties. The deterministic comparison contract selects
the exact largest-score set, resolves ties by smaller relative index, then
orders selected indices increasingly with trailing -1 padding. Both prefill
and decode must follow this contract. An independent Python ordering oracle
and CUDA graph replay tests cover it. The initial FlashInfer implementation
failed the first GPU call on SM89 RTX 4090 with `cudaErrorNotSupported`;
the failure log is retained with the private validation evidence.
Installed `flashinfer/topk.cuh:3535` documents that FilteredTopK requires 128 KiB
of dynamic shared memory; `:3678` requires that path for explicit index ties or
graph-safe selection. The target has 100 KiB per SM, so this API cannot satisfy
the contract on this hardware. Compilation alone did not establish support.

The replacement uses Torch stable descending argsort on static row tiles,
then sorts the selected relative indices increasingly. A gather places each
valid score interval first in relative-index order; masking its tail preserves
all valid scores, including valid negative infinity ahead of invalid padding.
There are no device scalar reads or host-dependent loop bounds. The tile
estimate allows 16 MiB at 64 bytes per score for explicit wide intermediates
and sort workspace, giving four rows at score width 65,536; a single wider
row is the minimum work unit. This estimate is additional to the input logits,
fixed-width output and one width-sized column vector. Actual CUDA allocator
peaks, sort latency and graph replay remain GPU qualification obligations.
The probe includes width 65,536 and a separate one-shot 2,048-by-65,536 prefill
timing. Native flag-off selection, score values, HTTP fixtures and exact
comparison requirements stay unchanged. No score perturbation or fitted
tolerance is used.

Long input-logprob requests exceeded available GPU scratch at the upstream
2,048-row logprob default. The serving profile uses 256-row logprob chunks to
bound vocabulary-sized scratch; model chunked prefill stays at 2,048 tokens.

## Batch-dependent reduction follow-up

Snapshot 5 passed all 16 frozen single-request cold/warm cases through a
262,016-token prefix, but two of eight warm 65,536-token concurrent requests
produced different reasoning tokens. Both ranks passed strict restored-byte and
selected-cache checks. Their complete prefix event sequences agree; 1,265 host
evictions stayed within the budget, static GPU pools did not change, and all
logical pages were recovered. This is a failed numerical obligation, not a
waived result. A short-answer concurrency test alone also does not prove an
eight-request decode graph; the added fixed-length check requires the matching
request IDs in actual B2 and B8 replay events.

An independent Marlin probe using the model's actual group64 gate-up
(K=2560, N=640) and down (K=320, N=2560) geometry found 162 differing cases out
of 237 batch/placement/background fixtures, while each fixed case repeated
exactly. Stable expert placement does not remove batch-dependent K stripes.
No actual M96 server prefill aggregation is inferred from synthetic M96 tests.

The new deterministic-only correction fixes M/K/N launch dimensions and gives
each CUDA thread block complete K reductions. A compiled C++ integer oracle
checks complete, unique K-tile ownership and rejects the native counterexample.
It changes deterministic arithmetic order, so the affected independent GPU
oracles, graph tests and frozen HTTP comparisons must pass again. The ordinary
native path retains its existing launch and reduction policy. No fixture or
exact comparison requirement is removed, and no error tolerance is fitted.
