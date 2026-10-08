# Intentional deviations from the fork

The plugin reproduces fork `ee8fe158d6` except for the items below. Each item
names its inventory row, the reason, and the configurations it can affect.
Only the orchestrator adds items; the owner may revert any of them.

| ID | Row | Fork behavior | Plugin behavior | Reason | Affects |
| --- | --- | --- | --- | --- | --- |
| D1 | `FusedMoE.__init__` log (inventory §4) | "deferred finalize" message logged at info | stays at debug | Keeping it needs a ~235-line REPLACE of `FusedMoE.__init__` for a log level | Log output only |
| D2 | `freeze_gc` log | GC freeze message at info | stays at debug | Cosmetic; avoids a REPLACE | Log output only |
| D3 | F01 `ForwardBatch` fields | Class fields `req_pool_indices_cpu` and `kv_allocated_lens_cpu`; `init_new` fills `req_pool_indices_cpu` for every model and batch; two-batch overlap's `filter_batch` raises for every model | Since v0.5.21: uses upstream's `req_pool_indices_cpu` field (upstream fills it only for extend without speculative decoding); the `init_new` AFTER hook fills it for every batch only while the HiSparse runtime is active, otherwise the field keeps upstream's value; `EagerRunner.load_batch`'s `dataclasses.replace` copy carries it (no second hook); `kv_allocated_lens_cpu` (never read) dropped. At pin 76e06febab: an instance attribute plus a second hook on `EagerRunner.load_batch` | Avoids a class REPLACE pinning ~1,400 lines and does not reproduce a crash in an unsupported configuration | Two-batch overlap and DP/MLP-sync padding of the field (upstream handles it; outside the HiSparse contract); non-HiSparse batches keep upstream's value |
| D5 | S03 `Scheduler._build_hisparse_decode_batch` | With `return_logprob`, `token_ids_logprobs` is every prompt token ID: image requests crash decode (pad IDs exceed the vocabulary); text requests waste a whole-prompt gather whose result is dropped | Each request's own `token_ids_logprob` (as `ScheduleBatch` builds decode batches), target model only | Removes a crash and wasted work; text outputs and returned logprobs are unchanged | HiSparse decode batches with `return_logprob` |

Decided 2026-10-07 by the orchestrator from P0-A findings; D1–D3 accepted
by the owner on 2026-10-07; D5 accepted on 2026-10-08 (D4 was declined).
