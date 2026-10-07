# Intentional deviations from the fork

The plugin reproduces fork `ee8fe158d6` except for the items below. Each item
names its inventory row, the reason, and the configurations it can affect.
Only the orchestrator adds items; the owner may revert any of them.

| ID | Row | Fork behavior | Plugin behavior | Reason | Affects |
| --- | --- | --- | --- | --- | --- |
| D1 | `FusedMoE.__init__` log (inventory §4) | "deferred finalize" message logged at info | stays at debug | Keeping it needs a ~235-line REPLACE of `FusedMoE.__init__` for a log level | Log output only |
| D2 | `freeze_gc` log | GC freeze message at info | stays at debug | Cosmetic; avoids a REPLACE | Log output only |
| D3 | F01 `ForwardBatch` fields | Class fields `req_pool_indices_cpu`, `kv_allocated_lens_cpu` for every model; two-batch overlap's `filter_batch` raises for every model | Instance attribute `req_pool_indices_cpu` set only when the HiSparse runtime is active, carried across `EagerRunner.load_batch`'s `dataclasses.replace` copy by a second hook; `kv_allocated_lens_cpu` (never read) dropped | Avoids a class REPLACE pinning 1,412 lines and does not reproduce a crash in an unsupported configuration | Two-batch overlap (outside the HiSparse contract); dataclass field introspection |

Decided 2026-10-07 by the orchestrator from P0-A findings; D1–D3 accepted
by the owner on 2026-10-07.
