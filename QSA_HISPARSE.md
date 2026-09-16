# QSA HiSparse runtime guide

The implementation lives in
[`python/sglang/srt/mem_cache/qsa_hisparse/`](python/sglang/srt/mem_cache/qsa_hisparse/).
It uses SGLang's QSA indexer and attention backend, with a CPU copy of raw K/V
and bounded GPU hot caches. Compressed index-K and Mamba state keep their normal
SGLang ownership.

## Read the code in this order

1. [`config.py`](python/sglang/srt/mem_cache/qsa_hisparse/config.py) validates
   model geometry, memory, execution mode, and graph configuration.
2. [`layout.py`](python/sglang/srt/mem_cache/qsa_hisparse/layout.py) defines C4
   packing, unpacking, and short-prefix initialization.
3. [`slots.py`](python/sglang/srt/mem_cache/qsa_hisparse/slots.py) owns physical
   staging/ring leases and rejects stale request generations.
4. [`runtime.py`](python/sglang/srt/mem_cache/qsa_hisparse/runtime.py) implements
   `QSAHiSparseRuntime`: allocation, prefill handoff, selection/refetch, graph
   replay, writeback, and release across request slots.
5. [`coordinator.py`](python/sglang/srt/mem_cache/qsa_hisparse/coordinator.py)
   connects leases to the scheduler and requires TP ranks to agree on request
   identities before decode admission.
6. [`single_request.py`](python/sglang/srt/mem_cache/qsa_hisparse/single_request.py)
   retains the eager comparison runtime and shared per-request writeback/checking
   routines. Its internal byte checks supplement independent tests; they do not
   establish correctness by themselves.

The old `qsa_hisparse_p2.py`, `qsa_hisparse_v3.py`, and `qsa_hisparse_slots.py`
contain compatibility imports only. Current runtime code and tests use the new
package. Historical drivers inspecting private attributes must use their
recorded revision.

## Storage and ownership

| Storage | Owner and layout |
| --- | --- |
| Logical tokens and compressed index-K | Ordinary SGLang paged allocator; retained until request release |
| Raw prefill K/V | One shared 262,144-token staging arena per TP rank |
| Request tail | Five rows per lease: padding row 0 and four C4 members |
| Host K/V | Pinned bytes indexed by request slot, layer, and logical C4 block |
| C4 record | Four K rows followed by four V rows; `8 × 256 = 2,048` FP8 bytes |
| Hot cache | 2,048 C4 cache entries plus reserved workspace per request/layer |
| Attention scratch | Selected 2,048 raw rows plus up to three pending tail rows |

At eight slots the raw host slab is `8 × 12 × 65,536 × 2,048` bytes = 12 GiB
per TP rank, before model/PLE host storage. A decode batch row is temporary
routing metadata; it is never a persistent lease identity.

```text
prefill -> copying -> host_ready -> decode
                     (TP agreement)
finish/abort -> drain GPU/copy events -> flush logical free group -> release lease
```

Staging cannot be reused before every layer's handoff completes. A decode C4
close must finish D2H writeback before that block is refetched. Release drains
outstanding work before freeing logical/Mamba ownership; physical slots become
reusable only after the allocator's free group flushes. Failed copies retain
ownership and propagate the failure.

## Modes and configuration

Existing environment spellings remain supported. `V3` and `P2` in these names
are historical experiment labels, not SGLang version requirements.

| `SGLANG_QSA_HISPARSE_V3` | Runtime |
| --- | --- |
| Unset | Ordinary upstream QSA |
| `p2-offload` | Multi-request CPU offload; eager or bounded decode graphs |
| `p2-resident` | Multi-request resident comparison; eager only |
| `offload` | Single-request eager offload comparison |
| `resident` | Single-request eager resident comparison |

`SGLANG_QSA_HISPARSE_V3_OBSERVE` defaults to `strict`; `light` reduces observation
overhead. Strict mode requires `--enable-deterministic-inference`.
`SGLANG_QSA_HISPARSE_V3_EVENTS=/path/to/events` optionally writes per-rank JSONL
ledgers. The multi-request path rejects the legacy tensor-capture option.

The frozen geometry is TP2/PP1, 12 full-attention layers, one KV head per rank,
head dimension 256, plain NHD FP8 E4M3, C4, page64, and top512 C4 blocks. Context
length is 262,144; total logical capacity is context length multiplied by
request slots. Multi-request chunked prefill is 2,048 or 4,096 tokens. Graph
decode captures every batch size from 1 through capacity, with graph padding
and prefill graphs disabled.

Radix/prefix sharing, overlap scheduling, speculation, PD disaggregation,
mixed prefill/decode, preemption, and concurrent physical prefill staging remain
outside this contract. Unsupported settings fail validation. This source
migration does not widen the supported model geometry.

[`examples/qsa_hisparse/serve.sh`](examples/qsa_hisparse/serve.sh) spells out the
B8 options. For the original AutoRound INT8-row PLE checkpoint, add:

```bash
MODEL_PATH=/path/to/int8-row-ple-model \
  bash examples/qsa_hisparse/serve.sh \
    --ple-offload-embedding \
    --json-model-override-args '{"text_config":{"ple_embedding_dtype":"int8_row"}}'
```

These options require actual INT8-row PLE weights and scales; they do not
quantize or convert arbitrary checkpoints. NUMA placement and model loading
options are host-specific and can be appended to the script.

## SGLang integration boundaries

Paths below are relative to `python/sglang/srt/`.

| Component | Contract |
| --- | --- |
| `model_executor/pool_configurator.py`, `mem_cache/kv_cache_configurator.py` | Size logical/index storage separately from raw staging |
| `layers/attention/qwen_sparse_attn_backend.py` | Remap raw writes only; keep logical index-K addresses; consume compact KV |
| `model_executor/model_runner.py`, `managers/scheduler.py` | Attach the coordinator and admit ready leases |
| `model_executor/runner/decode_cuda_graph_runner.py` | Capture persistent buffers; route replay by current leases |
| `mem_cache/common.py`, `mem_cache/allocator/paged.py` | Drain, logical release, free-group flush, then physical release |
| `layers/attention/qsa/sparse_attn.py`, `hisparse_graph.py` | FP8 scales, selection order, padding and byte movement |

The shared runtime attachment is `qsa_hisparse`; the multi-request scheduler
marker is `uses_qsa_hisparse_leases`. Runtime operations use the checked-in
source directly, without patch application or external source-tree lookups.
