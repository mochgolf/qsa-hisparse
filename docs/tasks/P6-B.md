# P6-B graph, pools and forward batch at v0.5.22

Read `P6-common.md`. Rows G01–G03, R01–R02, F01, C01–C02, K01–K03.
Changed at v0.5.22 (survey): G01, G02 (REPLACE, rebase), K02 (REPLACE),
F01, C01, C02, K01, K03 and many graph/pool `depends`. Check:

- #35594 (in v0.5.22) threads vendor KV pool classes through the in-tree
  builders (`_resolve_kv_pool_class`, PlatformCapabilities): Phase 3 noted
  K02/K03 may narrow onto it (P3).
- #37771 HiSparse speculative decoding and #42175 (Blackwell BF16 decode) in
  the decode graph runner and model runner: re-check G01/G02/G03/R02 and D6,
  D7 against v0.5.22.
- D3 (F01 `req_pool_indices_cpu`) against v0.5.22's `ForwardBatch.init_new`.

Tests: `tests/graph/`, `tests/pools/`.
