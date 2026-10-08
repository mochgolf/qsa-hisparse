# P5-B graph, pools and forward batch at 35f3c96ff4

Read `P5-common.md`. Rows of P4-B (G*, R*, F*, C*, K*). Port with production
as the reference (changed: G02, K02 and the graph/pool depends). Map the `?`
hunks of `model_executor/forward_batch_info.py` (4: `req_pool_indices_cpu`
for QSA decode, `kv_allocated_lens_cpu`), `model_runner.py` (1: QSA coordinator
wiring, production's live TP CPU group) and `pool_configurator.py` (1). D3,
D6 and D7 are in this area: report each against production's code.
