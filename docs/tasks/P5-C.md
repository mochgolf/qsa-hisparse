# P5-C model, QSA, quantization and kernels at 35f3c96ff4

Read `P5-common.md`. Rows of P4-C (A*, E*, H*, J*, Q*, T*, Z*). Port with
production as the reference (changed REPLACE: Q01, Q08, Q10, Q11, Q12, T03,
T04 and depends). Map the `?` hunks: `kernels/jit/csrc/elementwise/fast_topk.cuh`
(12, production fix `773f3c2d84`: keep all candidates on radix overflow — the
plugin must ship it), `qwen_sparse_attn_backend.py` (24), `qsa/kernel.py`
(7) and `qsa/sparse_attn.py` (4) (FP8 KV descale in prefill and reference
reads, `bdb935d70f`, and production's re-merge), `models/qwen4_exp.py` (6),
`fused_marlin_moe.py` (1). Re-derive the fork digests in
`tests/model_compat/test_model_compat_copies.py` from production; `hc_mix`
is at `kernels/ops/gemm/hc_mix.py` in both pin and production.
