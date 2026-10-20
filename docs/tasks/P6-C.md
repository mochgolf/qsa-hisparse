# P6-C model, QSA, quantization and kernels at v0.5.22

Read `P6-common.md`. Rows A*, E*, H*, J03, Q*, T*, Z*. Changed at v0.5.22
(survey): Q08, Q10, T03 (REPLACE), H08 (around), and `depends` in every
model_compat/QSA fingerprint file. Check, R2 first:

- #41933 (`047405f534`): `forward_extend` now packs prefill K/V with
  `pack_qsa_prefill_kv(..., output_dtype=q.dtype)` (a cast, no scale).
  Production's FP8 KV descale in prefill and reference reads (`bdb935d70f`,
  plugin rows with `_kv_descales`, A05/Q08/Q13) must keep its numerics: the
  scale applied exactly once, never lost; FP8 cached prefixes bitwise equal
  to production's path where production's path is deterministic. Narrow or
  keep Q08 accordingly; a plugin copy of `pack_qsa_prefill_kv` only if the
  descale cannot be applied around it.
- #43463 (`23a1f53fc3`): decode indexer bounded by context length vs T03.
- #42175 / #41933 / H08 (`GatedResidual.mix`, -4 +28): the HC path and the
  plugin's stable/batch-invariant HC.
- Kernels: plugin Marlin JIT copy and `fast_topk` (production `773f3c2d84`)
  against v0.5.22's sources (sglang-kernel 0.4.9 unchanged); re-derive
  `tests/model_compat/test_model_compat_copies.py` digests only where the
  reference changed (it did not: production is still the reference).

Tests: `tests/model_compat/`, `tests/qsa/`.
