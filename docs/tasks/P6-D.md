# P6-D runtime, image, activation, NUMA and evidence at v0.5.22

Read `P6-common.md`. Paths of P4-D/P5-D except `hisparse/prefix_cache.py`,
`hisparse/coordinator.py`'s `allocate_spec_scratch` and `tests/prefix/`
(P6-A).

1. Runtime package: re-check `RUNTIME_DEPENDS` (`hisparse/depends.py`; 33
   changed, `utils.common.Range` missing at v0.5.22) and record v0.5.22
   interface changes in `PIN_EDITS` of `tests/runtime/test_moved_sources.py`
   (e.g. `prefix_indices`/`extend_range` reads outside `prefix_cache.py`,
   `release_kv_cache(checkpoint=)`, `mamba_lazy_checkpoint`).
2. Image identity (I1): any change to `MultimodalDataItem` hashing, pad values
   or `qwen_vl` preprocessing between the pins; image prefix keys must stay
   exact (page-64 hits inside images as in window 3 / G5). #43043 (Kimi
   image identity, merged 10-18) is not in v0.5.22: information only.
3. Activation: FW1 and `plugins/` changes; #44444 (`SGLANG_REQUIRED_PLUGINS`,
   open) is information only. NUMA rows U02–U04 against v0.5.22's
   `environ.py`/`utils/numa_utils.py`.
4. `tests/regression/test_plugin_off.py`: pinned results of pristine
   v0.5.22 with no plugin on the Phase 6 interpreter.
5. Evidence tools (prepared, not run): `run_g2.py`, `run_compat.py`, the G2-1
   scripts (step-8 inventory for v0.5.22's test paths) and the G2-3 runner —
   fork arm = production code on production's interpreter, plugin arm =
   v0.5.22 on `results/phase6-20261018/runtime-env`.
