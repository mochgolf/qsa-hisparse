# Maintaining the SGLang fork

`origin` is QSA HiSparse. `upstream` is
[`sgl-project/sglang`](https://github.com/sgl-project/sglang). Both histories are
retained, so future updates use a normal three-way Git merge.

The migration imported original QSA head
`5f8ae43640404eaee4c645d8cad2d0ef6e7dc6b0`, then merged upstream
`76e06febab732d28a61b75a61b7835284568cdfb`: 226 commits after the old
`4309c7ce19` base. The original artifact history is retained too. No neighboring
checkout, Git alternates directory, or patch series is required.

## Update workflow

Start with a clean working tree and an isolated runtime environment. Add the
remote once after a fresh clone:

```bash
git remote add upstream https://github.com/sgl-project/sglang.git
```

For each update:

```bash
git fetch upstream main --tags
git switch -c maintenance/sglang-update
git merge --no-ff --no-commit upstream/main
# Resolve and stage conflicts, then inspect the integration points below.
QSA_PYTHON=python bash scripts/test_qsa_hisparse_cpu.sh
git diff --check
git status
git commit
```

Keep the QSA README as the entry point. Refresh `README.sglang.md` from upstream
when it changes. Record the merged revision, dependency changes, test results,
and GPU validation status in `PROVENANCE.json` and `VALIDATION.md`. Review the
diff against `upstream/main` to see the downstream integration.

## Compatibility review

- **Pool geometry:** logical/index capacity stays separate from raw staging;
  request slots bound physical rings and hot caches.
- **Scheduler/release:** preserve readiness, TP agreement, generation checks,
  terminal drain, and logical free-group ordering.
- **Graph replay:** preserve stable backing, supported shapes, selection order,
  and request-to-batch routing.
- **KV extraction:** retain FP8 descales, 64-bit element offsets, and page-tail
  zero filling together. CPU-interpreter comparisons use non-unit scales and
  poisoned scratch to catch lost operations.
- **Model storage:** preserve INT8-row PLE/meta allocation, upstream FP8
  checkpoint detection, and the AutoRound TP2 Marlin path.
- **API removals:** inspect automatically merged code too. This update removed
  tokenwise QSA and `QSAProfile.variant`; downstream references were removed
  from sizing and graph setup.

The 2026-09-16 merge combined upstream gather memory-safety changes and PLE file
storage with QSA scale handling, deterministic HC, and the offload graph path.
The focused suite is in `test/qsa_hisparse/`; its runner also selects affected
upstream pool, scheduler, PLE, and release tests.

The 2026-09-23 update merges upstream `172b1b4825` (421 commits after
`76e06febab`). It carries the upstream `owned_kv_len` cache-release contract
through the QSA host-prefix adapter, keeps QSA's CPU request identities for
decode while speculative modes use upstream device slots, and combines the
upstream meta-device PLE table construction with QSA INT8-row storage. The
`README.sglang.md` snapshot is byte-identical to upstream's README at this
revision. See [validation](VALIDATION.md) for checks and remaining GPU scope.

CPU tests and packaging checks do not qualify a CUDA deployment. Before
replacing a serving process, run applicable GPU/kernel and service checks with
the new dependencies in an available test window. Historical acceptance
contracts and measurements remain in `archive/` at their original revisions.
