# P5-D runtime, image, activation, NUMA and evidence at 35f3c96ff4

Read `P5-common.md`. Paths of P4-D. (1) Runtime: the moved modules must equal
production's `mem_cache/qsa_hisparse/*` (production changed some, e.g. TP rank
from the published parallel bundle); re-check `RUNTIME_DEPENDS` (26 changed).
(2) NUMA interleave (`eec9df4723`, production sets `SGLANG_NUMA_INTERLEAVE=1`):
map the `?` hunks of `environ.py` (1) and `utils/numa_utils.py` (7) to new
rows and implement them (env var registration and the numactl argument
handling). (3) `environment.lock.json` and the launcher for the production
interpreter. (4) Evidence tools (prepared, not run): `run_g2.py`,
`run_compat.py` and the G2-1 scripts with production's code as the fork arm
(`FORK_ROOT`, interpreter, production's test paths), the plugin arm on the
new pin; a G2-3 runner for the native profile comparing production with the
plugin (production weights; latency reported side by side, no threshold;
`docs/baseline.md` step 16).
