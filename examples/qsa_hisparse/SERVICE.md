# Local service lifecycle

`scripts/qsa_service.py` runs the server in a transient systemd **user** service.
It requires Linux cgroup v2, Python 3, `systemd-run`, and a working
`systemctl --user` manager. The controller imports no SGLang modules and checks
the configured HTTP readiness endpoint. SGLang's `/health` can issue an internal
one-token generation when `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION` is enabled;
the controller does not submit a separate generation request. Choose a
non-generating endpoint or disable that server option when readiness must avoid
inference.

Copy [service-profile.example.json](service-profile.example.json) outside the repository and fill in absolute
paths. The example is a generic TP2/B8/256K profile; preserve any additional
machine-specific launch flags in your own profile. Keep local checkpoint paths,
profiles, ownership records, and logs outside this checkout.

```bash
python3 scripts/qsa_service.py start --profile /absolute/path/to/service/current.json
python3 scripts/qsa_service.py status --profile /absolute/path/to/service/current.json
python3 scripts/qsa_service.py logs --profile /absolute/path/to/service/current.json --follow
python3 scripts/qsa_service.py restart --profile /absolute/path/to/service/current.json
python3 scripts/qsa_service.py stop --profile /absolute/path/to/service/current.json
```

The machine-specific `qwen-service.sh` wrapper accepts the same commands and
options. It selects `service/current.json` by default. For rollback, invoke
`../qwen-service.sh restart --profile /absolute/path/to/service/rollback.json`
from this repository's root, or use the wrapper's absolute path.
`QWEN_SERVICE_PROFILE` can select another default profile. Profiles that manage
the same service must use the same `name` and `state_dir`; `restart` stops the
owned invocation before applying the selected profile. Editing a profile while
the service runs does not alter it: every start saves a private launch snapshot,
and status checks that snapshot. An idempotent start refuses profile drift.

`start` blocks until a listener in the service's cgroup returns HTTP 200 from
the configured readiness path and, when configured, `/v1/models` contains the
expected served model name. It refuses a conflicting listener before launch
and checks socket ownership before trusting HTTP readiness. An early exit,
readiness timeout, or interrupted startup stops the entire managed cgroup and
reports recent logs. A healthy repeated `start` preserves the running process.
Concurrent lifecycle mutations are serialized by a private file lock. Read-only
status and logs remain available while a start waits for model loading.

`stop` sends SIGTERM and waits up to `stop_timeout_seconds` (45 seconds by
default); systemd then sends SIGKILL to remaining workers. Cgroup cleanup covers
workers that create their own process groups or sessions. Only a systemd unit
with the recorded invocation ID can be stopped: stale PID data is never a kill
target, and an existing unmanaged server or a reused unit is never adopted.
Each launch precommits a unique marker to its record and unit; that marker can
recover the invocation ID if the controller exits between dispatch and saving
the ID. It must match before any cleanup is authorized.
An ownership conflict requires investigating the record/unit rather than
deleting the record and retrying. The script never uses `pkill` or model-name
process matching.

The service survives closure of Codex, the launching terminal, and tmux.
It has `Restart=no`: there is no automatic restart, GPU unload/sleep policy,
or boot startup. User systemd follows the user's session policy; persistence
after the final user logout requires separately configured user lingering.
No session or lingering settings are changed by this controller.

`status --json` returns machine-readable status and the launch profile/log path.
Exit codes are 0 for ready, 3 for stopped, 4 for an active unhealthy service,
and 1 for a failed service, a conflicting listener, or configuration, ownership,
or control failures. `logs --lines 100`
shows recent append-only logs; add `--follow` to use `tail -F`. A runtime crash
is visible through status and logs and does not trigger a restart loop.

Prefix-cache environment values in a profile are configuration, not evidence
of runtime support or qualification. Keep `--disable-radix-cache`, use
`SGLANG_QSA_HISPARSE_PREFIX_CACHE_MB=0` for a no-cache candidate, and enable a
positive cache budget only after the chosen source implements and passes its
prefix checks. The rollback profile preserves its original source and launch
environment. Health/model probes do not certify prefix reuse.

With `--enable-deterministic-inference`, Marlin uses stable integer alignment,
blockM8, K/N tiles64/128, 128 threads, and one CTA per SM with complete K
slices. This removes batch-dependent split-K accumulation partitions. It changes
floating accumulation order in deterministic mode; the flag-off path retains
native alignment, launch heuristics, and split-K arithmetic. Other model
backends still require their own cold/warm and concurrent qualification.

The independent `test/manual/marlin_batch_invariance.py` fixture uses the actual
TP-local symmetric W4/group64 profile: gate-up2560×640, down320×2560,
BF16/E512/top10. Its default compares target rows at M1–8 and larger selected
shapes under different routing backgrounds; `--cuda-graphs` also changes routing
between graph replays. It reports independent float64 CPU dequantization error
and requires bitwise agreement across selected target placements, with no fitted
tolerance. The earlier group128/M76 alignment diagnostic is a separate fixture.

```bash
python3 test/manual/marlin_batch_invariance.py --cpu-only --output /tmp/marlin-cpu.json
# Root-run GPU checks, without a model load:
python3 test/manual/marlin_batch_invariance.py --output /tmp/marlin-batch.json
python3 test/manual/marlin_batch_invariance.py --cuda-graphs --batch-sizes 1 8 96 2048 --patterns identical spread_routes --output /tmp/marlin-graphs.json
```

CPU lifecycle qualification (uses isolated fake HTTP servers and user units):

```bash
python3 -m unittest discover -s test/qsa_hisparse -p test_service_control.py -v
```
