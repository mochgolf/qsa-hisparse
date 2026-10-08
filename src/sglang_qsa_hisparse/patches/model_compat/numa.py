"""Rows U02-U04: production's ``SGLANG_NUMA_INTERLEAVE`` (eec9df4723).

With ``SGLANG_NUMA_INTERLEAVE=1`` the NUMA V2 path starts each scheduler
subprocess under ``numactl <cpu binding> --interleave=all`` instead of
``--membind=<node>``; when the kernel rejects that policy the probe falls back
to the CPU-only binding. The variable defaults to off, which is upstream's
behavior. ``configure_subprocess`` runs in the server's main process, which
activates the plugin (``load_plugins()``) before it starts the schedulers.
"""

from sglang.srt.environ import EnvBool, envs
from sglang_qsa_hisparse.patching import attach_value, patch

NUMA = "sglang.srt.utils.numa_utils"
NOT_SCOPED = (
    "Not scoped (rule 9): the variable is its own opt-in, it sets the host "
    "memory policy of worker processes rather than model computation, and the "
    "main process that runs configure_subprocess publishes no served-model "
    "config (as for I1)."
)

# Envs fields learn their name from __set_name__, which setattr on the
# finished class does not call.
_interleave = EnvBool(False)
_interleave.name = "SGLANG_NUMA_INTERLEAVE"

attach_value(
    "sglang.srt.environ.Envs",
    "SGLANG_NUMA_INTERLEAVE",
    _interleave,
    feature="model_compat",
    row="U02",
    depends=(
        "sglang.srt.environ.Envs",
        "sglang.srt.environ.EnvField",
        "sglang.srt.environ.EnvBool",
        "sglang.srt.environ.exportable_env_vars",
    ),
    reason=(
        "model_compat: the variable changes worker memory placement while "
        "SGLANG_QSA_HISPARSE_V3 is unset. Production declares the field in Envs "
        "(default False); attached as a class attribute it is read through "
        "envs like any field and is listed by exportable_env_vars, which "
        "iterates vars(Envs). The name is set by hand because setattr does not "
        "call EnvField.__set_name__. " + NOT_SCOPED
    ),
)


@patch(
    f"{NUMA}._numactl_cpu_mem_args",
    "after",
    feature="model_compat",
    row="U03",
    depends=(f"{NUMA}.configure_subprocess", f"{NUMA}._probe_numactl_args"),
    reason=(
        "model_compat (see U02). After hook: every non-None result of the "
        "pinned function is '<cpu binding> --membind=<node>' (--cpunodebind or "
        "--physcpubind, the latter a comma list without spaces); production "
        "builds the same CPU binding and swaps only the memory argument for "
        "--interleave=all when the variable is set, and returns None on the "
        "same empty-CPU-intersection path. configure_subprocess calls this "
        "function by module global, so it sees the hook. " + NOT_SCOPED
    ),
)
def _interleave_memory(result, node, gpu_id):
    interleave = envs.SGLANG_NUMA_INTERLEAVE.get()
    if result is None or not interleave:
        return result
    cpu_arg, _membind = result.rsplit(" ", 1)
    return f"{cpu_arg} --interleave=all"


@patch(
    f"{NUMA}._strip_memory_args",
    "after",
    feature="model_compat",
    row="U04",
    depends=(f"{NUMA}._probe_numactl_args",),
    reason=(
        "model_compat (see U02). After hook: production drops tokens starting "
        "with --membind or --interleave; the pinned function already drops "
        "--membind and returns the kept tokens joined by single spaces, so "
        "dropping --interleave tokens from its result gives the same string. "
        "_probe_numactl_args reaches this only for its CPU-only fallback "
        "(--interleave=all skips its --membind -> --preferred step). " + NOT_SCOPED
    ),
)
def _strip_interleave(result, numactl_args):
    return " ".join(
        token for token in result.split() if not token.startswith("--interleave")
    )
