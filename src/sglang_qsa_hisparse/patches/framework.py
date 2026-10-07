"""Framework hooks installed whenever any feature is active."""

from sglang_qsa_hisparse.patching import FRAMEWORK, patch, verify_final


@patch(
    "sglang.srt.managers.scheduler.Scheduler.__init__",
    "before",
    feature=FRAMEWORK,
    reason=(
        "Runs in every scheduler/TP process after load_plugins() returned, so "
        "hooks registered by later plugins are visible; records activation."
    ),
)
def _verify_scheduler_activation(self, *args, **kwargs):
    # Positional order is pinned: server_args, port_args, gpu_id, tp_rank, ...
    tp_rank = kwargs.get("tp_rank", args[3] if len(args) > 3 else None)
    pp_rank = kwargs.get("pp_rank", args[5] if len(args) > 5 else None)
    verify_final("scheduler", tp_rank=tp_rank, pp_rank=pp_rank)
