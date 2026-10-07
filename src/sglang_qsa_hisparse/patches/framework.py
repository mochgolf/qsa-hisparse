"""Framework hooks installed whenever any feature is active."""

from sglang_qsa_hisparse.patching import FRAMEWORK, patch, verify_final


@patch(
    "sglang.srt.managers.scheduler.configure_scheduler_process",
    "before",
    feature=FRAMEWORK,
    row="FW1",
    depends=(
        # Private registry API used by patching.activate/verify_final.
        "sglang.srt.plugins.hook_registry.HookRegistry.register",
        "sglang.srt.plugins.hook_registry.HookRegistry._apply_target",
        "sglang.srt.plugins.hook_registry.HookRegistry._target_sort_key",
        "sglang.srt.plugins.hook_registry._wrap_fn",
        "sglang.srt.plugins.hook_registry._propagate_patch",
        "sglang.srt.plugins.load_plugins",
        "sglang.srt.managers.scheduler.run_scheduler_process",
    ),
    reason=(
        "Module-level function that run_scheduler_process calls by global name "
        "right after load_plugins(), in every scheduler/TP process: replacing "
        "the Scheduler class cannot remove it, and later hooks on it are "
        "rejected by verify_final."
    ),
)
def _verify_scheduler_activation(*args, **kwargs):
    # Pinned order: server_args, gpu_id, tp_rank, attn_cp_rank, moe_dp_rank,
    # moe_ep_rank, pp_rank, dp_rank, ...
    tp_rank = kwargs.get("tp_rank", args[2] if len(args) > 2 else None)
    pp_rank = kwargs.get("pp_rank", args[6] if len(args) > 6 else None)
    dp_rank = kwargs.get("dp_rank", args[7] if len(args) > 7 else None)
    verify_final("scheduler", tp_rank=tp_rank, pp_rank=pp_rank, dp_rank=dp_rank)
