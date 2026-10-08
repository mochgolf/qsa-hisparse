"""Framework hooks installed whenever any feature is active."""

from sglang.srt.runtime_context import get_parallel
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
        # publish() records this process's ranks, which the hook reads.
        "sglang.srt.runtime_context.publish",
        "sglang.srt.runtime_context.get_parallel",
    ),
    reason=(
        "Module-level function that run_scheduler_process calls by global name "
        "right after load_plugins() and publish(), in every scheduler/TP "
        "process: replacing the Scheduler class cannot remove it, and later "
        "hooks on it are rejected by verify_final. Its arguments no longer "
        "carry ranks; the hook reads them from the published context."
    ),
)
def _verify_scheduler_activation(*args, **kwargs):
    parallel = get_parallel()
    verify_final(
        "scheduler",
        tp_rank=parallel.tp_rank,
        pp_rank=parallel.pp_rank,
        dp_rank=parallel.dp_rank,
    )
