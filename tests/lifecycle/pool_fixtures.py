"""Construct the upstream typed state/slot inputs for CPU lifecycle fixtures."""

from sglang.srt.mem_cache.kv.plan import TREE_STATE, TREE_TOKEN, TREE_WINDOW, TreeSlot
from sglang.srt.mem_cache.static_kv_storage import acquire_rows


def tree_slots(*components):
    families = {"FULL": TREE_TOKEN, "SWA": TREE_WINDOW, "MAMBA": TREE_STATE}
    return tuple(
        TreeSlot(
            component=component.name,
            family=families[component.name],
            checkpoint_interval=64 if component.name == "MAMBA" else None,
        )
        for component in components
    )


def allocate_rows(pool, reqs):
    return acquire_rows(pool, getattr(pool, "state", None), reqs)
