#!/usr/bin/env python3
"""Plugin arm (P) of the fork's standalone GPU probes (docs/baseline.md, G2-1).

    python tools/evidence/plugin_probe.py <fork script> [script args...]
    python tools/evidence/plugin_probe.py -m pytest [pytest args...]

Runs an unmodified fork script (or module) in a process where the plugin is
active as for the target model, so the script's assertions and output JSON
are its own:

1. ``model_compat`` is activated through ``plugin.load()`` (HiSparse off),
   with ``scope.target_model_active`` forced to True: the probes load no
   model, so there is no served architecture to read.
2. ``sglang.kernels.ops.moe.moe_wna16_marlin.moe_wna16_marlin_gemm`` is
   pointed at the plugin's Marlin op (J01/J02, its own JIT module), which the
   Marlin probes import inside ``main()``. The top-k probe's
   ``qsa_fast_topk(..., deterministic=True)`` reaches the plugin's stable
   top-k through the activated T02 hook (scope forced on).
3. A script runs as a byte copy from a temporary tree whose
   ``python/sglang/srt/layers/moe/fused_moe_triton/stable_align.py`` is the
   plugin's J04 copy: the Marlin probes load that helper relative to their own
   path (``--source`` defaults to it), so ``--source`` is refused.

Needs ``PYTHONPATH=<plugin>/src:<pin>/python``. Before running, prints one
``QSA_PLUGIN_PROBE {...}`` line to stderr recording what was redirected.
"""

import hashlib
import json
import os
import runpy
import shutil
import sys
import tempfile
from pathlib import Path

MARKER = "QSA_PLUGIN_PROBE "
HELPER = Path("python/sglang/srt/layers/moe/fused_moe_triton/stable_align.py")
IN_TREE_GEMM = "sglang.kernels.ops.moe.moe_wna16_marlin"
TOPK = "sglang.srt.layers.attention.qsa.kernel.qsa_fast_topk"


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def activate() -> dict:
    os.environ["SGLANG_QSA_MODEL_COMPAT"] = "1"
    os.environ.pop("SGLANG_QSA_HISPARSE_V3", None)
    from sglang_qsa_hisparse import patching, plugin, scope

    scope.target_model_active = lambda: True
    plugin.load()

    import importlib

    from sglang_qsa_hisparse.kernels import marlin_moe

    in_tree = importlib.import_module(IN_TREE_GEMM)
    in_tree.moe_wna16_marlin_gemm = marlin_moe.moe_wna16_marlin_gemm
    return {
        "features": list(patching._activated.active),
        "patches": len(patching._applied),
        "target_model_active": "forced",
        "moe_wna16_marlin_gemm": marlin_moe.__name__,
        "qsa_fast_topk_patched": TOPK in patching._applied,
    }


def main(argv: list[str]) -> None:
    if not argv or argv[0] == "-m" and len(argv) < 2:
        sys.exit(__doc__)
    if argv[0] == "-m":
        info = activate()
        info["module"] = argv[1]
        print(MARKER + json.dumps(info), file=sys.stderr, flush=True)
        sys.argv = [argv[1], *argv[2:]]
        runpy.run_module(argv[1], run_name="__main__", alter_sys=True)
        return
    script, args = Path(argv[0]).resolve(), argv[1:]
    if any(arg == "--source" or arg.startswith("--source=") for arg in args):
        sys.exit("plugin_probe: --source would load the fork's stable_align.py")
    info = activate()
    from sglang_qsa_hisparse.kernels import stable_align

    with tempfile.TemporaryDirectory(prefix="qsa-plugin-probe-") as directory:
        root = Path(directory)
        copy = root / "test" / "manual" / script.name
        copy.parent.mkdir(parents=True)
        shutil.copyfile(script, copy)
        (root / HELPER).parent.mkdir(parents=True)
        shutil.copyfile(stable_align.__file__, root / HELPER)
        info.update(
            script=str(script),
            script_sha256=_sha256(script),
            stable_align=stable_align.__file__,
            stable_align_sha256=_sha256(root / HELPER),
        )
        print(MARKER + json.dumps(info), file=sys.stderr, flush=True)
        sys.argv = [str(copy), *args]
        sys.path[0] = str(copy.parent)  # As `python <script>` would.
        runpy.run_path(str(copy), run_name="__main__")


if __name__ == "__main__":
    main(sys.argv[1:])
