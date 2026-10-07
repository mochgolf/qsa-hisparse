"""tools/evidence/plugin_probe.py: the plugin arm of the fork's GPU probes.

CPU only: stand-in scripts import the same names as the fork probes
(test/manual/marlin_*.py, qsa_deterministic_topk_probe.py) and report what
they resolved to; nothing calls a GPU kernel or the Marlin JIT. With
QSA_FORK_ROOT set, the real fork Marlin probes also run in their --cpu-only
mode through the runner.
"""

import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import sglang_qsa_hisparse.kernels.stable_align as stable_align

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "tools" / "evidence" / "plugin_probe.py"
HELPER = "python/sglang/srt/layers/moe/fused_moe_triton/stable_align.py"
MARKER = "QSA_PLUGIN_PROBE "

# Imports and helper loading exactly as the fork probes do them.
STAND_IN = textwrap.dedent(
    """
    import importlib.util
    import json
    import sys
    from pathlib import Path

    import torch

    from sglang.srt.layers.attention.qsa.kernel import qsa_fast_topk

    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT / "python"))
    helper = ROOT / "{helper}"
    spec = importlib.util.spec_from_file_location("stand_in_helper", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


    def main():
        from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
        from sglang_qsa_hisparse.kernels import marlin_moe, stable_align

        scores = torch.tensor([[1.0, 3.0, 3.0, 2.0, 3.0]])
        ids = torch.tensor([[3, 1], [1, 0]])
        print(json.dumps(dict(
            name=__name__,
            argv=sys.argv[1:],
            helper_is_plugin_copy=helper.read_bytes()
            == Path(stable_align.__file__).read_bytes(),
            helper_aligns=[
                a.tolist() == b.tolist()
                for a, b in zip(
                    module.moe_align_block_size_stable(ids, 4, 4),
                    stable_align.moe_align_block_size_stable(ids, 4, 4),
                )
            ],
            gemm_is_plugin=moe_wna16_marlin_gemm is marlin_moe.moe_wna16_marlin_gemm,
            deterministic_topk=qsa_fast_topk(
                scores, torch.tensor([0]), torch.tensor([5]), 2, deterministic=True
            ).tolist(),
            cuda_initialized=torch.cuda.is_initialized(),
        )))


    if __name__ == "__main__":
        main()
    """
).format(helper=HELPER)


def run(*args, extra_path=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("SGLANG_QSA_")}
    # The P-arm server environment sets this; the probes must not activate HiSparse.
    env["SGLANG_QSA_HISPARSE_V3"] = "p2-offload"
    if extra_path:
        env["PYTHONPATH"] = f"{extra_path}{os.pathsep}{env.get('PYTHONPATH', '')}"
    return subprocess.run(
        [sys.executable, str(RUNNER), *map(str, args)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


def marker(result):
    (line,) = [l for l in result.stderr.splitlines() if l.startswith(MARKER)]
    return json.loads(line[len(MARKER) :])


@pytest.mark.integration  # Activates every model_compat row in a subprocess.
def test_fork_probe_names_resolve_to_the_plugin(tmp_path):
    fork = tmp_path / "fork"
    script = fork / "test" / "manual" / "stand_in.py"
    script.parent.mkdir(parents=True)
    script.write_text(STAND_IN)
    # A probe that loaded the helper next to the fork script would fail here.
    (fork / HELPER).parent.mkdir(parents=True)
    (fork / HELPER).write_text("raise RuntimeError('fork helper loaded')\n")

    result = run(script, "--output", tmp_path / "out.json")
    assert result.returncode == 0, result.stderr[-4000:]
    info = marker(result)
    assert info["features"] == ["model_compat"]
    assert info["qsa_fast_topk_patched"] is True
    assert info["script_sha256"] == hashlib.sha256(script.read_bytes()).hexdigest()
    assert info["stable_align_sha256"] == hashlib.sha256(
        Path(stable_align.__file__).read_bytes()
    ).hexdigest()
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report == dict(
        name="__main__",
        argv=["--output", str(tmp_path / "out.json")],
        helper_is_plugin_copy=True,
        helper_aligns=[True, True, True],
        gemm_is_plugin=True,
        # Stable ties prefer the smaller logical index (pinned qsa_fast_topk
        # has no deterministic argument; the T02 hook in target scope does).
        deterministic_topk=[[1, 2]],
        cuda_initialized=False,
    )


@pytest.mark.integration  # Activates every model_compat row in a subprocess.
def test_module_mode_runs_under_the_same_activation_and_fails_loudly(tmp_path):
    (tmp_path / "qsa_probe_module.py").write_text(
        textwrap.dedent(
            """
            import sys

            from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
            from sglang_qsa_hisparse.kernels import marlin_moe

            if __name__ == "__main__":
                print("redirected", moe_wna16_marlin_gemm is marlin_moe.moe_wna16_marlin_gemm)
                assert sys.argv[1:] == ["-q"], sys.argv
                raise AssertionError("probe assertion")
            """
        )
    )
    result = run("-m", "qsa_probe_module", "-q", extra_path=tmp_path)
    assert result.returncode != 0
    assert "redirected True" in result.stdout
    assert "AssertionError: probe assertion" in result.stderr
    assert marker(result)["module"] == "qsa_probe_module"


def test_source_override_is_refused(tmp_path):
    result = run(tmp_path / "test" / "manual" / "x.py", "--source", tmp_path)
    assert result.returncode != 0
    assert "--source" in result.stderr
    assert MARKER not in result.stderr


def test_requested_whole_k_gpu_run_fails_without_the_fork_script():
    # Selecting only this node: it fails before any subprocess or CUDA call.
    node = (
        "tests/model_compat/test_marlin_deterministic_alignment.py"
        "::test_whole_k_actual_group64_cuda_graph_target_batch_reference"
    )
    env = {k: v for k, v in os.environ.items() if k != "QSA_FORK_ROOT"}
    env["QSA_GPU_TESTS"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", node],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 1, result.stdout[-3000:]
    assert "1 failed" in result.stdout and "set QSA_FORK_ROOT" in result.stdout


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("QSA_FORK_ROOT"), reason="QSA_FORK_ROOT unset")
def test_fork_marlin_probes_run_their_cpu_mode_on_the_plugin_helper(tmp_path):
    manual = Path(os.environ["QSA_FORK_ROOT"]) / "test" / "manual"
    plugin_helper = hashlib.sha256(Path(stable_align.__file__).read_bytes()).hexdigest()

    whole_k = tmp_path / "whole-k.json"
    result = run(
        manual / "marlin_batch_invariance.py",
        "--cpu-only", "--batch-sizes", "1", "8", "--output", whole_k,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    report = json.loads(whole_k.read_text())
    assert report["cpu_fixture_and_integer_oracle_pass"] is True
    assert report["helper_sha256"] == plugin_helper
    assert not report["source"].startswith(os.environ["QSA_FORK_ROOT"])

    align = tmp_path / "align.json"
    result = run(manual / "marlin_deterministic_alignment.py", "--cpu-only", "--output", align)
    assert result.returncode == 0, result.stderr[-4000:]
    assert json.loads(align.read_text())["stable_alignment_reference_pass"] is True
