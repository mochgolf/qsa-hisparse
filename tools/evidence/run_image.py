#!/usr/bin/env python3
"""Run the Track I GPU evidence (I-C) on the plugin arm.

    run_image.py --base-profile <service/candidate-acceptance.json>
                 --fixtures <frozen image fixtures>
                 --text-fixtures <G2 fixtures.json> --text-reference <G2 fork qualification.json>
                 --output <new directory>

Two sessions, one after the other, each a fresh plugin server: run_g2.py's
plugin arm (the deterministic p2-offload base profile through the launcher,
checkpoint observer on) plus ``--mm-preprocess-cache-size-mb 512`` (artifact
keys exist only on the preprocess-cache path) and the ViT encode observer
(``QSA_EVIDENCE_VIT_DIR``); both share ``<output>/cache`` as
``SGLANG_CACHE_DIR``:

- ``vit-cache-on``: ``SGLANG_VLM_CACHE_SIZE_MB`` unset (pin default, the
  per-image ViT cache is on);
- ``vit-cache-off``: ``SGLANG_VLM_CACHE_SIZE_MB=0``, so every request
  encodes the images it computes (straddling images on hits, and the batch
  invariance check of image_prefix_harness.py). Before it starts, the run
  waits up to GPU_IDLE_TIMEOUT for nvidia-smi to list no compute process.

Each session runs image_prefix_harness.py (image cases and the text control,
with both observer directories, so each image hit needs its own restore at
the expected length on every rank) into ``<session>/image-prefix.json``,
stops its server, then checks the observers: every TP rank wrote a ViT log
and a checkpoint log with a capture and a restore, and every restore equals
the capture it restored (compare.py). ``summary.json`` holds each session's harness exit status and
observer problems; the exit status is 0 only if both sessions pass.
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import image_prefix_harness  # noqa: E402
import run_g2  # noqa: E402
from compare import missing_observer, observer_consistency  # noqa: E402
from run_compat import option, preflight, stop, wait_ready  # noqa: E402

PREPROCESS_CACHE = ["--mm-preprocess-cache-size-mb", "512"]
SESSIONS = (("vit-cache-on", None), ("vit-cache-off", "0"))
GPU_IDLE_TIMEOUT, GPU_IDLE_POLL = 300, 2.0  # seconds


def wait_gpu_idle():
    """Wait (bounded) until nvidia-smi lists no compute process: the previous
    session's server processes can still be releasing the GPUs after its
    process group exited (window 2)."""
    deadline = time.monotonic() + GPU_IDLE_TIMEOUT
    while True:
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
        if not apps:
            return
        if time.monotonic() >= deadline:
            raise SystemExit(f"GPU compute processes still listed after {GPU_IDLE_TIMEOUT} s:\n{apps}")
        time.sleep(GPU_IDLE_POLL)


def observer_problems(output, ranks):
    problems = missing_observer(output, ranks)
    for rank in range(ranks):
        path = output / "observer" / f"rank-{rank}.jsonl"
        message = observer_consistency(path) if path.exists() else None
        if message is not None:
            problems.append(f"observer/{path.name} restore vs capture: {message}")
        if not (output / "vit" / f"rank-{rank}.jsonl").exists():
            problems.append(f"vit/rank-{rank}.jsonl is missing")
    return problems


def session(base, output, vlm_cache_mb, harness_args):
    output.mkdir()
    for name in ("events", "observer", "vit"):
        (output / name).mkdir()
    command, env, ready_file = run_g2.server("plugin", base, output)
    command += PREPROCESS_CACHE
    env["QSA_EVIDENCE_VIT_DIR"] = str(output / "vit")
    # One JIT cache for both sessions (content-hashed keys): build Marlin once.
    env["SGLANG_CACHE_DIR"] = str(output.parent / "cache")
    env.pop("SGLANG_VLM_CACHE_SIZE_MB", None)
    if vlm_cache_mb is not None:
        env["SGLANG_VLM_CACHE_SIZE_MB"] = vlm_cache_mb
        harness_args = [*harness_args, "--vit-cache-off"]
    host, port = base["listen"]["host"], base["listen"]["port"]
    url = f"http://{host}:{port}"
    preflight(host, port)
    provenance = run_g2.provenance("plugin", command, env)
    (output / "run.json").write_text(json.dumps(provenance, indent=2, sort_keys=True))
    with (output / "server.log").open("w") as log:
        process = subprocess.Popen(
            command, env=env, cwd=output, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )  # fmt: skip
    try:
        wait_ready(process, url + base["readiness"]["path"], base["readiness"], ready_file)
        with urllib.request.urlopen(url + "/get_server_info", timeout=30) as response:
            (output / "server_info.json").write_bytes(response.read())
        status = image_prefix_harness.main(
            ["--url", url, *harness_args, "--vit-log", str(output / "vit"),
             "--observer-log", str(output / "observer"),
             "--tp-size", option(command, "--tp-size"),
             "--output", str(output / "image-prefix.json")]
        )  # fmt: skip
    finally:
        stop(process, base["stop_timeout_seconds"])
    problems = observer_problems(output, int(option(command, "--tp-size")))
    return {"harness_exit": status, "observer_problems": problems}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-profile", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--text-fixtures", type=Path, required=True)
    parser.add_argument("--text-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    base = json.loads(args.base_profile.read_text())
    json.loads(args.fixtures.read_text())["cases"]  # Fail before starting a server.
    image_prefix_harness.load_text(
        args.text_fixtures, args.text_reference, image_prefix_harness.TEXT_MAX_PREFIX
    )
    harness_args = [
        "--fixtures", str(args.fixtures.resolve()),
        "--text-fixtures", str(args.text_fixtures.resolve()),
        "--text-reference", str(args.text_reference.resolve()),
    ]  # fmt: skip
    output = args.output.resolve()
    output.mkdir(parents=True)  # Refuses an existing directory.
    summary = {}
    for index, (name, vlm_cache_mb) in enumerate(SESSIONS):
        if index:
            wait_gpu_idle()
        summary[name] = session(base, output / name, vlm_cache_mb, harness_args)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    passed = all(not s["harness_exit"] and not s["observer_problems"] for s in summary.values())
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
