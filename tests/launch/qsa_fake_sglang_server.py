"""Stand-in for ``sglang.launch_server`` in launcher tests (no model, no GPU).

Run as ``python -m qsa_fake_sglang_server <sglang args>`` through the
launcher's test-only ``--server-module``. It starts one process per TP rank
with the ``spawn`` start method (as SGLang starts its schedulers), waits until
every rank finished its startup, and serves ``/health``.
``QSA_FAKE_SERVER_MODE`` selects what the ranks do:

- ``loader``: SGLang's real plugin loader, then the framework's scheduler
  verifier (activation is restricted to the framework rows plus one stand-in
  model_compat row, because the real feature is not complete yet);
- ``records``: write an activation record directly;
- ``missing-rank``: like ``records``, but rank 1 writes none;
- ``wrong-versions``: like ``records`` with a different torch version;
- ``exit``: the server exits with status 17 after starting its ranks.

Process ids go to ``$QSA_FAKE_SERVER_DIR/processes.json``.
"""

import argparse
import http.server
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path


def _activate_through_loader(rank):
    import sglang_qsa_hisparse.patching as patching

    manifest = {
        row: entry
        for row, entry in patching.load_manifest().items()
        if entry["feature"] == "framework"
    }
    target = "sglang.srt.managers.scheduler.run_scheduler_process"
    manifest["T1"] = {
        "feature": "model_compat",
        "attach": [],
        "patches": [{"target": target, "hook_type": "around"}],
    }
    patching.load_manifest = lambda: manifest

    def import_features(feature):
        patching.patch(target, "around", feature="model_compat", row="T1")(
            lambda original, *a, **k: original(*a, **k)
        )

    patching._import_feature_modules = import_features

    from sglang.srt.plugins import load_plugins

    load_plugins()
    from sglang_qsa_hisparse.patches.framework import _verify_scheduler_activation

    # The arguments run_scheduler_process passes: server_args, gpu_id,
    # tp_rank, attn_cp_rank, moe_dp_rank, moe_ep_rank, pp_rank, dp_rank.
    _verify_scheduler_activation(None, 0, rank, 0, 0, 0, 0, None)


def _write_record(mode, rank):
    from sglang_qsa_hisparse.features import read_features
    from sglang_qsa_hisparse.launch import record_details

    if mode == "missing-rank" and rank == 1:
        return
    features = read_features()
    record = {
        "pid": os.getpid(),
        "role": "scheduler",
        "features": list(features.active),
        "hisparse_mode": features.hisparse_mode,
        "tp_rank": rank,
        **record_details(),
    }
    if mode == "wrong-versions":
        record["native_versions"]["torch"] = "0.0.0"
    directory = Path(os.environ["SGLANG_QSA_ACTIVATION_DIR"])
    (directory / f"scheduler-{os.getpid()}.json").write_text(json.dumps(record))


def _rank(mode, rank, started):
    if mode == "loader":
        _activate_through_loader(rank)
    else:
        _write_record(mode, rank)
    started.set()
    while True:
        time.sleep(60)


class _Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--tp-size", type=int, default=1)
    args, _ = parser.parse_known_args()
    mode = os.environ["QSA_FAKE_SERVER_MODE"]
    context = multiprocessing.get_context("spawn")
    events = [context.Event() for _ in range(args.tp_size)]
    ranks = [
        context.Process(target=_rank, args=(mode, rank, events[rank]))
        for rank in range(args.tp_size)
    ]
    for process in ranks:
        process.start()
    Path(os.environ["QSA_FAKE_SERVER_DIR"], "processes.json").write_text(
        json.dumps({"server": os.getpid(), "ranks": [p.pid for p in ranks]})
    )
    print("fake server started", mode, flush=True)
    if mode == "exit":
        sys.exit(17)
    while not all(event.is_set() for event in events):
        if any(process.exitcode is not None for process in ranks):
            print("a rank failed during startup", flush=True)
            sys.exit(1)
        time.sleep(0.05)
    http.server.HTTPServer((args.host, args.port), _Health).serve_forever()


if __name__ == "__main__":
    main()
