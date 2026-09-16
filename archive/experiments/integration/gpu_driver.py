#!/usr/bin/env python3
"""One guarded GPU window for V1/V2/V4 plus the real-model V3 probe."""

from __future__ import annotations

import importlib.util
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HARNESS_ROOT = Path(os.environ.get("QSA_HISPARSE_HARNESS_ROOT", "<HARNESS_ROOT>"))
OLD = Path(os.environ.get("QSA_HISPARSE_RUNNER_ROOT", "<RUNNER_ROOT>"))
SOURCE = Path(os.environ.get("QSA_HISPARSE_SOURCE", ROOT / "sources/sglang-hisparse"))
PROD_SOURCE = Path(os.environ.get("QSA_HISPARSE_BASE_SOURCE", ROOT / "sources/sglang"))
PROD_LOG = Path(os.environ.get("QSA_HISPARSE_BASELINE_LOG", "<BASELINE_LOG>"))
INPUT = Path(os.environ.get("QSA_HISPARSE_REQUEST_JSON", "<REQUEST_JSON>"))
PORT, TEST_PORT = 30000, 30001


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


runner = load("hisparse_old_runner", OLD / "run_tests.py")
capture = runner.load_module("hisparse_capture", runner.CAPTURE_PATH)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def nvml_sampler(stop, rows):
    while not stop.is_set():
        started = time.time()
        try:
            p = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2)
            rows.append({"time": started, "returncode": p.returncode, "rows": p.stdout.splitlines()})
        except Exception as exc:
            rows.append({"time": started, "error": f"{type(exc).__name__}: {exc}"})
        stop.wait(.05)


def first_token_request(output):
    payload = json.loads(INPUT.read_text())
    sampling = dict(payload.get("sampling_params") or {})
    sampling["max_new_tokens"] = 1
    payload["sampling_params"] = sampling
    payload["stream"] = True
    started = time.monotonic(); events=[]
    req = urllib.request.Request(f"http://127.0.0.1:{TEST_PORT}/generate", data=json.dumps(payload).encode(), headers={"Content-Type":"application/json","Accept":"text/event-stream"})
    with urllib.request.urlopen(req, timeout=1800) as response:
        for raw in response:
            line=raw.decode(errors="replace").strip()
            if not line.startswith("data:"): continue
            data=line[5:].strip()
            if data=="[DONE]": break
            event=json.loads(data); event["_elapsed_s"]=time.monotonic()-started; events.append(event)
    save(output / "v3-events.json", events)
    meta = events[-1].get("meta_info", {}) if events else {}
    return {"e2e_s":time.monotonic()-started,"event_count":len(events),"prompt_tokens":meta.get("prompt_tokens"),"completion_tokens":meta.get("completion_tokens"),"first_output_s":events[0]["_elapsed_s"] if events else None}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--attempt",required=True); args=parser.parse_args()
    output = HERE / "execution" / args.attempt
    output.mkdir(parents=True, exist_ok=True)
    manifest={"started_at":runner.stamp(),"status":"preflight","script_commit":subprocess.check_output(["git","-C",str(HARNESS_ROOT),"rev-parse","HEAD"],text=True).strip()}
    save(output/"manifest.json",manifest)
    prod_pid=runner.listener_pid(PORT)
    snapshot, prod_env=runner.snapshot_service(prod_pid, PROD_SOURCE)
    save(output/"service-snapshot.json",snapshot)
    if runner.listener_pid(PORT)!=prod_pid: raise RuntimeError("production listener changed during snapshot")
    if runner.server_processes() and any(p["pid"] not in {prod_pid, *[c.pid for c in __import__('psutil').Process(prod_pid).children(recursive=True)]} for p in runner.server_processes()):
        raise RuntimeError(f"unexpected model processes: {runner.server_processes()}")
    runner.assert_test_free(TEST_PORT)
    manifest["idle_gate"]=runner.idle_gate(capture,PORT,PROD_LOG,3.0,120.0)
    manifest["source_commit"]=runner.git_commit(SOURCE)
    manifest["status"]="ready-to-stop"; save(output/"manifest.json",manifest)
    stopped=False; candidate=None; primary=None
    try:
        stopped=True; runner.stop_tree(capture,prod_pid)
        manifest["status"]="v1-v2-v4"; save(output/"manifest.json",manifest)
        cmd=[sys.executable,str(HERE/"integration_spike.py"),"gpu","--output",str(output/"gpu")]
        with (output/"gpu-driver.log").open("w") as log:
            done=subprocess.run(cmd,cwd=HARNESS_ROOT,stdout=log,stderr=subprocess.STDOUT,text=True)
        manifest["gpu_returncode"]=done.returncode
        if done.returncode: raise RuntimeError(f"integration_spike GPU failed rc={done.returncode}")

        argv=runner.deterministic_candidate_argv(snapshot["argv"],TEST_PORT)
        runner.validate_candidate_command(argv)
        env=dict(prod_env); env["PYTHONPATH"]=str(SOURCE/"python")+os.pathsep+env.get("PYTHONPATH",""); env["PWD"]=env["WT"]=str(SOURCE)
        manifest["candidate_argv"]=argv; manifest["status"]="v3-launch"; save(output/"manifest.json",manifest)
        candidate=runner.launch_server(capture,argv,SOURCE,env,output/"v3-candidate-server.log",TEST_PORT,900)
        before=capture.http(TEST_PORT,"/server_info",timeout=30); save(output/"v3-server-info-before.json",before)
        samples=[]; stop=threading.Event(); thread=threading.Thread(target=nvml_sampler,args=(stop,samples),daemon=True); thread.start()
        try: request=first_token_request(output)
        finally: stop.set(); thread.join(5)
        after=capture.http(TEST_PORT,"/server_info",timeout=30); save(output/"v3-server-info-after.json",after); save(output/"v3-nvml-samples.json",samples)
        parsed=[]
        for sample in samples:
            vals=[]
            for line in sample.get("rows",[]):
                parts=[x.strip() for x in line.split(',')]
                if len(parts)==3 and int(parts[1]) in {candidate.pid,*[c.pid for c in __import__('psutil').Process(candidate.pid).children(recursive=True)]}: vals.append(int(parts[2]))
            if vals: parsed.append(sum(vals))
        v3={"status":"BLOCKED/SERVICE_DEPENDENCY","request":request,"input_path":str(INPUT),"chunked_prefill_size":before.get("chunked_prefill_size"),"nvml_sample_count":len(samples),"nvml_sample_period_target_ms":50,"nvml_process_sum_peak_mib":max(parsed) if parsed else None,"allocator_allocated_reserved_peak":"unavailable outside scheduler process","full_gpu_pool_released":False,"blocking_interface":"QSATokenToKVPool/HybridLinearKVPool owns the complete raw K/V allocation; QwenSparseAttnBackend has no QSA C4 host/hot adapter or allocator release handoff","claim":"real cold prefill and first output measured; no synthetic transfer is presented as a real handoff"}
        save(output/"v3.json",v3)
        manifest["status"]="complete"
    except BaseException as exc:
        primary=exc; manifest["status"]="FAIL"; manifest["error"]=f"{type(exc).__name__}: {exc}"
    finally:
        if candidate is not None:
            try: runner.stop_tree(capture,candidate.pid)
            except BaseException as exc: manifest["candidate_cleanup_error"]=f"{type(exc).__name__}: {exc}"
        if stopped:
            try:
                try: existing=runner.listener_pid(PORT)
                except Exception: existing=None
                if existing is None: restored=runner.launch_server(capture,snapshot["argv"],Path(snapshot["cwd"]),prod_env,PROD_LOG,PORT,900); pid=restored.pid
                else: pid=existing
                actual,_=runner.snapshot_service(pid,Path(snapshot["cwd"]))
                restore={"checked_at":runner.stamp(),"pid":pid,"health_http":200 if capture.http(PORT,"/health",timeout=10) is not None else None,"models":capture.http(PORT,"/v1/models",timeout=30),"argv_matches":actual["argv"]==snapshot["argv"],"cwd":actual["cwd"],"source_commit":actual["source_commit"]}
                if not restore["argv_matches"] or restore["source_commit"]!=snapshot["source_commit"]: raise RuntimeError(f"restore mismatch: {restore}")
                save(output/"restore-verification.json",restore); manifest["restore"]=restore
            except BaseException as exc:
                manifest["status"]="BLOCKED/RESTORE"; manifest["restore_error"]=f"{type(exc).__name__}: {exc}"; primary=primary or exc
        manifest["ended_at"]=runner.stamp(); save(output/"manifest.json",manifest)
    if primary: raise primary


if __name__=="__main__": main()
