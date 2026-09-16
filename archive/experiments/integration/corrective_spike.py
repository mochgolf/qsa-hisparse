#!/usr/bin/env python3
"""Corrective CPU preflight and isolated dual-GPU QSA/HiSparse validation."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import multiprocessing as mp
import os
import queue
import statistics
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "sources/sglang-hisparse-spike-20260908"
OLD = HERE.parent / "qwen38-hisparse-tests-20260907/execution-02/r1"
LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47)
TOPK, RATIO, RAW_TOPK, HEAD = 512, 4, 2051, 256
ITEM, LOGICAL, HOT, PAGE, STEPS, DEADLINE = 2048, 65536, 2048, 64, 128, 3600


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def git_head(path: Path) -> str:
    import subprocess
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def observe_numa(tensor):
    pointer=tensor.data_ptr();maps=Path("/proc/self/maps").read_text().splitlines();vma=None
    for row in maps:
        first,last=[int(x,16) for x in row.split(None,1)[0].split("-")]
        if first<=pointer<last:vma=first;break
    line=next((x for x in Path("/proc/self/numa_maps").read_text().splitlines() if vma is not None and int(x.split(None,1)[0],16)==vma),None)
    nodes={x.split("=",1)[0]:int(x.split("=",1)[1]) for x in (line or "").split() if x.startswith("N") and "=" in x}
    return {"data_ptr":hex(pointer),"numa_maps_line":line,"node_pages":nodes}


def nr(values, fraction):
    ordered = sorted(float(x) for x in values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def stats(values):
    return {"n": len(values), "p50_ms": nr(values, .5), "p95_ms": nr(values, .95),
            "p99_ms": nr(values, .99), "max_ms": max(values), "mean_ms": statistics.fmean(values)}


def pair_rank_samples(rows):
    by = {}
    for row in rows:
        key = (row["trace"], row["request_count"], row["replay"], row["step"])
        by.setdefault(key, {})[row["rank"]] = float(row["elapsed_ms"])
    if not by or any(set(v) != {0, 1} for v in by.values()):
        raise ValueError("each sample key must contain exactly ranks 0 and 1")
    paired = [{"trace": k[0], "request_count": k[1], "replay": k[2], "step": k[3],
               "elapsed_ms": max(v.values())} for k, v in sorted(by.items())]
    return paired


# Every encoded byte is a finite E4M3 bit pattern. Four base-32 digits encode
# all logical blocks; the remaining fields occupy disjoint lanes.
def encode_identity(block, layer_index, rank, request, member, kv, payload_variant=0):
    if not 0 <= block < (1 << 20): raise ValueError(block)
    values = [(block >> (5 * i)) & 31 for i in range(4)]
    values += [32 + member, 40 + layer_index, 52 + rank, 56 + request, 68 + kv, 72 + payload_variant]
    return bytes(values)


def decode_identity(record: bytes):
    b = list(record[:10])
    block = sum(b[i] << (5 * i) for i in range(4))
    return (block, b[5] - 40, b[6] - 52, b[7] - 56, b[4] - 32, b[8] - 68, b[9] - 72)


def packed_record_bytes(block, layer_index, rank, request, variant=0):
    out = bytearray(ITEM)
    for member in range(RATIO):
        for kv in range(2):
            off = kv * 1024 + member * HEAD
            tag = encode_identity(block, layer_index, rank, request, member, kv, variant)
            out[off:off + len(tag)] = tag
            # finite, payload-dependent E4M3 values in the remaining lanes
            out[off + len(tag):off + HEAD] = bytes([0x30 + ((block + layer_index + rank + request + member + kv + variant) & 7)]) * (HEAD - len(tag))
    return bytes(out)


def expected_rows(torch, blocks, layer_index, rank, request, variant=0):
    ids=torch.as_tensor(blocks,dtype=torch.int64); out=torch.empty((len(ids),ITEM),dtype=torch.uint8); out.zero_()
    for member in range(4):
        for kv in range(2):
            off=kv*1024+member*HEAD
            for digit in range(4):out[:,off+digit]=((ids>>(5*digit))&31).to(torch.uint8)
            out[:,off+4]=32+member;out[:,off+5]=40+layer_index;out[:,off+6]=52+rank;out[:,off+7]=56+request;out[:,off+8]=68+kv;out[:,off+9]=72+variant
            fill=0x30+((ids+layer_index+rank+request+member+kv+variant)&7)
            out[:,off+10:off+HEAD]=fill[:,None].to(torch.uint8)
    return out


def decode_record(record: bytes):
    return [decode_identity(record[kv * 1024 + member * HEAD:]) for kv in range(2) for member in range(4)]


def _cpu_child(kind, barrier, out):
    try:
        barrier.wait(2)
        if kind == "fail": raise RuntimeError("injected-child-failure")
        if kind == "missing": return
        out.put((kind, "ok"))
    except BaseException:
        try: barrier.abort()
        except BaseException: pass
        if kind != "missing": out.put((kind, traceback.format_exc()))


def supervise_specs(specs, timeout=3):
    ctx = mp.get_context("spawn"); out = ctx.Queue(); barrier = ctx.Barrier(len(specs), timeout=2)
    ps = [ctx.Process(target=_cpu_child, args=(s, barrier, out)) for s in specs]
    for p in ps: p.start()
    messages=[]; deadline=time.monotonic()+timeout
    while len(messages)<len(ps) and time.monotonic()<deadline:
        try: messages.append(out.get(timeout=min(.1, max(.01, deadline-time.monotonic()))))
        except queue.Empty:
            if not any(p.is_alive() for p in ps): break
    for p in ps:
        p.join(.5)
        if p.is_alive(): p.terminate(); p.join(1)
        if p.is_alive(): p.kill(); p.join(1)
    if len(messages) != len(ps) or any("Traceback" in m[1] for m in messages):
        raise RuntimeError("bounded supervisor observed child failure/missing result")
    return messages


def cpu_check(output: Path):
    import inspect
    import torch
    # Actual packed byte recovery, including every logical-block code and all fields.
    for block in range(LOGICAL):
        rec = packed_record_bytes(block, block % len(LAYERS), block % 2, block % 8, block % 2)
        decoded = decode_record(rec)
        assert len(set(decoded)) == 8
        for kv in range(2):
            for member in range(4):
                assert (block, block % len(LAYERS), block % 2, block % 8, member, kv, block % 2) in decoded
    assert packed_record_bytes(0,0,0,0) != packed_record_bytes(17,0,0,0)
    _,_,positions=read_trace("A",0);chosen=(positions[0],positions[len(positions)//2],positions[-2],positions[-1])
    assert {((x+1)%4) for x in chosen}=={0,1,2,3}
    signature=inspect.signature(LifecycleAdapter.close)
    assert "dependency" not in signature.parameters and "producer_event" in signature.parameters
    good=torch.tensor([1],dtype=torch.uint8)
    try:validate_refetch_record(torch,torch.tensor([2],dtype=torch.uint8),good)
    except AssertionError:forced_missing_copy_rejected=True
    else:forced_missing_copy_rejected=False
    assert forced_missing_copy_rejected
    rows=[]
    for step in range(100):
        rows += [{"trace":"X","request_count":1,"replay":0,"step":step,"rank":0,"elapsed_ms":10 if step in (0,2,4,6) else 0},
                 {"trace":"X","request_count":1,"replay":0,"step":step,"rank":1,"elapsed_ms":10 if step in (1,3,5,7) else 0}]
    paired=pair_rank_samples(rows)
    assert max(nr([x["elapsed_ms"] for x in rows if x["rank"]==r],.95) for r in (0,1)) == 0
    assert nr([x["elapsed_ms"] for x in paired],.95) == 10
    restored=[]
    failures=[]
    for specs in (("ok","fail"),("ok","missing")):
        try: supervise_specs(specs)
        except RuntimeError as exc: failures.append(str(exc))
        finally: restored.append(True)
    assert len(failures)==2 and all(restored)
    result={"status":"PASS","gpu_used":False,"actual_packed_identity_records":LOGICAL,
            "paired_statistic_counterexample":{"wrong_ms":0,"correct_ms":10},
            "v1_tail_remainders":[(x+1)%4 for x in chosen],"v2_close_requires_producer_event":True,
            "v4_forced_missing_copy_checker_rejected":forced_missing_copy_rejected,
            "supervisor":{"failure_cases":2,"bounded":True,"peer_cleanup":True,"simulated_restore_hooks":2},
            "outer_deadline_s":DEADLINE,"source_commit":git_head(SOURCE)}
    save(output,result); return result


def read_trace(kind, rank):
    stem=f"r1-{kind}"; path=OLD/stem/f"{stem}-tp_rank{rank}-pp_rank0.jsonl"
    rows=[json.loads(x) for x in path.open()]
    by={(int(x["position"]),int(x["layer_id"])):x for x in rows}
    pos=sorted({int(x["position"]) for x in rows})
    if len(pos)!=767: raise ValueError("trace window")
    return str(path),by,pos


def expand_cpu(blocks, position, seq_len):
    valid=[]
    for b in blocks: valid.extend((4*b,4*b+1,4*b+2,4*b+3))
    tail_start=((position+1)//4)*4
    for x in range(tail_start,min(position+1,tail_start+3)): valid.append(x)
    return (valid+[-1]*RAW_TOPK)[:RAW_TOPK]


def fill_selected(torch, host, expanded, layer_i, rank, request, variant):
    valid=[x for x in expanded if x>=0]
    records={x//4 for x in valid}
    for block in records:
        host[block].copy_(torch.frombuffer(bytearray(packed_record_bytes(block,layer_i,rank,request,variant)),dtype=torch.uint8))


class Pool:
    def __init__(self,k,v,layer_id): self.k,self.v,self.layer_id=k,v,layer_id
    def get_key_buffer(self,layer):
        if layer!=self.layer_id:raise AssertionError(f"pool layer mismatch {layer} != {self.layer_id}")
        return self.k
    def get_value_buffer(self,layer):
        if layer!=self.layer_id:raise AssertionError(f"pool layer mismatch {layer} != {self.layer_id}")
        return self.v


def make_backend(torch, backend_mod, k, v, req_table, seq_len, layer_id):
    index_meta=SimpleNamespace()
    meta=backend_mod.QwenSparseAttnMetadata(
        sequence_lengths=seq_len, token_to_batch_idx=torch.tensor([0],dtype=torch.int32,device=k.device),
        token_slot_table=req_table, indexer_metadata=index_meta,
        row_req_pool_indices=torch.tensor([0],dtype=torch.int32,device=k.device), is_cuda_graph=True,
        fa2_valid_counts=torch.empty(1,dtype=torch.int32,device=k.device),
        fa2_cu_seqlens_k=torch.empty(2,dtype=torch.int32,device=k.device),
        fa2_cu_seqlens_q=torch.arange(2,dtype=torch.int32,device=k.device))
    backend=backend_mod.QwenSparseAttnBackend(None); backend.token_to_kv_pool=Pool(k,v,layer_id)
    backend.req_to_token_pool=SimpleNamespace(req_to_token=req_table); backend.forward_metadata=meta
    backend._cuda_graph_max_tokens=1
    return backend


def validate_storage_case(torch,got,expected,kbytes,expected_kbytes,vbytes,expected_vbytes,mapping,expected_mapping,scales,expected_scales):
    if not torch.equal(got,expected):raise AssertionError("decode output")
    if not torch.equal(kbytes,expected_kbytes):raise AssertionError("payload bytes")
    if not torch.equal(vbytes,expected_vbytes):raise AssertionError("V payload bytes")
    if not torch.equal(mapping,expected_mapping):raise AssertionError("physical mapping")
    if scales!=expected_scales:raise AssertionError("KV scales")


def v1(torch, rank, device):
    from sglang.srt.layers.attention.qsa.kernel import expand_qsa_block_indices
    import sglang.srt.layers.attention.qwen_sparse_attn_backend as bm
    branch="trtllm" if bm._resolve_trtllm_sparse_decode() is not None else f"fa2:{bm._resolve_flash_attn_varlen_func().__module__}"
    max_raw=262144; physical=RAW_TOPK+1
    graph_host=torch.empty((RAW_TOPK,2,HEAD),dtype=torch.uint8,pin_memory=True)
    graph_dev=torch.empty_like(graph_host,device=device)
    k=torch.zeros((physical,1,HEAD),dtype=torch.float8_e4m3fn,device=device); v=torch.zeros_like(k)
    kr=torch.zeros_like(k); vr=torch.zeros_like(k)
    req=torch.zeros((1,max_raw),dtype=torch.int32,device=device)
    seq=torch.empty(1,dtype=torch.int32,device=device); topk=torch.empty((1,RAW_TOPK),dtype=torch.int32,device=device)
    # 2051 is prime to 73, giving a nontrivial physical permutation.
    phys=((torch.arange(RAW_TOPK,dtype=torch.long,device=device)*73)%RAW_TOPK)+1
    q=(torch.arange(12*HEAD,device=device).reshape(1,12,HEAD).remainder(17).float()/32-.25).bfloat16()
    fb=SimpleNamespace(req_pool_indices=torch.tensor([0],dtype=torch.int32,device=device))
    checks=[]; graph_ms=[]; oracle=[]; negatives={"mapping":False,"k_bytes":False,"v_bytes":False,"k_scale":False,"v_scale":False}
    trace_data={kind:read_trace(kind,rank) for kind in ("A","B")}
    for li,lid in enumerate(LAYERS):
        layer=SimpleNamespace(layer_id=lid,tp_q_head_num=12,head_dim=HEAD,scaling=HEAD**-.5,k_scale_float=.1875,v_scale_float=.40625)
        backend=make_backend(torch,bm,k,v,req,seq,lid); resident=make_backend(torch,bm,kr,vr,req,seq,lid)
        seq.fill_(max_raw);topk.fill_(-1);topk[0,:2048]=torch.arange(2048,device=device)
        backend._forward_paged_attention(q,layer,fb,topk);torch.cuda.synchronize(device)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_dev.copy_(graph_host,non_blocking=True)
            k.view(torch.uint8).index_copy_(0,phys,graph_dev[:,0,:].reshape(RAW_TOPK,1,HEAD))
            v.view(torch.uint8).index_copy_(0,phys,graph_dev[:,1,:].reshape(RAW_TOPK,1,HEAD))
            graph_out=backend._forward_paged_attention(q,layer,fb,topk)
        for kind in ("A","B"):
            path,rows,positions=trace_data[kind]
            selected_positions=(positions[0],positions[len(positions)//2],positions[-2],positions[-1])
            if {((x+1)%4) for x in selected_positions}!={0,1,2,3}:raise AssertionError("tail remainder coverage")
            for position in selected_positions:
                row=rows[(position,lid)]; blocks=[int(x) for x in row["block_indices"]]; raw_len=position+1
                block_gpu=torch.tensor([blocks],dtype=torch.int32,device=device)
                expanded_gpu=expand_qsa_block_indices(block_gpu,torch.tensor([position],device=device),torch.tensor([raw_len],device=device),4,2048)
                expanded=expand_cpu(blocks,position,raw_len)
                if expanded_gpu.cpu().tolist()[0]!=expanded: raise AssertionError("source expansion mismatch")
                valid=[x for x in expanded if x>=0]; req.zero_(); req[0,torch.tensor(valid,device=device)]=phys[:len(valid)].to(torch.int32)
                seq.fill_(raw_len); topk.copy_(expanded_gpu)
                payload_outputs=[]
                for variant in (0,1):
                    immutable={block:packed_record_bytes(block,li,rank,0,variant) for block in {x//4 for x in valid}}
                    source=torch.empty((len(valid),2,HEAD),dtype=torch.uint8)
                    for i,x in enumerate(valid):
                        rec=immutable[x//4];member=x%4
                        source[i,0].copy_(torch.frombuffer(bytearray(rec[member*HEAD:(member+1)*HEAD]),dtype=torch.uint8))
                        source[i,1].copy_(torch.frombuffer(bytearray(rec[1024+member*HEAD:1024+(member+1)*HEAD]),dtype=torch.uint8))
                    graph_host.zero_();graph_host[:len(valid)].copy_(source)
                    resident_source=source.clone()
                    kr.zero_(); vr.zero_()
                    kr.view(torch.uint8).index_copy_(0,phys[:len(valid)],resident_source[:,0].to(device).reshape(-1,1,HEAD))
                    vr.view(torch.uint8).index_copy_(0,phys[:len(valid)],resident_source[:,1].to(device).reshape(-1,1,HEAD))
                    expected=resident._forward_paged_attention(q,layer,fb,topk).clone()
                    a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record(); graph.replay(); b.record(); b.synchronize(); graph_ms.append(a.elapsed_time(b))
                    got=graph_out.clone()
                    actual_k=k.index_select(0,phys[:len(valid)]).view(torch.uint8)[:,0,:];actual_v=v.index_select(0,phys[:len(valid)]).view(torch.uint8)[:,0,:]
                    expected_k=source[:,0].to(device);expected_v=source[:,1].to(device);actual_mapping=req[0,torch.tensor(valid,device=device)];expected_mapping=phys[:len(valid)].to(torch.int32)
                    validate_storage_case(torch,got,expected,actual_k,expected_k,actual_v,expected_v,actual_mapping,expected_mapping,(layer.k_scale_float,layer.v_scale_float),(.1875,.40625))
                    # FP64 diagnostic from immutable original bytes; it is not a qualification threshold.
                    kf=source[:,0].view(torch.float8_e4m3fn).double().to(device)*layer.k_scale_float
                    vf=source[:,1].view(torch.float8_e4m3fn).double().to(device)*layer.v_scale_float
                    qf=q[0].double(); ref=torch.softmax(qf@kf.T*(HEAD**-.5),-1)@vf
                    diff=(got[0].reshape(12,HEAD).double()-ref).abs(); nz=ref!=0; rel=(diff[nz]/ref[nz].abs()) if nz.any() else diff.new_empty(0)
                    oracle.append({"max_abs":float(diff.max()),"max_true_rel":float(rel.max()) if rel.numel() else None,"zero_reference":int((~nz).sum()),"zero_reference_nonzero_error":int(((~nz)&(diff!=0)).sum())})
                    payload_outputs.append(got)
                if torch.equal(*payload_outputs): raise AssertionError("payload mutation did not change output")
                if not all(negatives.values()):
                    cases={"mapping":(actual_k,expected_k,actual_v,expected_v,expected_mapping.flip(0),expected_mapping,(.1875,.40625),(.1875,.40625)),
                           "k_bytes":(actual_k,expected_k.flip(0),actual_v,expected_v,expected_mapping,expected_mapping,(.1875,.40625),(.1875,.40625)),
                           "v_bytes":(actual_k,expected_k,actual_v,expected_v.flip(0),expected_mapping,expected_mapping,(.1875,.40625),(.1875,.40625)),
                           "k_scale":(actual_k,expected_k,actual_v,expected_v,expected_mapping,expected_mapping,(.375,.40625),(.1875,.40625)),
                           "v_scale":(actual_k,expected_k,actual_v,expected_v,expected_mapping,expected_mapping,(.1875,.8125),(.1875,.40625))}
                    for name,(ak,ek,av,ev,am,em,asc,esc) in cases.items():
                        try:validate_storage_case(torch,payload_outputs[-1],payload_outputs[-1],ak,ek,av,ev,am,em,asc,esc)
                        except AssertionError:negatives[name]=True
                tail=[x for x in expanded[2048:] if x>=0]
                checks.append({"trace":kind,"trace_path":path,"position":position,"layer":lid,"pool_layer":backend.token_to_kv_pool.layer_id,"raw_remainder":raw_len%4,"tail":tail,"tail_count":len(tail),"raw_valid":len(valid),"raw_width":RAW_TOPK,"k_bytes":True,"v_bytes":True,"bitwise":True})
    if not all(negatives.values()):raise AssertionError(f"negative checker failed: {negatives}")
    return {"status":"PASS_STORAGE_DECODE_EQUIVALENCE","resolved_branch":branch,"resolved_backend":bm.__file__,"shape":{"q":[1,12,256],"kv_heads":1,"raw_width":RAW_TOPK},"cases":len(checks),"payloads":2,"case_metadata":checks,"graph":stats(graph_ms),"negative_checker":negatives,"fp64_diagnostics":{"status":"INDEPENDENT_KERNEL_ACCURACY_NOT_QUALIFIED","max_abs":max(x["max_abs"] for x in oracle),"max_true_rel":max(x["max_true_rel"] for x in oracle if x["max_true_rel"] is not None),"zero_reference":sum(x["zero_reference"] for x in oracle),"zero_reference_nonzero_error":sum(x["zero_reference_nonzero_error"] for x in oracle)}}


@dataclass(frozen=True)
class Lease: slot:int; generation:int; request:int


class LifecycleAdapter:
    def __init__(self,torch,device,rank,layer_i):
        self.torch=torch; self.device=device; self.rank=rank; self.layer_i=layer_i
        self.host=torch.full((2,4,ITEM),0x7f,dtype=torch.uint8,pin_memory=True); self.hot=torch.empty((2,ITEM),dtype=torch.uint8,device=device)
        self.stream=torch.cuda.Stream(device=device); self.free=[0,1]; self.gen=[0,0]; self.owner={}; self.pending={}; self.acknowledged=set(); self.expected={}; self.tail={}; self.events=[];self.refetch_events={};self.d2h_events=[];self.h2d_events=[]
    def admit(self,request):
        if not self.free: raise RuntimeError("allocator exhausted")
        s=self.free.pop(0); self.gen[s]+=1; lease=Lease(s,self.gen[s],request); self.owner[s]=lease; self.tail[request]=[]; return lease
    def guard(self,lease):
        if self.owner.get(lease.slot)!=lease: raise RuntimeError("stale lease")
    def append(self,lease,k,v): self.guard(lease); self.tail[lease.request].append((k.clone(),v.clone()))
    def close(self,lease,block,producer_event):
        self.guard(lease)
        if len(self.tail[lease.request])!=4: raise RuntimeError("tail incomplete")
        if producer_event is None: raise RuntimeError("producer event required")
        rec=self.torch.cat([self.torch.stack([x[0] for x in self.tail[lease.request]]).flatten(),self.torch.stack([x[1] for x in self.tail[lease.request]]).flatten()]).view(self.torch.uint8)
        expected=bytes(rec.cpu().tolist()); self.expected[(lease.request,block)]=expected
        with self.torch.cuda.stream(self.stream):
            self.stream.wait_event(producer_event); start=self.torch.cuda.Event(True); end=self.torch.cuda.Event(True); start.record(self.stream); self.host[lease.request,block].copy_(self.hot[lease.slot],non_blocking=True); end.record(self.stream)
        self.pending[lease]=(block,start,end);self.d2h_events.append((start,end));self.events.append("wait_event");self.tail[lease.request]=[];return end
    def poll_writeback(self,lease):
        self.guard(lease); ready=self.pending[lease][2].query()
        if ready:self.acknowledged.add(lease)
        return ready
    def evict(self,lease):
        self.guard(lease)
        if lease in self.pending and lease not in self.acknowledged: raise RuntimeError("writeback not acknowledged")
        if lease in self.pending:
            block,start,end=self.pending.pop(lease); self.acknowledged.discard(lease); end.synchronize()
            if bytes(self.host[lease.request,block].tolist())!=self.expected[(lease.request,block)]: raise AssertionError("D2H bytes")
        self.owner.pop(lease.slot); self.free.append(lease.slot)
    def release(self,lease):
        self.guard(lease)
        if lease in self.pending and lease not in self.acknowledged: raise RuntimeError("release pending")
        self.evict(lease)
    def refetch(self,lease,block):
        self.guard(lease)
        with self.torch.cuda.stream(self.stream):
            start=self.torch.cuda.Event(True);end=self.torch.cuda.Event(True);start.record(self.stream);self.hot[lease.slot].copy_(self.host[lease.request,block],non_blocking=True);end.record(self.stream)
        self.refetch_events[lease]=(block,start,end);self.h2d_events.append((start,end));return end
    def read_selected(self,lease,consumer_stream):
        self.guard(lease)
        if lease not in self.refetch_events:raise RuntimeError("refetch completion missing")
        consumer_stream.wait_event(self.refetch_events[lease][2]);self.events.append("consumer_wait_event");return self.hot[lease.slot]


def v2(torch,rank,device):
    import sglang.srt.layers.attention.qwen_sparse_attn_backend as bm
    checks=[];metrics=[]
    for li,_ in enumerate(LAYERS):
        wall_start=time.monotonic()
        exhausted=missing_dep=pending_evict=pending_release=stale=False
        a=LifecycleAdapter(torch,device,rank,li); l0=a.admit(0); l1=a.admit(1)
        try: a.admit(2)
        except RuntimeError: exhausted=True
        sentinel=a.host.clone(); delay=torch.cuda.Stream(device=device)
        with torch.cuda.stream(delay): torch.cuda._sleep(20_000_000); a.hot[l0.slot].fill_(1); producer=torch.cuda.Event(); producer.record(delay)
        # Actual API rejects before enqueue and leaves sentinel untouched.
        rec=packed_record_bytes(0,li,rank,0);incomplete=[]
        for member in range(4):
            try:a.close(l0,0,producer)
            except RuntimeError:incomplete.append(len(a.tail[0]))
            a.append(l0,torch.frombuffer(bytearray(rec[member*HEAD:(member+1)*HEAD]),dtype=torch.uint8),torch.frombuffer(bytearray(rec[1024+member*HEAD:1024+(member+1)*HEAD]),dtype=torch.uint8))
        before_state=(dict(a.pending),dict(a.expected),list(a.events),a.host.clone())
        try: a.close(l0,0,None)
        except RuntimeError: missing_dep=True
        assert missing_dep and not a.pending and not a.expected and a.events==before_state[2] and torch.equal(a.host,before_state[3]) and incomplete==[0,1,2,3]
        # Legal producer fills the exact append-derived record.
        expected=torch.cat([torch.stack([x[0] for x in a.tail[0]]).flatten(),torch.stack([x[1] for x in a.tail[0]]).flatten()]).view(torch.uint8)
        with torch.cuda.stream(delay): a.hot[l0.slot].copy_(expected.to(device)); producer2=torch.cuda.Event(); producer2.record(delay)
        event=a.close(l0,0,producer2)
        try: a.evict(l0)
        except RuntimeError: pending_evict=True
        try: a.release(l0)
        except RuntimeError: pending_release=True
        event.synchronize(); assert a.poll_writeback(l0); a.evict(l0); fresh=a.admit(0)
        try: a.read_selected(l0,torch.cuda.current_stream(device))
        except RuntimeError: stale=True
        h2d0=a.refetch(fresh,0);consumer=torch.cuda.Stream(device=device)
        with torch.cuda.stream(consumer):record0=a.read_selected(fresh,consumer);observed0=record0.clone()
        h2d0.synchronize();consumer.synchronize()
        if bytes(observed0.cpu().tolist())!=bytes(expected.tolist()):raise AssertionError("H2D bytes")
        # A consecutive second C4 closes with a newly empty tail on the same lease.
        rec_second=packed_record_bytes(1,li,rank,0,1)
        for member in range(4):a.append(fresh,torch.frombuffer(bytearray(rec_second[member*HEAD:(member+1)*HEAD]),dtype=torch.uint8),torch.frombuffer(bytearray(rec_second[1024+member*HEAD:1024+(member+1)*HEAD]),dtype=torch.uint8))
        expected_second=torch.frombuffer(bytearray(rec_second),dtype=torch.uint8)
        with torch.cuda.stream(delay):a.hot[fresh.slot].copy_(expected_second.to(device));producer_second=torch.cuda.Event();producer_second.record(delay)
        second=a.close(fresh,1,producer_second);second.synchronize();assert a.poll_writeback(fresh);a.release(fresh)
        # Same logical block, different request payload.
        rec1=packed_record_bytes(0,li,rank,1); assert rec1!=bytes(expected.tolist())
        for member in range(4):a.append(l1,torch.frombuffer(bytearray(rec1[member*HEAD:(member+1)*HEAD]),dtype=torch.uint8),torch.frombuffer(bytearray(rec1[1024+member*HEAD:1024+(member+1)*HEAD]),dtype=torch.uint8))
        expected1=torch.frombuffer(bytearray(rec1),dtype=torch.uint8)
        with torch.cuda.stream(delay):a.hot[l1.slot].copy_(expected1.to(device));producer3=torch.cuda.Event();producer3.record(delay)
        event1=a.close(l1,0,producer3);event1.synchronize();assert a.poll_writeback(l1);a.evict(l1);fresh1=a.admit(1);h2d1=a.refetch(fresh1,0)
        assert bytes(a.host[0,0].tolist())==bytes(expected.tolist()) and bytes(a.host[1,0].tolist())==rec1
        # Feed the refetched record to the same production paged-decode entry as V1.
        pk=torch.zeros((5,1,HEAD),dtype=torch.float8_e4m3fn,device=device);pv=torch.zeros_like(pk)
        with torch.cuda.stream(consumer):record=a.read_selected(fresh1,consumer);pk.view(torch.uint8)[1:].copy_(record[:1024].reshape(4,1,HEAD));pv.view(torch.uint8)[1:].copy_(record[1024:].reshape(4,1,HEAD))
        h2d1.synchronize()
        req_table=torch.tensor([[1,2,3,4]],dtype=torch.int32,device=device);seq_len=torch.tensor([4],dtype=torch.int32,device=device)
        backend=make_backend(torch,bm,pk,pv,req_table,seq_len,LAYERS[li]);top=torch.full((1,RAW_TOPK),-1,dtype=torch.int32,device=device);top[0,:4]=torch.arange(4,device=device)
        rpk=torch.zeros_like(pk);rpv=torch.zeros_like(pv);rpk.view(torch.uint8)[1:].copy_(expected1[:1024].to(device).reshape(4,1,HEAD));rpv.view(torch.uint8)[1:].copy_(expected1[1024:].to(device).reshape(4,1,HEAD));resident_backend=make_backend(torch,bm,rpk,rpv,req_table,seq_len,LAYERS[li])
        layer=SimpleNamespace(layer_id=LAYERS[li],tp_q_head_num=12,head_dim=HEAD,scaling=HEAD**-.5,k_scale_float=.1875,v_scale_float=.40625);fb=SimpleNamespace(req_pool_indices=torch.tensor([0],dtype=torch.int32,device=device));q=torch.zeros((1,12,HEAD),dtype=torch.bfloat16,device=device)
        with torch.cuda.stream(consumer):consumed=backend._forward_paged_attention(q,layer,fb,top)
        resident_consumed=resident_backend._forward_paged_attention(q,layer,fb,top);consumer.synchronize();consumer_ok=torch.equal(consumed,resident_consumed)
        checks.append({"layer":LAYERS[li],"incomplete_tail_rejected":incomplete==[0,1,2,3],"tail_cleared_after_close":a.tail[0]==[],"second_c4_closed":bytes(a.host[0,1].tolist())==rec_second,"allocator_exhaustion":exhausted,"missing_dependency_rejected":missing_dep,"host_sentinel_unchanged":True,"pending_evict_rejected":pending_evict,"pending_release_rejected":pending_release,"stale_lease_rejected":stale,"cross_request_same_logical_distinct":True,"producer_wait_event":a.events.count("wait_event")==3,"consumer_wait_event":a.events.count("consumer_wait_event")==2,"mapping_unique":len(set(req_table.cpu().tolist()[0]))==4,"selected_complete":int((top>=0).sum())==4,"scales":(layer.k_scale_float,layer.v_scale_float)==(.1875,.40625),"production_decode_matches_resident":consumer_ok})
        metrics.append({"layer":LAYERS[li],"d2h_bytes":3*ITEM,"h2d_bytes":2*ITEM,"d2h_event_ms":[float(x.elapsed_time(y)) for x,y in a.d2h_events],"h2d_event_ms":[float(x.elapsed_time(y)) for x,y in a.h2d_events],"wall_ms":(time.monotonic()-wall_start)*1000})
    if not all(all(v is True for k,v in row.items() if k!="layer") for row in checks):raise AssertionError("V2 boolean acceptance check")
    return {"status":"PASS_PROTOTYPE_CORRECTIVE","checks":checks,"metrics":metrics,"layers":len(checks),"total_d2h_bytes":sum(x["d2h_bytes"] for x in metrics),"total_h2d_bytes":sum(x["h2d_bytes"] for x in metrics)}


def reset_host(torch,host,batch,rank):
    host.zero_()
    ids=torch.arange(LOGICAL,dtype=torch.int64)
    for req in range(batch):
        for li in range(len(LAYERS)):
            view=host[req,li]
            for member in range(4):
                for kv in range(2):
                    off=kv*1024+member*HEAD
                    for digit in range(4):view[:,off+digit]=((ids>>(5*digit))&31).to(torch.uint8)
                    view[:,off+4]=32+member;view[:,off+5]=40+li;view[:,off+6]=52+rank;view[:,off+7]=56+req;view[:,off+8]=68+kv;view[:,off+9]=72
                    view[:,off+10:off+HEAD]=(0x30+((ids+li+rank+req+member+kv)&7))[:,None].to(torch.uint8)
    return host


def init_host(torch,batch,rank):
    host=torch.empty((batch,len(LAYERS),LOGICAL,ITEM),dtype=torch.uint8,pin_memory=True)
    reset_host(torch,host,batch,rank)
    return host


def validate_refetch_record(torch,got,expected):
    if not torch.equal(got,expected):raise AssertionError("refetched payload differs from append oracle")


def check_host_rows(torch,host,device_buf,device_tokens,out,topk,req,li,rank,versions):
    loc=out[req].long(); got_ids=device_tokens.view(-1).index_select(0,loc).cpu(); expected=topk[req].cpu()
    if not torch.equal(got_ids,expected):
        bad=(got_ids!=expected).nonzero().flatten();i=int(bad[0])
        raise AssertionError(f"physical mapping req={req} layer={LAYERS[li]} col={i} expected={int(expected[i])} got={int(got_ids[i])} out_loc={int(loc[i])}")
    got=device_buf.index_select(0,loc).cpu()
    exp=expected_rows(torch,expected.tolist(),li,rank,req)
    changed=[i for i,b in enumerate(expected.tolist()) if versions.get(int(b),0)==1]
    if changed:
        blocks=[int(expected[i]) for i in changed];exp[changed]=expected_rows(torch,blocks,li,rank,req,1)
    validate_refetch_record(torch,got,exp)


def prepare_v4_steps(torch,traces,batch,rank):
    prepared=[]
    for step in range(STEPS):
        layers=[]
        for li,lid in enumerate(LAYERS):
            top=torch.empty((batch,TOPK),dtype=torch.int32,pin_memory=True);seq=torch.empty(batch,dtype=torch.int32,pin_memory=True);records=[];positions=[]
            for req in range(batch):
                kind="A" if req%2==0 else "B";_,rows,pos=traces[kind];p=pos[step+4*req];row=rows[(p,lid)]
                top[req].copy_(torch.tensor(row["block_indices"],dtype=torch.int32));seq[req]=int(row["compressed_length"]);newest=int(row["compressed_length"])-1
                rec=torch.empty(ITEM,dtype=torch.uint8,pin_memory=True);rec.copy_(torch.frombuffer(bytearray(packed_record_bytes(newest,li,rank,req,1)),dtype=torch.uint8));records.append(rec);positions.append(p)
            layers.append((top,seq,records,positions))
        prepared.append(layers)
    return prepared


def v4(torch,rank,device,barrier):
    from sglang.kernels.ops.kvcache.hisparse import load_cache_to_device_buffer_mla
    traces={k:read_trace(k,rank) for k in ("A","B")}; results={}; raw_samples=[]
    for batch in (1,4,8):
        required=batch*len(LAYERS)*LOGICAL*ITEM
        host=init_host(torch,batch,rank); assert host.numel()==required
        physical=HOT+PAGE
        buffers=[torch.empty((batch*physical,ITEM),dtype=torch.uint8,device=device) for _ in LAYERS]
        tokens=[torch.full((batch,physical),-1,dtype=torch.int32,device=device) for _ in LAYERS]
        lrus=[torch.arange(HOT,dtype=torch.int16,device=device).repeat(batch,1) for _ in LAYERS]
        host_locs=[(torch.arange(LOGICAL,device=device,dtype=torch.int64)[None,:]+torch.arange(batch,device=device,dtype=torch.int64)[:,None]*(len(LAYERS)*LOGICAL)+li*LOGICAL) for li in range(len(LAYERS))]
        dev_locs=torch.arange(physical,device=device,dtype=torch.int32)[None,:]+torch.arange(batch,device=device,dtype=torch.int32)[:,None]*physical
        topk=torch.empty((batch,TOPK),dtype=torch.int32,device=device); seq=torch.empty(batch,dtype=torch.int32,device=device); reqs=torch.arange(batch,dtype=torch.int64,device=device); real=torch.tensor([batch],dtype=torch.int32,device=device)
        out=[torch.empty((batch,TOPK),dtype=torch.int32,device=device) for _ in LAYERS]
        ms=[torch.empty((batch,TOPK),dtype=torch.int64,device=device) for _ in LAYERS]; md=[torch.empty((batch,TOPK),dtype=torch.int32,device=device) for _ in LAYERS]; mc=[torch.zeros(batch,dtype=torch.int32,device=device) for _ in LAYERS]
        gathered=[torch.empty((batch,TOPK,ITEM),dtype=torch.uint8,device=device) for _ in LAYERS]
        unpack_k=[torch.empty((batch,TOPK*4,HEAD),dtype=torch.uint8,device=device) for _ in LAYERS];unpack_v=[torch.empty_like(unpack_k[0]) for _ in LAYERS]
        copy_stream=torch.cuda.Stream(device=device); scale_state=[(.1875+li/1024,.40625+li/1024) for li in range(len(LAYERS))]
        host_scales=torch.tensor([[scale_state[li] for li in range(len(LAYERS))] for _ in range(batch)],dtype=torch.float32,pin_memory=True);device_scales=host_scales.to(device)
        prepared=prepare_v4_steps(torch,traces,batch,rank)
        init_meta={"host_bytes":host.numel(),"host_formula_version":1,"tokens":-1,"lru_complete":True,"pending":0,"generation":0}
        torch.cuda.reset_peak_memory_stats(device)
        d2h_bytes=h2d_bytes=natural_refetches=0
        for replay in range(3):
            generated=[[set() for _ in LAYERS] for _ in range(batch)]
            reset_host(torch,host,batch,rank)
            for x in buffers:x.zero_()
            for x in tokens:x.fill_(-1)
            for x in lrus:x.copy_(torch.arange(HOT,dtype=torch.int16,device=device).repeat(batch,1))
            # Forced real D2H write/displace/GPU-resolver refetch correctness, then reset.
            topk.copy_(torch.arange(TOPK,dtype=torch.int32,device=device).repeat(batch,1));topk[:,-1]=LOGICAL-2;seq.fill_(LOGICAL)
            forced=0
            for req in range(batch):
                for li in range(len(LAYERS)):
                    block=LOGICAL-2; rec=torch.frombuffer(bytearray(packed_record_bytes(block,li,rank,req,1)),dtype=torch.uint8)
                    tokens[li][req].fill_(-1);lrus[li][req].copy_(torch.arange(HOT,dtype=torch.int16,device=device))
                    tail=req*physical+HOT; buffers[li][tail].copy_(rec.to(device)); tokens[li][req,HOT]=block
                    producer=torch.cuda.Event();producer.record();done=torch.cuda.Event()
                    with torch.cuda.stream(copy_stream):copy_stream.wait_event(producer);host[req,li,block].copy_(buffers[li][tail],non_blocking=True);done.record(copy_stream)
                    done.synchronize()
                    if not torch.equal(host[req,li,block],rec):raise AssertionError("forced writeback")
                    buffers[li][tail].fill_(0x55);tokens[li][req,HOT]=-1
                    load_cache_to_device_buffer_mla(topk,tokens[li],host_locs[li],dev_locs,host.view(-1,ITEM),buffers[li],out[li],reqs,seq,lrus[li],ITEM,TOPK,HOT,PAGE,1024,real,ms[li],md[li],mc[li])
                    torch.cuda.synchronize(device);count=int(mc[li][req]);src=ms[li][req,:count];expected_src=req*len(LAYERS)*LOGICAL+li*LOGICAL+block
                    matches=(src==expected_src).nonzero().flatten()
                    if count<=0 or matches.numel()!=1:raise AssertionError("forced GPU miss plan lacks generated host block")
                    plan_col=int(matches[0]);dest=int(md[li][req,plan_col]);loc=int(out[li][req,-1]);lo=req*physical
                    if loc!=dest or not lo<=loc<lo+HOT:raise AssertionError("forced refetch did not land in normal hot slot")
                    validate_refetch_record(torch,buffers[li][loc].cpu(),rec)
                    buffers[li][loc].fill_(0x55)
                    try:validate_refetch_record(torch,buffers[li][loc].cpu(),rec)
                    except AssertionError:missing_copy_rejected=True
                    else:missing_copy_rejected=False
                    if not missing_copy_rejected:raise AssertionError("missing-copy checker")
                    forced+=1
            reset_host(torch,host,batch,rank)
            for x in buffers:x.zero_()
            for x in tokens:x.fill_(-1)
            for x in lrus:x.copy_(torch.arange(HOT,dtype=torch.int16,device=device).repeat(batch,1))
            for step in range(STEPS):
                step_data=prepared[step]
                barrier.wait(300)
                start=torch.cuda.Event(True); end=torch.cuda.Event(True); start.record()
                for li,lid in enumerate(LAYERS):
                    producers=[];dones=[]
                    top_cpu,seq_cpu,records,positions=step_data[li]
                    topk.copy_(top_cpu,non_blocking=True);seq.copy_(seq_cpu,non_blocking=True)
                    for req in range(batch):
                        p=positions[req];newest=int(seq_cpu[req])-1;tail=req*physical+HOT;rec=records[req]
                        buffers[li][tail].copy_(rec,non_blocking=True);tokens[li][req,HOT]=newest
                        if (p+1)%4==0:
                            producer=torch.cuda.Event();producer.record();done=torch.cuda.Event()
                            with torch.cuda.stream(copy_stream):copy_stream.wait_event(producer);host[req,li,newest].copy_(buffers[li][tail],non_blocking=True);done.record(copy_stream)
                            producers.append(producer);dones.append(done)
                            d2h_bytes+=ITEM
                            generated[req][li].add(newest)
                    for done in dones:torch.cuda.current_stream(device).wait_event(done)
                    load_cache_to_device_buffer_mla(topk,tokens[li],host_locs[li],dev_locs,host.view(-1,ITEM),buffers[li],out[li],reqs,seq,lrus[li],ITEM,TOPK,HOT,PAGE,1024,real,ms[li],md[li],mc[li])
                    torch.index_select(buffers[li],0,out[li].reshape(-1).long(),out=gathered[li].view(-1,ITEM))
                    for member in range(4):
                        unpack_k[li][:,member::4].copy_(gathered[li][:,:,member*HEAD:(member+1)*HEAD])
                        unpack_v[li][:,member::4].copy_(gathered[li][:,:,1024+member*HEAD:1024+(member+1)*HEAD])
                end.record(); end.synchronize(); elapsed=float(start.elapsed_time(end))
                for li in range(len(LAYERS)):
                    h2d_bytes+=int(mc[li].sum().item())*ITEM
                    for req in range(batch):
                        count=int(mc[li][req].item());base=req*len(LAYERS)*LOGICAL+li*LOGICAL
                        fetched=(ms[li][req,:count]-base).cpu().tolist()
                        natural_refetches+=sum(int(x in generated[req][li]) for x in fetched)
                raw_samples.append({"trace":"mixed-A-even-B-odd","request_count":batch,"replay":replay,"step":step,"rank":rank,"elapsed_ms":elapsed})
                if step in (0,STEPS//2,STEPS-1) or any((step_data[0][3][r]+1)%4==0 or (step_data[0][3][r]+1)%64==0 for r in range(batch)):
                    for li in range(len(LAYERS)):
                        if scale_state[li] != (.1875+li/1024,.40625+li/1024) or not torch.equal(device_scales[:,li].cpu(),host_scales[:,li]):raise AssertionError("scale state")
                        for req in range(batch):
                            active=dict.fromkeys(generated[req][li],1);active[int(step_data[li][1][req])-1]=1
                            check_host_rows(torch,host,buffers[li],tokens[li],out[li],step_data[li][0],req,li,rank,active)
                            validate_refetch_record(torch,gathered[li][req].cpu(),buffers[li].index_select(0,out[li][req].long()).cpu())
                barrier.wait(300)
        vals=[x["elapsed_ms"] for x in raw_samples if x["request_count"]==batch]
        results[str(batch)]={"status":"PASS_COMPONENT_CORRECTIVE","raw_samples":len(vals),"all":stats(vals),"host_pinned_bytes":host.numel(),"host_numa":observe_numa(host),"initialization":init_meta,"forced_lifecycle_checks":batch*len(LAYERS)*3,"forced_gpu_miss_plan":True,"forced_missing_copy_rejected":True,"scale_checks":batch*len(LAYERS)*3,"d2h_bytes":d2h_bytes,"h2d_bytes":h2d_bytes,"device_peak_allocated_bytes":torch.cuda.max_memory_allocated(device),"device_peak_reserved_bytes":torch.cuda.max_memory_reserved(device),"naturally_generated_refetches":natural_refetches,"event_scope":"offline trace/payload preparation excluded; start/end enclose GPU metadata copies, producer-gated D2H, dependency waits, GPU resolver/H2D, selected-record gather and packed K/V unpack"}
        del host,host_scales,device_scales,prepared,buffers,tokens,lrus,host_locs,dev_locs,out,ms,md,mc,gathered,unpack_k,unpack_v; torch.cuda.empty_cache()
    return results,raw_samples


def gpu_worker(rank,gpu,node,outdir,barrier,result_queue,stages):
    try:
        lib=ctypes.CDLL("libnuma.so.1");lib.numa_run_on_node(node);lib.numa_set_preferred(node)
        sys.path.insert(0,str(SOURCE/"python")); import torch
        torch.cuda.set_device(gpu); device=torch.device(f"cuda:{gpu}")
        result={"rank":rank,"gpu":gpu,"numa_node":node,"source_commit":git_head(SOURCE)}
        samples=[]
        if "v1" in stages:result["v1"]=v1(torch,rank,device);save(Path(outdir)/f"rank{rank}-v1.json",result["v1"])
        if "v2" in stages:result["v2"]=v2(torch,rank,device);save(Path(outdir)/f"rank{rank}-v2.json",result["v2"])
        if "v4" in stages:result["v4"],samples=v4(torch,rank,device,barrier)
        save(Path(outdir)/f"rank{rank}.json",result); save(Path(outdir)/f"rank{rank}-samples.json",samples); result_queue.put((rank,result,samples,None))
    except BaseException:
        try: barrier.abort()
        except BaseException: pass
        result_queue.put((rank,None,None,traceback.format_exc()))


def gpu_run(output:Path,stages):
    ctx=mp.get_context("spawn"); q=ctx.Queue(); barrier=ctx.Barrier(2,timeout=300)
    ps=[ctx.Process(target=gpu_worker,args=(r,r,3-r,str(output),barrier,q,stages)) for r in range(2)]
    for p in ps:p.start()
    messages=[]; deadline=time.monotonic()+DEADLINE
    while len(messages)<2 and time.monotonic()<deadline:
        try: messages.append(q.get(timeout=min(5,max(.1,deadline-time.monotonic()))))
        except queue.Empty:
            if not any(p.is_alive() for p in ps):break
    for p in ps:
        p.join(10)
        if p.is_alive():p.terminate();p.join(5)
        if p.is_alive():p.kill();p.join(5)
    if len(messages)!=2: raise RuntimeError(f"worker missing result, exitcodes={[p.exitcode for p in ps]}")
    errors=[x[3] for x in messages if x[3]]
    if errors: raise RuntimeError("\n".join(errors))
    ranks=[x[1] for x in sorted(messages)]; samples=sum((x[2] for x in messages),[])
    summary={"status":"PASS","source_commit":git_head(SOURCE),"stages":list(stages),"ranks":ranks}
    if samples:
        paired=pair_rank_samples(samples);summary["paired"]={}
        for b in (1,4,8): summary["paired"][str(b)]=stats([x["elapsed_ms"] for x in paired if x["request_count"]==b])
        save(output/"all-rank-samples.json",samples); save(output/"paired-slower-rank-samples.json",paired)
    save(output/"gpu-summary.json",summary)


def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("cpu","gpu"));p.add_argument("--output",type=Path,required=True);p.add_argument("--stages",default="v1,v2,v4");a=p.parse_args();stages=tuple(a.stages.split(","))
    if a.mode=="cpu":cpu_check(a.output)
    else:gpu_run(a.output,stages)


if __name__=="__main__": main()
