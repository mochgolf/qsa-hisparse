"""ViT encode observer (I-C): its records, and its site hook on the pinned
Qwen-VL vision encoder entry that Qwen4-Exp inherits."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

SITE = Path(__file__).resolve().parents[2] / "tools" / "evidence" / "site"


def digest(tensor):
    tensor = tensor.contiguous()
    header = f"{tensor.dtype}:{tuple(tensor.shape)}:".encode()
    return hashlib.sha256(header + tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def test_each_encode_records_offsets_hash_and_item_embedding_digests(evidence, tmp_path, monkeypatch):
    observer = evidence("vit_observer")
    monkeypatch.setenv("QSA_EVIDENCE_VIT_DIR", str(tmp_path))
    items = [SimpleNamespace(offsets=[(8, 11)], hash=7), SimpleNamespace(offsets=[(20, 21)], hash=9)]
    features = torch.arange(18, dtype=torch.float32).reshape(6, 3).to(torch.bfloat16)
    per_item = [features[:4].reshape(2, 2, 3), features[4:]]

    class Batched:
        def get_image_feature(self, items):
            return features

    class PerItem:
        def get_image_feature(self, items):
            return per_item

    for cls in (Batched, PerItem):
        module = SimpleNamespace(Qwen3VLForConditionalGeneration=cls)
        observer.wrap(module)
        observer.wrap(module)  # Idempotent: one record per call.
    assert Batched().get_image_feature(items) is features
    assert PerItem().get_image_feature(items) is per_item
    rows = [json.loads(line) for line in (tmp_path / "rank-0.jsonl").read_text().splitlines()]
    assert [row["items"] for row in rows] == [
        [
            {"offsets": [[8, 11]], "hash": 7, "sha256": digest(features[:4])},
            {"offsets": [[20, 21]], "hash": 9, "sha256": digest(features[4:])},
        ],
        [
            {"offsets": [[8, 11]], "hash": 7, "sha256": digest(per_item[0])},
            {"offsets": [[20, 21]], "hash": 9, "sha256": digest(per_item[1])},
        ],
    ]
    assert all(row["rank"] == 0 and isinstance(row["time"], float) for row in rows)


def test_site_hook_wraps_the_pinned_qwen_vl_encoder_entry(tmp_path):
    code = (
        "import inspect\n"
        "import sglang.srt.models.qwen3_vl as m\n"
        "from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration as Q\n"
        "f = m.Qwen3VLForConditionalGeneration.get_image_feature\n"
        "print(getattr(f, '__qsa_evidence_vit__', False), Q.get_image_feature is f,\n"
        "      ','.join(inspect.signature(f).parameters), sep='|')\n"
    )
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(SITE), os.environ["PYTHONPATH"]]),
        QSA_EVIDENCE_VIT_DIR=str(tmp_path / "vit"),
    )
    env.pop("QSA_EVIDENCE_OBSERVER_DIR", None)
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[-1] == "True|True|self,items"
    assert not (tmp_path / "vit").exists()  # Nothing encoded, nothing written.

    env.pop("QSA_EVIDENCE_VIT_DIR")
    quick = "import sys; print('qsa_evidence_vit_observer' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", quick], env=env, capture_output=True, text=True)
    assert result.stdout.split() == ["False"]
