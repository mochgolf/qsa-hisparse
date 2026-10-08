"""ViT encode observer for the Track I GPU evidence run (I-C).

Enable it in the server like observer.py: ``tools/evidence/site`` first on
``PYTHONPATH`` and ``QSA_EVIDENCE_VIT_DIR`` set to a new absolute directory.
The site hook wraps ``Qwen3VLForConditionalGeneration.get_image_feature``,
the Qwen-VL vision encoder entry that Qwen4-Exp inherits and that SGLang
calls once per ViT batch (``mm_schedule._batch_encode_per_image_misses``),
when ``sglang.srt.models.qwen3_vl`` is imported. After each call, each
process appends one line to ``<dir>/rank-<rank>.jsonl``:

    {"time": <time.time()>, "rank": <rank>,
     "items": [{"offsets": [[start, end]], "hash": <item hash>,
                "sha256": <embedding digest>}, ...]}

``offsets`` are the item's token positions in its request (end inclusive),
in batch order. An item's embedding is its rows of the returned features
(``end - start + 1`` rows per span, or the item's own tensor when the encoder
returns a list); its digest is SHA-256 over ``"<dtype>:<shape>:"`` and the raw
bytes. The wrapper adds one blocking device-to-host copy of the returned
features on the current stream and never writes a tensor.
"""

import functools
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ENV = "QSA_EVIDENCE_VIT_DIR"
MODULE = "sglang.srt.models.qwen3_vl"
_MARK = "__qsa_evidence_vit__"


def digest(tensor):
    import torch

    tensor = tensor.detach().contiguous().cpu()
    h = hashlib.sha256(f"{tensor.dtype}:{tuple(tensor.shape)}:".encode())
    h.update(tensor.reshape(-1).view(torch.uint8).numpy())
    return h.hexdigest()


def record(items, features):
    import torch

    if isinstance(features, (list, tuple)):
        rows = list(features)
    else:
        counts = [sum(end - start + 1 for start, end in item.offsets) for item in items]
        rows = torch.split(features.reshape(-1, features.shape[-1]), counts)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    row = {
        "time": time.time(),
        "rank": rank,
        "items": [
            {"offsets": [list(span) for span in item.offsets], "hash": item.hash, "sha256": digest(emb)}
            for item, emb in zip(items, rows, strict=True)
        ],
    }
    directory = Path(os.environ[ENV])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"rank-{rank}.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")


def wrap(module):
    """Wrap the Qwen-VL image encoder entry of ``module`` in place; idempotent."""
    cls = module.Qwen3VLForConditionalGeneration
    original = cls.get_image_feature
    if getattr(original, _MARK, False):
        return

    @functools.wraps(original)
    def get_image_feature(self, items):
        features = original(self, items)
        record(items, features)
        return features

    setattr(get_image_feature, _MARK, True)
    cls.get_image_feature = get_image_feature


def install():
    """Wrap the module when it is imported; called by the site hook."""
    import importlib.abc
    import importlib.machinery

    if not Path(os.environ[ENV]).is_absolute():
        raise ValueError(f"{ENV} must be an absolute directory")

    class Loader(importlib.abc.Loader):
        def __init__(self, original):
            self.original = original

        def __getattr__(self, name):
            return getattr(self.original, name)

        def create_module(self, spec):
            return self.original.create_module(spec)

        def exec_module(self, module):
            self.original.exec_module(module)
            wrap(module)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != MODULE:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is not None and spec.loader is not None:
                spec.loader = Loader(spec.loader)
            return spec

    sys.meta_path.insert(0, Finder())
    if MODULE in sys.modules:
        wrap(sys.modules[MODULE])
