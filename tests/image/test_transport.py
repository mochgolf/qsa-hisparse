"""Row I1: the full artifact key reaches the scheduler's request.

A real ``QwenVLImageProcessor`` (upstream's CPU fixture: tiny tokenizer, HF
Qwen2-VL image processor, ``qwen4_exp`` config, preprocess cache on) runs its
artifact fast path with only row I1 active (manifest restricted to I1). The
output then takes the default tokenizer-to-scheduler transport
(``TokenizedGenerateReqInput`` over msgpack, as ``sock_send``/``sock_recv``),
the pickle IPC and TP broadcast encoding, and the scheduler's
``MultimodalInputs.from_processor_output`` + ``Req.extend_image_inputs``.
"""

import asyncio
import base64
import contextlib
import hashlib
import importlib
import io
import pickle
from array import array
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from sglang.srt.managers.io_struct import (
    TokenizedGenerateReqInput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.schedule_batch import MultimodalInputs, Req
from sglang.srt.multimodal.cache.identity import build_artifact_key
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import ImageData
from sglang_qsa_hisparse import patching
from sglang_qsa_hisparse.features import Features
from sglang_qsa_hisparse.hisparse.image_identity import ImageRecord, page_digests
from sglang_qsa_hisparse.hisparse.image_request import (
    ARTIFACT_KEY,
    BYPASS,
    identity_for,
)

VISION = "<|vision_start|><|image_pad|><|vision_end|>"
PROMPT = f"hello {VISION} hello {VISION} hello {VISION} hello"


@contextlib.contextmanager
def i1_active():
    """Activate row I1 alone against its manifest entry and pinned fingerprints."""
    from sglang.srt.plugins.hook_registry import HookRegistry

    importlib.import_module("sglang_qsa_hisparse.patches.hisparse.image_identity")
    manifest = {"I1": patching.load_manifest()["I1"]}
    specs = [spec for spec in patching._declared if spec.row == "I1"]
    originals = {spec.target: patching._raw_attribute(spec.target) for spec in specs}
    saved = (
        list(patching._declared),
        list(patching._attached),
        patching.load_manifest,
        patching._import_feature_modules,
        patching._import_framework_hooks,
    )
    patching._declared[:] = specs
    patching._attached[:] = []
    patching.load_manifest = lambda: manifest
    patching._import_feature_modules = lambda feature: None
    patching._import_framework_hooks = lambda: None
    try:
        patching.activate(Features(hisparse_mode="p2-offload"))
        yield
    finally:
        for target, original in originals.items():
            owner, name = target.rsplit(".", 1)
            setattr(patching.pkgutil.resolve_name(owner), name, original)
        HookRegistry.reset()
        patching._activated = None
        patching._applied.clear()
        patching._frozen_hooks.clear()
        patching._frozen_depends.clear()
        patching._attached_live.clear()
        (
            patching._declared[:],
            patching._attached[:],
            patching.load_manifest,
            patching._import_feature_modules,
            patching._import_framework_hooks,
        ) = saved


class _Config(SimpleNamespace):
    def to_dict(self):
        return {"model_type": self.model_type, "architectures": self.architectures}


def make_processor(cache_mb):
    """Upstream's ``rust/qwen/_fixtures.make_processor``, as ``qwen4_exp``."""
    from tokenizers import Tokenizer, decoders
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen2VLProcessor,
        Qwen2VLVideoProcessor,
    )
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
        Qwen2VLImageProcessor,
    )

    from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor

    vocab = ["<unk>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>",
             "hello", "<|video_pad|>", "<pad>"]  # fmt: skip
    backend = Tokenizer(WordLevel({t: i for i, t in enumerate(vocab)}, "<unk>"))
    backend.pre_tokenizer, backend.decoder = WhitespaceSplit(), decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token=vocab[0],
        pad_token=vocab[-1],
        additional_special_tokens=vocab[1:4] + [vocab[5]],
    )
    hf_processor = Qwen2VLProcessor(
        image_processor=Qwen2VLImageProcessor(
            patch_size=14, merge_size=2, temporal_patch_size=2, min_pixels=56 * 56,
            max_pixels=28 * 28 * 1280, image_mean=[0.5] * 3, image_std=[0.5] * 3,
        ),  # fmt: skip
        video_processor=Qwen2VLVideoProcessor(),
        tokenizer=tokenizer,
    )
    hf_config = _Config(
        model_type="qwen4_exp",
        architectures=["Qwen4ExpForConditionalGeneration"],
        vision_start_token_id=1,
        image_token_id=2,
        vision_end_token_id=3,
        video_token_id=5,
        vision_config=SimpleNamespace(spatial_merge_size=2, tokens_per_second=2),
    )
    server_args = SimpleNamespace(
        model_impl="sglang", keep_mm_feature_on_device=False,
        mm_feature_transport="cpu", mm_enable_dp_encoder=False,
        image_processor_backend="auto", disable_fast_image_processor=True,
        skip_tokenizer_init=False, mm_preprocess_cache_size_mb=cache_mb,
        trust_mm_content_hashes=False, tp_size=1, dist_init_addr=None,
        mm_process_config={}, mm_io_worker_num=1, mm_processor_worker_num=1,
        tokenizer_worker_num=1, base_gpu_id=0, rl_on_policy_target=None,
        allowed_media_domains=[], media_url_max_file_size_mb=64,
    )  # fmt: skip
    publish(
        ServerArgs(
            model_path="dummy",
            model_impl="sglang",
            mm_feature_transport="cpu",
            mm_process_config={},
            allowed_media_domains=[],
            disable_fast_image_processor=True,
            mm_preprocess_cache_size_mb=cache_mb,
        ),
        role="tokenizer",
    )
    return QwenVLImageProcessor(
        hf_config, server_args, hf_processor, None, skip_mm_pool=True
    )


@pytest.fixture
def processor_factory():
    made = []

    def factory(cache_mb=16):
        made.append(make_processor(cache_mb))
        return made[-1]

    yield factory
    for processor in made:
        processor.io_executor.shutdown()
        processor.cpu_executor.shutdown()
    reset_context()


def png(width, height, seed):
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def process(processor, images):
    request = SimpleNamespace(video_data=None, audio_data=None, rid="transport")
    return asyncio.run(
        processor.process_mm_data_async(
            image_data=images, input_text=PROMPT, request_obj=request
        )
    )


def scheduler_request(output):
    """Tokenizer -> ZMQ msgpack -> scheduler ``Req``, as at the pin."""
    sent = TokenizedGenerateReqInput(
        input_text=PROMPT,
        input_ids=array("q", output.input_ids),
        input_embeds=None,
        mm_inputs=output,
        token_type_ids=None,
        sampling_params=SamplingParams(),
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
    )
    sent.wrap_pickle_fields()
    received = msgpack_decode(msgpack_encode(sent))
    received.unwrap_pickle_fields()
    ids = output.padded_input_ids or output.input_ids  # The scheduler pads later.
    req = Req("transport", PROMPT, array("q", ids), SamplingParams())
    req.extend_image_inputs(MultimodalInputs.from_processor_output(received.mm_inputs))
    return req


def test_fast_path_artifact_keys_reach_the_scheduler_request(processor_factory):
    processor = processor_factory()
    first, second = png(224, 224, 0), png(280, 168, 1)  # 64 and 60 tokens.
    # Same content as the first image, with a preprocessing option.
    detailed = ImageData(
        url="data:image/png;base64," + base64.b64encode(first).decode(),
        detail="high",
    )

    def expected_key(content, **kwargs):
        return build_artifact_key(
            "sha256:" + hashlib.sha256(content).hexdigest(),
            modality="image",
            processor_fingerprint=processor.processor_fingerprint,
            preprocess_kwargs=kwargs,
        )

    keys = [
        expected_key(first),
        expected_key(first, detail="high"),
        expected_key(second),
    ]
    assert len(set(keys)) == 3
    with i1_active():
        output = process(processor, [first, detailed, second])
        cached = process(processor, [first, detailed, second])  # Cache hits.
    for result in (output, cached):
        assert [i.model_specific_data[ARTIFACT_KEY] for i in result.mm_items] == keys
    # The preprocess cache keeps the artifacts without the request-owned key.
    assert all(
        ARTIFACT_KEY not in processor.mm_preprocess_cache.get(k).model_specific_data
        for k in keys
    )

    req = scheduler_request(output)
    identity = identity_for(req)
    assert identity.records == tuple(
        ImageRecord(
            key,
            order,
            item.offsets[0][0],
            item.offsets[0][1] + 1,
            tuple(item.model_specific_data["image_grid_thw"].flatten().tolist()),
        )
        for order, (key, item) in enumerate(zip(keys, output.mm_items))
    )
    # M-RoPE positions arrive too: three complete pages of the 198-token prompt.
    assert identity.page_digests == page_digests(output.mrope_positions, 198)
    assert len(output.input_ids) == 198 and len(identity.page_digests) == 3
    # Pickle IPC (SGLANG_USE_PICKLE_IPC) and the TP broadcast of the inputs.
    for inputs in (
        pickle.loads(pickle.dumps(output)),
        pickle.loads(pickle.dumps(req.multimodal_inputs)),
    ):
        assert [i.model_specific_data[ARTIFACT_KEY] for i in inputs.mm_items] == keys


def test_without_the_hook_or_off_the_fast_path_requests_bypass(processor_factory):
    images = [png(96, 80, 0), png(112, 88, 1), png(64, 64, 2)]
    output = process(processor_factory(), images)  # I1 not active.
    assert all(ARTIFACT_KEY not in i.model_specific_data for i in output.mm_items)
    assert identity_for(scheduler_request(output)) is BYPASS
    # qwen4_exp takes the artifact path only with a preprocess cache budget.
    uncached = processor_factory(cache_mb=0)
    with i1_active():
        output = process(uncached, images)
    assert all(ARTIFACT_KEY not in i.model_specific_data for i in output.mm_items)
    assert identity_for(scheduler_request(output)) is BYPASS
