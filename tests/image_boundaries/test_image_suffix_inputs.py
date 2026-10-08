"""I3 items 2 and 5: after a host hit of length L inside or after an image,
the pinned model inputs of the suffix are those of the uncached chunked
prefill's chunk starting at L. No plugin hook is involved: the embedding
routine slices by ``extend_prefix_len`` and M-RoPE slices the request's table.

The fake ViT embeds each image as a function of that image only, so a
mismatch can only come from slicing, skipping or caching.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.managers import mm_schedule
from sglang.srt.managers.mm_utils import general_mm_embed_routine
from sglang.srt.mem_cache.multimodal_cache import MultiModalStaticCache
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

from image_boundaries.prompts import STANDARD, VOCAB, image_prompt

HIDDEN = 8
BOUNDARIES = (64, 128, 192, 256)  # Between images, inside B, at C, after all.


def vit_rows(item):
    start, end = item.offsets[0]
    rows = torch.arange((end - start + 1) * HIDDEN, dtype=torch.float32)
    return rows.reshape(-1, HIDDEN) + (item.hash % 997) * 10_000


class FakeVit:
    def __init__(self):
        self.calls = []

    def __call__(self, items):
        self.calls.append([item.artifact_key for item in items])
        return torch.cat([vit_rows(item) for item in items])


class FakeLanguageModel:
    def __init__(self):
        torch.manual_seed(0)
        self.embed = nn.Embedding(VOCAB, HIDDEN)

    def get_input_embeddings(self):
        return self.embed

    def __call__(self, input_ids, forward_batch, input_embeds, **kwargs):
        return input_embeds.clone()


@pytest.fixture(autouse=True)
def embedding_cache(monkeypatch, published_context):
    cache = MultiModalStaticCache(1 << 30)
    monkeypatch.setattr(mm_schedule, "embedding_cache", cache)
    return cache


def forward_batch(prompt, start, end):
    return ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=torch.tensor(prompt.ids[start:end]),
        req_pool_indices=torch.zeros(1, dtype=torch.int64),
        seq_lens=torch.tensor([end]),
        out_cache_loc=torch.arange(end - start),
        seq_lens_sum=end,
        seq_lens_cpu=torch.tensor([end]),
        mm_inputs=[prompt.mm],
        extend_prefix_lens_cpu=[start],
        extend_seq_lens_cpu=[end - start],
    )


def embed(prompt, start, end, vit, lm=None):
    """Language-model input embeddings of the extend chunk ``[start, end)``."""
    batch = forward_batch(prompt, start, end)
    with torch.no_grad():
        embeds = general_mm_embed_routine(
            input_ids=batch.input_ids,
            forward_batch=batch,
            language_model=lm or FakeLanguageModel(),
            multimodal_model=SimpleNamespace(get_image_feature=vit),
        )
    # PLE's n-gram reads forward_batch.input_ids, which the routine clamped.
    assert int(batch.input_ids.max()) < VOCAB
    return embeds


def reference(prompt, start, end):
    """Independent rows: ViT rows of the covering image, else the token row."""
    lm = FakeLanguageModel()
    items = {item.artifact_key: item for item in prompt.mm.mm_items}
    rows = []
    for position in range(start, end):
        image = [s for s in prompt.spans if s[1] <= position < s[2]]
        if image:
            key, image_start = image[0][:2]
            rows.append(vit_rows(items[key])[position - image_start])
        else:
            rows.append(lm.embed.weight[prompt.ids[position]].detach())
    return torch.stack(rows)


@pytest.mark.parametrize("hit", BOUNDARIES)
def test_hit_suffix_embedding_is_the_uncached_chunk_at_the_same_boundary(
    hit, embedding_cache, monkeypatch
):
    prompt = image_prompt(300)
    control_vit = FakeVit()
    embed(prompt, 0, hit, control_vit)
    control = embed(prompt, hit, prompt.length, control_vit)

    # The hit request runs only [hit, end) and its ViT cache is cold.
    monkeypatch.setattr(mm_schedule, "embedding_cache", MultiModalStaticCache(1 << 30))
    warm_prompt, warm_vit = image_prompt(300), FakeVit()
    warm = embed(warm_prompt, hit, warm_prompt.length, warm_vit)

    assert torch.equal(warm, control)
    assert torch.equal(warm, reference(prompt, hit, prompt.length))
    # Images that end before the hit are skipped; one straddling it is
    # encoded whole (vit_rows covers its full span) and sliced at the hit.
    reaching = [key for key, _, stop, _ in prompt.spans if stop > hit]
    assert warm_vit.calls == ([reaching] if reaching else [])


def test_hit_reuses_cached_images_and_encodes_only_new_suffix_images(monkeypatch):
    hit = 128
    cold = image_prompt(300)
    embed(cold, 0, hit, FakeVit())
    embed(cold, hit, cold.length, FakeVit())

    # Same images: B (straddling) and C come from the per-image cache.
    warm_vit = FakeVit()
    warm = embed(image_prompt(300), hit, 300, warm_vit)
    assert warm_vit.calls == []
    assert torch.equal(warm, reference(cold, hit, 300))

    # A different image after the hit is the only ViT work.
    images = STANDARD[:2] + (("C2", 192, (1, 12, 12)),)
    divergent_vit = FakeVit()
    divergent = embed(image_prompt(300, images), hit, 300, divergent_vit)
    assert divergent_vit.calls == [["C2"]]
    monkeypatch.setattr(mm_schedule, "embedding_cache", MultiModalStaticCache(1 << 30))
    control_prompt, control_vit = image_prompt(300, images), FakeVit()
    embed(control_prompt, 0, hit, control_vit)
    assert torch.equal(divergent, embed(control_prompt, hit, 300, control_vit))
    assert torch.equal(divergent, reference(control_prompt, hit, 300))


def mrope_extend(prompt, prefix):
    batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=torch.tensor(prompt.ids[prefix:]),
        req_pool_indices=torch.zeros(1, dtype=torch.int64),
        seq_lens=torch.tensor([prompt.length]),
        out_cache_loc=torch.arange(prompt.length - prefix),
        seq_lens_sum=prompt.length,
        seq_lens_cpu=torch.tensor([prompt.length]),
    )
    schedule = SimpleNamespace(
        multimodal_inputs=[prompt.mm],
        prefix_lens=[prefix],
        extend_lens=[prompt.length - prefix],
        reqs=[SimpleNamespace(session=None)],
    )
    batch._compute_mrope_positions(SimpleNamespace(device="cpu"), schedule)
    return batch.mrope_positions


def reference_mrope(prompt):
    """Text advances every axis by one; an image is its (t, h, w) grid."""
    columns, next_position, cursor = [], 0, 0
    for _, start, stop, (t, h, w) in prompt.spans:
        for _ in range(cursor, start):
            columns.append((next_position,) * 3)
            next_position += 1
        grid_h, grid_w = h // 2, w // 2
        for offset in range(stop - start):
            i, rest = divmod(offset, grid_h * grid_w)
            columns.append(
                (next_position + i, next_position + rest // grid_w, next_position + rest % grid_w)
            )
        next_position += max(t, grid_h, grid_w)
        cursor = stop
    for _ in range(cursor, prompt.length):
        columns.append((next_position,) * 3)
        next_position += 1
    return torch.tensor(columns).T


@pytest.mark.parametrize("hit", BOUNDARIES)
def test_hit_suffix_mrope_positions_are_the_requests_own_slice(hit):
    prompt = image_prompt(300)
    positions = mrope_extend(prompt, hit)
    assert torch.equal(positions, prompt.mm.mrope_positions[:, hit:])
    assert torch.equal(positions, reference_mrope(prompt)[:, hit:])
    if hit == 128:  # Inside B (tokens 90..189, a 10 x 10 grid after merge).
        row, column = divmod(hit - 90, 10)
        base = int(prompt.mm.mrope_positions[0, 90])
        assert positions[:, 0].tolist() == [base, base + row, base + column]
