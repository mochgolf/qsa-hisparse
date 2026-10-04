import pytest
import torch

from sglang.kernels.ops.attention.fast_topk import fast_topk
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _check_topk_values(score, lengths, indices, topk, row_starts):
    """fast_topk leaves order and tie-breaking unspecified,
    so compare the sorted top-k values rather than index sets."""
    for b in range(score.shape[0]):
        start = int(row_starts[b]) if row_starts is not None else 0
        length = int(lengths[b])
        section = score[b, start : start + length]
        row = indices[b]
        if length <= topk:
            # naive path: identity indices, then -1 fill
            assert torch.equal(
                row[:length].cpu(), torch.arange(length, dtype=torch.int32)
            )
            assert (row[length:] == -1).all()
            continue
        assert (row >= 0).all(), "long rows must fill every slot"
        assert (row < length).all(), "indices must be relative to the valid row"
        assert row.unique().numel() == topk, "duplicate selected indices"
        picked = section[row.long()]
        expected = torch.topk(section, topk).values
        assert torch.equal(
            picked.sort(descending=True).values, expected.sort(descending=True).values
        ), f"row {b}: top-{topk} value multiset mismatch"


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize(
    "batch,length",
    [
        (1, 4096),
        (7, 3000),
        (33, 32768),
        (128, 2050),
    ],
)
def test_fast_topk_long_rows(topk, batch, length):
    torch.manual_seed(0)
    score = torch.randn(batch, length, dtype=torch.float32, device="cuda")
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")

    indices = fast_topk(score, lengths, topk)
    _check_topk_values(score, lengths, indices, topk, None)


@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_short_and_mixed_rows(topk):
    torch.manual_seed(0)
    max_len = topk + 128
    batch = 8
    score = torch.randn(batch, max_len, dtype=torch.float32, device="cuda")
    # rows shorter than k (naive path), exactly k, and longer than k
    lens = [1, topk // 3, topk - 1, topk, topk + 1, topk + 7, 17, max_len]
    lengths = torch.tensor(lens[:batch], dtype=torch.int32, device="cuda")

    indices = fast_topk(score, lengths, topk)
    _check_topk_values(score, lengths, indices, topk, None)


@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_ragged_with_row_starts(topk):
    torch.manual_seed(0)
    batch, width = 16, 8192
    score = torch.randn(batch, width, dtype=torch.float32, device="cuda")
    row_starts = torch.randint(0, 2048, (batch,), dtype=torch.int32, device="cuda")
    lengths = torch.randint(1, 2048, (batch,), dtype=torch.int32, device="cuda")
    lengths = torch.minimum(lengths, width - row_starts).to(torch.int32)
    # ensure some rows are longer than k
    lengths[0] = min(width - int(row_starts[0]), topk + 100)

    indices = fast_topk(score, lengths, topk, row_starts=row_starts)
    _check_topk_values(score, lengths, indices, topk, row_starts)


@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_row_stride(topk):
    torch.manual_seed(0)
    batch, length = 8, 4096
    base = torch.randn(batch, 2 * length, dtype=torch.float32, device="cuda")
    score = base[:, :length]  # stride(0) == 2*length, stride(1) == 1
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")

    indices = fast_topk(score, lengths, topk)
    _check_topk_values(score, lengths, indices, topk, None)


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize(
    "fill",
    [
        "binary",  # only 0s and 1s: extreme duplication at the threshold bin
        "few_levels",  # a handful of distinct levels incl. negatives
        "constant",  # whole rows of one value
    ],
)
def test_fast_topk_duplicate_heavy(topk, fill):
    torch.manual_seed(0)
    batch, length = 16, 8192
    if fill == "binary":
        score = torch.randint(0, 2, (batch, length), dtype=torch.float32, device="cuda")
    elif fill == "few_levels":
        levels = torch.tensor([-5.0, -1.0, 0.0, 0.5, 2.0], device="cuda")
        score = levels[torch.randint(0, 5, (batch, length), device="cuda")]
    else:
        score = torch.full((batch, length), 3.25, dtype=torch.float32, device="cuda")
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")

    indices = fast_topk(score, lengths, topk)
    _check_topk_values(score, lengths, indices, topk, None)


@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_negative_and_zero(topk):
    torch.manual_seed(0)
    batch, length = 8, 16384
    score = torch.randn(batch, length, dtype=torch.float32, device="cuda") * 100
    score[:, : length // 3] = 0.0  # long zero prefix
    score[:, length // 3 : length // 2] = -1e30  # very negative block
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")

    indices = fast_topk(score, lengths, topk)
    _check_topk_values(score, lengths, indices, topk, None)


def _overflow_fixture(topk, sign):
    # Distinct fp32 scores share their fp16 coarse bin and first two exact-key
    # bytes, so refinement must retain >4096 candidates across several rounds.
    lens = [4095, 4096, 4097, 16384, topk + 1, topk, 17, 0]
    starts = [17, 31, 5, 63, 7, 23, 11, 19]
    width = max(lens) + max(starts) + 32
    base = torch.full((len(lens), width + 37), 99.0, device="cuda")
    score = base[:, :width]  # padded row stride; sentinels must not be selected
    for b, (length, start) in enumerate(zip(lens, starts)):
        values = sign * (1.0 + torch.arange(length, device="cuda") / 2**23)
        # A fixed permutation, with maxima spread beyond the candidate cache.
        order = (torch.arange(length, device="cuda") * 4051) % max(length, 1)
        score[b, start : start + length] = values[order]
    lengths = torch.tensor(lens, dtype=torch.int32, device="cuda")
    row_starts = torch.tensor(starts, dtype=torch.int32, device="cuda")
    for b in range(4):
        section = score[b, starts[b] : starts[b] + lens[b]]
        assert section.unique().numel() == lens[b]
        assert (
            section.half().view(torch.int16).bitwise_right_shift(8).unique().numel()
            == 1
        )
        assert section.view(torch.int32).bitwise_right_shift(16).unique().numel() == 1
    return score, lengths, row_starts


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_fast_topk_candidate_overflow(topk, sign):
    """Discarding same-bin candidates silently loses larger distinct scores."""
    score, lengths, starts = _overflow_fixture(topk, sign)
    indices = fast_topk(score, lengths, topk, row_starts=starts)
    _check_topk_values(score, lengths, indices, topk, starts)


@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_overflow_graph_replay(topk):
    """Replay must use current scores and lengths, including overflow changes."""
    score, lengths, starts = _overflow_fixture(topk, 1.0)
    # Compile outside capture, then warm up on a separate stream.
    fast_topk(score, lengths, topk, row_starts=starts)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fast_topk(score, lengths, topk, row_starts=starts)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        indices = fast_topk(score, lengths, topk, row_starts=starts)
    for replay in range(3):
        if replay:
            score.neg_()
            lengths[3] = 4096 if replay == 1 else 16384
            lengths[7] = 17 if replay == 1 else 0
        graph.replay()
        _check_topk_values(score, lengths, indices, topk, starts)


def test_fast_topk_unsupported_k():
    score = torch.randn(2, 4096, dtype=torch.float32, device="cuda")
    lengths = torch.full((2,), 4096, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="topk"):
        fast_topk(score, lengths, 1024)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
