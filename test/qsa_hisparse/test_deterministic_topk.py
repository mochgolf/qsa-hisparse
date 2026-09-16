"""Exact deterministic selection contract, independent of CUDA collectors.

Contract: preserve the input score values, select largest scores with smaller
logical indices at ties, emit ascending logical indices and then -1 padding.
The Python sorting oracle does not call Torch/FlashInfer top-k or argsort.
Actual CUDA sort and graph evidence comes from the separate GPU probe.
"""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import torch

from sglang.srt.layers.attention.qsa.kernel import qsa_fast_topk
from sglang.srt.layers.attention.qsa.metadata import QSAIndexerMetadata
from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer


def oracle(scores, starts, ends, topk):
    rows = []
    for score, start, end in zip(scores.tolist(), starts.tolist(), ends.tolist()):
        values = score[start:end]
        ranked = sorted(range(len(values)), key=lambda index: (-values[index], index))
        selected = sorted(ranked[:topk])
        rows.append(selected + [-1] * (topk - len(selected)))
    return torch.tensor(rows, dtype=torch.int32).reshape(len(rows), topk)


class TestDeterministicQSATopK(unittest.TestCase):
    def test_exact_score_set_ties_short_and_threshold_rows(self):
        for topk in (512, 2048):
            width = topk + 37
            lengths = [0, 1, topk - 1, topk, topk + 1, width - 11]
            starts = torch.tensor([0, 7, 11, 3, 5, 11], dtype=torch.int32)
            ends = starts + torch.tensor(lengths, dtype=torch.int32)
            for mode in ("untied", "constant", "few_levels"):
                scores = torch.full((6, width + 11), float("nan"))
                for row, (start, length) in enumerate(zip(starts.tolist(), lengths)):
                    values = [
                        float(index)
                        if mode == "untied"
                        else 0.0
                        if mode == "constant"
                        else float((index * 11 + row) % 7 - 3)
                        for index in range(length)
                    ]
                    scores[row, start : start + length] = torch.tensor(values)
                with self.subTest(topk=topk, mode=mode):
                    actual = qsa_fast_topk(
                        scores, starts, ends, topk, deterministic=True
                    )
                    self.assertTrue(
                        torch.equal(actual, oracle(scores, starts, ends, topk))
                    )

    def test_equal_prefix_ignores_score_width_and_row_stride(self):
        starts = torch.tensor([13, 29], dtype=torch.int32)
        ends = starts + 513
        base = torch.full((2, 4096), float("nan"))
        for row, start in enumerate(starts.tolist()):
            base[row, start : start + 513] = torch.arange(513).remainder(3)
        narrow = base[:, :1024]
        wide = base[:, :2048].contiguous()
        expected = oracle(base, starts, ends, 512)
        self.assertTrue(
            torch.equal(qsa_fast_topk(narrow, starts, ends, 512, True), expected)
        )
        self.assertTrue(
            torch.equal(qsa_fast_topk(wide, starts, ends, 512, True), expected)
        )
        for shape in ((0, 513), (3, 0)):
            output = qsa_fast_topk(
                torch.empty(shape),
                torch.zeros(shape[0]),
                torch.zeros(shape[0]),
                512,
                True,
            )
            self.assertEqual(output.shape, (shape[0], 512))
            self.assertTrue(bool((output == -1).all()))

    def test_cuda_route_uses_bounded_static_tiles_without_device_scalar_reads(self):
        width, rows = 1031, 9
        scores = torch.arange(width).remainder(7).float().repeat(rows, 1)
        starts = torch.tensor([0, 3, 7, 13, 0, 5, 1, 9, 11], dtype=torch.int32)
        ends = torch.minimum(starts + 513, torch.full_like(starts, width))
        expected = oracle(scores, starts, ends, 512)
        tile_budget = 3 * width * 64
        with (
            patch.dict(sys.modules, {"flashinfer": None}),
            patch(
                "sglang.srt.layers.attention.qsa.kernel._QSA_DETERMINISTIC_TOPK_TILE_BYTES",
                tile_budget,
            ),
            patch.object(torch, "argsort", wraps=torch.argsort) as sorts,
            patch.object(
                torch.Tensor, "item", side_effect=AssertionError("device scalar read")
            ),
            patch.object(
                torch.Tensor, "tolist", side_effect=AssertionError("device host copy")
            ),
            patch.object(
                torch.Tensor,
                "__int__",
                side_effect=AssertionError("device scalar read"),
            ),
            patch.object(
                torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True
            ),
        ):
            actual = qsa_fast_topk(scores, starts, ends, 512, deterministic=True)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(sorts.call_count, 3)
        for call in sorts.call_args_list:
            self.assertLessEqual(call.args[0].numel() * 64, tile_budget)
            self.assertTrue(call.kwargs["stable"])
            self.assertTrue(call.kwargs["descending"])
        from sglang.srt.layers.attention.qsa.kernel import (
            _qsa_deterministic_topk_tile_rows,
        )

        self.assertEqual(_qsa_deterministic_topk_tile_rows(2048, 65536), 4)

    def test_negative_infinity_inside_interval_precedes_invalid_padding(self):
        scores = torch.tensor(
            [[float("nan"), -float("inf"), -float("inf"), -float("inf"), float("nan")]]
        )
        starts, ends = torch.tensor([1]), torch.tensor([4])
        for topk in (2, 7):
            actual = qsa_fast_topk(scores, starts, ends, topk, True)
            self.assertTrue(torch.equal(actual, oracle(scores, starts, ends, topk)))

    def test_native_cuda_keeps_jit_collector_output_unchanged(self):
        expected = torch.arange(512, dtype=torch.int32).flip(0).reshape(1, 512)
        with (
            patch(
                "sglang.kernels.ops.elementwise.fast_topk.fast_topk",
                return_value=expected,
            ) as native,
            patch.object(
                torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True
            ),
        ):
            actual = qsa_fast_topk(
                torch.zeros((1, 513)), torch.tensor([0]), torch.tensor([513]), 512
            )
        self.assertIs(actual, expected)
        native.assert_called_once()

    def test_prefill_decode_and_metadata_propagate_runtime_deterministic_flag(self):
        indexer = QSAIndexer.__new__(QSAIndexer)
        torch.nn.Module.__init__(indexer)
        indexer.token_topk, indexer.block_topk, indexer.compress_ratio = 2048, 512, 4
        indices = torch.full((1, 512), -1, dtype=torch.int32)
        q = torch.zeros((1, 1, 16))
        vector = torch.tensor([0], dtype=torch.int32)
        config = SimpleNamespace(
            deterministic=SimpleNamespace(enable_deterministic_inference=True)
        )
        with (
            patch(
                "sglang.srt.layers.attention.qsa.qsa_indexer.get_context",
                return_value=SimpleNamespace(
                    is_config_namespace_published=lambda name: True
                ),
            ),
            patch.object(
                torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True
            ),
            patch(
                "sglang.srt.layers.attention.qsa.qsa_indexer.expand_qsa_block_indices",
                return_value=torch.full((1, 2051), -1, dtype=torch.int32),
            ),
            patch(
                "sglang.kernels.ops.elementwise.fast_topk.fast_topk",
                side_effect=AssertionError(
                    "native collector reached during deterministic decode"
                ),
            ),
            patch(
                "sglang.srt.layers.attention.qsa.qsa_indexer.get_exec",
                return_value=config,
            ),
            patch(
                "sglang.srt.layers.attention.qsa.qsa_indexer.qsa_mqa_prefill",
                return_value=torch.zeros((1, 513)),
            ),
            patch(
                "sglang.srt.layers.attention.qsa.qsa_indexer.qsa_mqa_decode",
                return_value=torch.zeros((1, 513)),
            ),
            patch(
                "sglang.srt.layers.attention.qsa.qsa_indexer.qsa_fast_topk",
                return_value=indices,
            ) as selector,
        ):
            indexer.select_prefill_tokens(
                q,
                torch.zeros((513, 1, 16)),
                vector,
                vector + 513,
                vector,
                vector + 2052,
            )
            indexer.select_decode_tokens(
                q,
                torch.zeros((513, 1, 16)),
                vector.reshape(1, 1),
                vector + 513,
                2052,
                vector,
                vector + 2052,
            )
            self.assertEqual(selector.call_count, 2)
            self.assertTrue(
                all(call.kwargs["deterministic"] for call in selector.call_args_list)
            )
        metadata = QSAIndexerMetadata(
            vector, vector, vector.reshape(1, 1), vector, None, 4, 512
        )
        with (
            patch(
                "sglang.srt.layers.attention.qsa.metadata.get_context",
                return_value=SimpleNamespace(
                    is_config_namespace_published=lambda name: True
                ),
            ),
            patch(
                "sglang.srt.layers.attention.qsa.metadata.get_exec", return_value=config
            ),
            patch(
                "sglang.srt.layers.attention.qsa.metadata.qsa_fast_topk",
                return_value=indices,
            ) as selector,
        ):
            metadata.topk_transform(torch.zeros((1, 513)), 512, vector, vector + 513)
            self.assertTrue(selector.call_args.kwargs["deterministic"])

    def test_unpublished_standalone_metadata_keeps_native_reference_selection(self):
        vector = torch.tensor([0], dtype=torch.int32)
        metadata = QSAIndexerMetadata(
            vector, vector, vector.reshape(1, 1), vector, None, 4, 512
        )
        with (
            patch(
                "sglang.srt.layers.attention.qsa.metadata.get_context",
                return_value=SimpleNamespace(
                    is_config_namespace_published=lambda name: False
                ),
            ),
            patch(
                "sglang.srt.layers.attention.qsa.metadata.get_exec",
                side_effect=AssertionError("unpublished exec config must not be read"),
            ),
        ):
            output = metadata.topk_transform(
                torch.tensor([[1.0, 3.0, 2.0]]), 512, vector, vector + 3
            )
        self.assertEqual(output[0, :3].tolist(), [1, 2, 0])
        self.assertTrue(bool((output[0, 3:] == -1).all()))


if __name__ == "__main__":
    unittest.main()
