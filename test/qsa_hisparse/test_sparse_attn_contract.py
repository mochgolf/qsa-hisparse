import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.qsa import sparse_attn


class _KernelCall:
    def __getitem__(self, grid):
        self.grid = grid
        return self

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class TestPackedDecodeKernelContract(unittest.TestCase):
    def test_bf16_scratch_uses_extended_chunk_kernel_signature(self):
        kernel = _KernelCall()
        q = torch.empty((2, 4, 16), dtype=torch.bfloat16)
        k = torch.empty((6, 2, 16), dtype=torch.bfloat16)
        v = torch.empty_like(k)
        indices = torch.empty((2, 3), dtype=torch.int32)
        cu_q = torch.tensor([0, 1, 2], dtype=torch.int32)
        cu_k = torch.tensor([0, 3, 6], dtype=torch.int32)
        kv_lens = torch.tensor([3, 3], dtype=torch.int32)

        with (
            patch.object(sparse_attn, "_get_best_config", return_value=(16, 1, 2)),
            patch.object(sparse_attn, "_sparse_gqa_chunk_prefill", kernel),
        ):
            output = sparse_attn.sparse_gqa_packed_decode_triton(
                q, k, v, indices, cu_q, cu_k, kv_lens, 0.125
            )

        self.assertEqual(output.shape, q.shape)
        self.assertEqual(kernel.grid, (1, 4))
        self.assertEqual(kernel.args[8:12], (0.125, 1.0, 1.0, 3))
        self.assertIs(kernel.kwargs["KV_IS_FP8"], False)

    def test_fp8_inputs_pass_scales_and_dtype_flag(self):
        kernel = _KernelCall()
        q = torch.empty((1, 4, 16), dtype=torch.bfloat16)
        k = torch.empty((3, 2, 16), dtype=torch.float8_e4m3fn)
        v = torch.empty_like(k)
        indices = torch.empty((1, 3), dtype=torch.int32)
        cu_q = torch.tensor([0, 1], dtype=torch.int32)
        cu_k = torch.tensor([0, 3], dtype=torch.int32)
        kv_lens = torch.tensor([3], dtype=torch.int32)

        with (
            patch.object(sparse_attn, "_get_best_config", return_value=(16, 1, 2)),
            patch.object(sparse_attn, "_sparse_gqa_chunk_prefill", kernel),
        ):
            sparse_attn.sparse_gqa_packed_decode_triton(
                q, k, v, indices, cu_q, cu_k, kv_lens, 0.125, 2.0, 3.0
            )

        self.assertEqual(kernel.args[8:12], (0.125, 2.0, 3.0, 3))
        self.assertIs(kernel.kwargs["KV_IS_FP8"], True)


if __name__ == "__main__":
    unittest.main()
