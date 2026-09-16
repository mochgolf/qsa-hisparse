import sys
import unittest
from unittest.mock import patch

from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    _resolve_flash_attn_varlen_func,
)


class TestQwenSparseFlashAttentionResolution(unittest.TestCase):
    def test_sm89_uses_vendored_flash_attention_without_classic_fa2(self):
        _resolve_flash_attn_varlen_func.cache_clear()
        with patch.dict(sys.modules, {"flash_attn": None}), patch(
            "torch.cuda.get_device_capability", return_value=(8, 9)
        ):
            resolved = _resolve_flash_attn_varlen_func()
        self.assertEqual(
            resolved.__module__, "sglang.kernels.ops.attention.flash_attention"
        )
        _resolve_flash_attn_varlen_func.cache_clear()


if __name__ == "__main__":
    unittest.main()
