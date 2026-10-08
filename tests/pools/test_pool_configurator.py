"""Port of the fork's added ``test_pool_configurator.py`` case (rows C01, C02).

The fork test (``test/registered/unit/model_executor/test_pool_configurator.py``
at ``ee8fe158d6``) is copied with its assertions unchanged and runs with the
plugin's C01/C02 hooks active. Its helpers are the pinned upstream test's
(unchanged by the fork). At v0.5.21 ``get_parallel().override`` validates the
whole topology, so the fork's ``override(attn_tp_size=2)`` is spelled
``override(tp_size=2, attn_tp_size=2, moe_tp_size=2)`` (upstream made the same
change to its own ``mock_cpu_env``).
"""

import os
import unittest
from unittest.mock import patch

import torch

from sglang.srt.runtime_context import get_parallel
from pools.activation import activated
from pools.pinned import make_model_runner as _make_model_runner


class TestDefaultConfigurator(unittest.TestCase):
    def setUp(self):
        context = activated("C01", "C02")
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)

    def test_qsa_p2_offload_prices_fixed_raw_staging(self):
        from sglang.srt.model_executor.pool_configurator import (
            DefaultPoolConfigurator,
        )
        from sglang.srt.runtime_context import get_context

        runner = _make_model_runner(
            self,
            num_kv_heads=1,
            head_dim=256,
            v_head_dim=256,
            num_layers=12,
            page_size=64,
            max_running_requests=4,
        )
        runner.kv_cache_dtype = torch.uint8
        hf_config = runner.model_config.hf_config
        runner.model_config.hf_text_config = hf_config
        hf_config.indexer_n_heads = 4
        hf_config.indexer_kv_heads = 1
        hf_config.indexer_head_dim = 128
        hf_config.indexer_budget = 2048
        hf_config.indexer_compress_ratio = 4

        logical = 4 * 262144
        raw_cell_size = 12 * 2 * 256
        qsa_cell_size = 12 * 128 * 2 // 4
        raw_pool_size = 262144 + 5 * 4
        ring_slots = 4 * 4
        ring_bytes = ring_slots * (128 * 2 * 12 + 3 * 8)
        fixed_bytes = (
            raw_cell_size * (raw_pool_size + 64)
            + qsa_cell_size * 64
            + ring_bytes
        )
        required_bytes = fixed_bytes + qsa_cell_size * logical

        with (
            patch.dict(os.environ, {"SGLANG_QSA_HISPARSE_V3": "p2-offload"}),
            get_context().override_server_args(
                max_running_requests=4,
                max_total_tokens=logical,
                page_size=64,
            ),
            get_parallel().override(tp_size=2, attn_tp_size=2, moe_tp_size=2),
        ):
            cfg = DefaultPoolConfigurator(runner)
            self.assertEqual(
                cfg.calculate_pool_sizes(required_bytes, 64).max_total_num_tokens,
                logical,
            )
            self.assertEqual(
                cfg.calculate_pool_sizes(
                    required_bytes - qsa_cell_size * 64, 64
                ).max_total_num_tokens,
                logical - 64,
            )

        with (
            patch.dict(os.environ, {"SGLANG_QSA_HISPARSE_V3": "p2-resident"}),
            get_parallel().override(tp_size=2, attn_tp_size=2, moe_tp_size=2),
        ):
            resident = DefaultPoolConfigurator(runner)
        self.assertEqual(resident._bias, 0)
        self.assertEqual(resident._cell_size, raw_cell_size + qsa_cell_size)
