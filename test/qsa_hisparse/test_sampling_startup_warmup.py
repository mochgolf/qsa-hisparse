"""The readiness warmup must exercise the non-greedy FlashInfer path."""

import unittest
from unittest.mock import Mock, patch

from sglang.srt.mem_cache.qsa_hisparse.startup_warmup import (
    warmup_non_greedy_sampling,
)


class SamplingStartupWarmupTests(unittest.TestCase):
    @patch("sglang.srt.mem_cache.qsa_hisparse.startup_warmup.requests.post")
    def test_uses_non_greedy_sampling_and_checks_output(self, post):
        post.return_value.json.return_value = {"output_ids": [42]}

        warmup_non_greedy_sampling(
            url="http://127.0.0.1:8082",
            api_key="test-key",
            verify=False,
            timeout=123,
        )

        post.return_value.raise_for_status.assert_called_once_with()
        args, kwargs = post.call_args
        self.assertEqual(args, ("http://127.0.0.1:8082/generate",))
        self.assertEqual(kwargs["headers"], {"Authorization": "Bearer test-key"})
        self.assertEqual(kwargs["timeout"], 123)
        self.assertFalse(kwargs["verify"])
        self.assertEqual(kwargs["json"]["sampling_params"], {
            "temperature": 1.0,
            "top_k": 20,
            "top_p": 0.95,
            "max_new_tokens": 1,
            "ignore_eos": True,
        })

    @patch("sglang.srt.mem_cache.qsa_hisparse.startup_warmup.requests.post")
    def test_empty_output_fails_readiness_warmup(self, post):
        post.return_value.json.return_value = {"output_ids": []}
        with self.assertRaisesRegex(RuntimeError, "produced no token"):
            warmup_non_greedy_sampling(url="http://127.0.0.1:8082")

    @patch("sglang.srt.mem_cache.qsa_hisparse.startup_warmup.requests.post")
    def test_http_failure_propagates(self, post):
        post.return_value.raise_for_status = Mock(side_effect=RuntimeError("HTTP 500"))
        with self.assertRaisesRegex(RuntimeError, "HTTP 500"):
            warmup_non_greedy_sampling(url="http://127.0.0.1:8082")


if __name__ == "__main__":
    unittest.main()
