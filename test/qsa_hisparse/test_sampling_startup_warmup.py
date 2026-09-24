"""The readiness warmup must exercise the non-greedy FlashInfer path."""

import unittest
from unittest.mock import Mock, patch

import requests

from sglang.srt.mem_cache.qsa_hisparse.startup_warmup import (
    wait_for_http_listener,
    warmup_non_greedy_sampling,
)


class SamplingStartupWarmupTests(unittest.TestCase):
    def setUp(self):
        listener_patch = patch(
            "sglang.srt.mem_cache.qsa_hisparse.startup_warmup.wait_for_http_listener"
        )
        self.listener = listener_patch.start()
        self.addCleanup(listener_patch.stop)

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
        self.listener.assert_called_once_with(
            url="http://127.0.0.1:8082",
            headers={"Authorization": "Bearer test-key"},
            verify=False,
        )
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


class ListenerWaitTests(unittest.TestCase):
    @patch("sglang.srt.mem_cache.qsa_hisparse.startup_warmup.time.sleep")
    @patch("sglang.srt.mem_cache.qsa_hisparse.startup_warmup.requests.get")
    def test_retries_until_listener_binds(self, get, sleep):
        get.side_effect = [requests.ConnectionError("not listening"), Mock()]

        wait_for_http_listener(
            url="http://127.0.0.1:8082",
            headers={"Authorization": "Bearer test-key"},
            verify=False,
            attempts=2,
        )

        self.assertEqual(get.call_count, 2)
        self.assertEqual(get.call_args.args, ("http://127.0.0.1:8082/model_info",))
        self.assertEqual(
            get.call_args.kwargs["headers"], {"Authorization": "Bearer test-key"}
        )
        self.assertFalse(get.call_args.kwargs["verify"])
        sleep.assert_called_once_with(1)

    @patch("sglang.srt.mem_cache.qsa_hisparse.startup_warmup.time.sleep")
    @patch("sglang.srt.mem_cache.qsa_hisparse.startup_warmup.requests.get")
    def test_exhausted_listener_wait_fails_startup(self, get, sleep):
        get.side_effect = requests.ConnectionError("not listening")

        with self.assertRaisesRegex(
            RuntimeError, "HTTP listener did not become available"
        ):
            wait_for_http_listener(
                url="http://127.0.0.1:8082", headers={}, verify=True, attempts=2
            )

        self.assertEqual(get.call_count, 2)
        sleep.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
