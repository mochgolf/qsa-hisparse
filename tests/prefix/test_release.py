"""A finished request drops its pending host-prefix match.

Since v0.5.21 ``ChunkCache`` has no ``cache_finished_req``: ``release_kv_cache``
frees the request's row itself and then calls ``tree_cache.on_release``. The
fork dropped a finished request's match in its ``cache_finished_req``
override; production's ``prefix_cache.py`` (the reference since Phase 5) does
it in ``on_release``.
"""

import unittest

from prefix import test_prefix_cache as prefix_tests
from sglang.srt.mem_cache import common
from sglang.srt.runtime_context import get_context


class TestReleaseDropsPendingMatch(unittest.TestCase):
    """The fork prefix suite's fake runtime with real pools and allocators."""

    req = prefix_tests.TestRuntimeHostPrefixes.req
    source = prefix_tests.TestRuntimeHostPrefixes.source
    match = prefix_tests.TestRuntimeHostPrefixes.match

    def setUp(self):
        prefix_tests.TestRuntimeHostPrefixes.setUp(self)
        self.patches.enter_context(
            get_context().override_server_args(attention_backend="torch_native", dcp_size=1)
        )

    def test_pinned_release_kv_cache_releases_the_match_reader(self):
        self.source(64)
        req = self.req("finished", range(65))
        self.assertEqual(self.match(req), 64)
        self.assertEqual(self.a.prefix_cache.stats()["prefix_cache_readers"], 1)
        # The request holds a row, its recurrent slot and one page of KV.
        self.a.req_pool.alloc([req])
        allocator = self.a.runner.token_to_kv_pool_allocator
        logical = allocator.available_size()
        pages = allocator.alloc(64)
        self.a.req_table[req.kv.req_pool_idx, :64] = pages.int()
        req.kv.kv_allocated_len = req.kv.kv_committed_len = 64
        req.owned_kv_len = lambda: req.kv.kv_committed_len
        req.skip_radix_cache_insert = False

        common.release_kv_cache(req, self.cache, is_insert=False)

        self.assertEqual(self.a.prefix_cache.stats()["prefix_cache_readers"], 0)
        self.assertNotIn(req.cache_request_handle, self.cache.matches)
        self.assertFalse(req.kv.holds_kv)
        self.assertEqual(allocator.available_size(), logical)
        self.assertEqual(self.a.req_pool.available_size(), 2)
