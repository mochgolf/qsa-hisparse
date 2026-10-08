"""Deviation D5: rebuilt HiSparse decode batches use each request's own
token-id logprob list instead of every prompt token id."""

from types import SimpleNamespace

from sglang_qsa_hisparse import scope
from sglang_qsa_hisparse.patches.model_compat import scheduler as hook


def _req(token_ids_logprob, prompt):
    return SimpleNamespace(
        multimodal_inputs=None,
        origin_input_ids=prompt,
        logprob=SimpleNamespace(token_ids_logprob=token_ids_logprob),
    )


def _batch(return_logprob, reqs):
    # As the pinned body leaves it: every prompt token id when logprobs are on.
    ids = [list(r.origin_input_ids) for r in reqs] if return_logprob else None
    return SimpleNamespace(return_logprob=return_logprob, token_ids_logprobs=ids)


def test_requests_own_token_ids_logprob_replace_prompt_ids(monkeypatch):
    monkeypatch.setattr(scope, "target_model_active", lambda: True)
    image_pad = 1_000_000_007  # Above any vocabulary: the pinned gather crashes.
    reqs = [_req(None, [1, image_pad, 2]), _req([5, 7], [3, 4])]
    batch = hook._carry_multimodal_inputs(_batch(True, reqs), None, reqs)
    assert batch.token_ids_logprobs == [None, [5, 7]]
    assert batch.multimodal_inputs == [None, None]


def test_batches_without_logprobs_are_unchanged(monkeypatch):
    monkeypatch.setattr(scope, "target_model_active", lambda: True)
    reqs = [_req(None, [1, 2])]
    batch = hook._carry_multimodal_inputs(_batch(False, reqs), None, reqs)
    assert batch.token_ids_logprobs is None


def test_non_target_models_keep_the_pinned_batch(monkeypatch):
    monkeypatch.setattr(scope, "target_model_active", lambda: False)
    reqs = [_req(None, [1, 2])]
    batch = _batch(True, reqs)
    assert hook._carry_multimodal_inputs(batch, None, reqs) is None
    assert batch.token_ids_logprobs == [[1, 2]]
