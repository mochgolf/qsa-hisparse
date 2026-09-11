from types import SimpleNamespace

from sglang.srt.models import qwen4_exp


def test_deterministic_inference_uses_stable_hc(monkeypatch):
    config = SimpleNamespace(
        deterministic=SimpleNamespace(enable_deterministic_inference=True)
    )
    monkeypatch.setattr(qwen4_exp, "get_exec", lambda: config)
    assert qwen4_exp._stable_hc()
