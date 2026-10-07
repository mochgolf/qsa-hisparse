"""Target-model scope (PLAN.md rule 9) against temporary model directories.

The reference answer for every decidable case is SGLang's own
``ModelConfig.from_server_args(...).hf_config.architectures`` built from the
same published configuration.
"""

import json

import pytest

from sglang_qsa_hisparse.errors import PluginActivationError
from sglang_qsa_hisparse.scope import (
    TARGET_ARCHITECTURES,
    ScopeUndecidable,
    target_model_active,
)

TARGET = "Qwen4ExpForConditionalGeneration"
QWEN = {"architectures": [TARGET], "model_type": "qwen4_exp"}
LLAMA = {"architectures": ["LlamaForCausalLM"], "model_type": "llama"}


@pytest.fixture(autouse=True)
def fresh_cache():
    target_model_active.cache_clear()
    yield
    target_model_active.cache_clear()


def model_dir(tmp_path, config, name="model", file="config.json"):
    directory = tmp_path / name
    directory.mkdir(exist_ok=True)
    (directory / file).write_text(json.dumps(config))
    return directory


def published(directory, **fields):
    from sglang.srt.runtime_context import get_context

    return get_context().override_server_args(model_path=str(directory), **fields)


def model_config_architectures():
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.runtime_context import get_server_args

    return ModelConfig.from_server_args(get_server_args()).hf_config.architectures


def test_target_set_is_the_fork_model():
    assert TARGET_ARCHITECTURES == {TARGET}


@pytest.mark.parametrize(
    "config, fields, expected",
    [
        (QWEN, {}, True),
        (LLAMA, {}, False),
        (LLAMA, {"json_model_override_args": json.dumps({"architectures": [TARGET]})}, True),
        (
            QWEN,
            {"json_model_override_args": json.dumps({"architectures": ["LlamaForCausalLM"]})},
            False,
        ),
    ],
    ids=["target", "other", "override-to-target", "override-to-other"],
)
def test_decision_matches_model_config(tmp_path, config, fields, expected):
    with published(model_dir(tmp_path, config), **fields):
        assert target_model_active() is expected
        assert (model_config_architectures()[0] in TARGET_ARCHITECTURES) is expected


def test_decrypted_config_file_is_read_like_model_config(tmp_path):
    directory = model_dir(tmp_path, LLAMA)
    decrypted = directory / "decrypted.json"
    decrypted.write_text(json.dumps({"architectures": [TARGET], "model_type": "llama"}))
    with published(directory, decrypted_config_file=str(decrypted)):
        assert model_config_architectures() == [TARGET]
        assert target_model_active() is True


def test_answer_is_cached_per_process(tmp_path):
    with published(model_dir(tmp_path, QWEN, "qwen")):
        assert target_model_active() is True
    with published(model_dir(tmp_path, LLAMA, "llama")):
        assert target_model_active() is True
        target_model_active.cache_clear()
        assert target_model_active() is False


def test_unpublished_configuration_is_undecidable(monkeypatch):
    from sglang.srt.runtime_context import get_context

    monkeypatch.setattr(get_context(), "_config_bags", None)
    with pytest.raises(ScopeUndecidable, match="not published"):
        target_model_active()


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "llama"},
        {"architectures": [], "model_type": "llama"},
        {"architectures": [TARGET, "LlamaForCausalLM"], "model_type": "llama"},
    ],
    ids=["no-architectures", "empty", "mixed"],
)
def test_ambiguous_architectures_are_undecidable(tmp_path, config):
    with published(model_dir(tmp_path, config)):
        with pytest.raises(ScopeUndecidable):
            target_model_active()


def test_unreadable_configuration_is_undecidable(tmp_path):
    (tmp_path / "empty").mkdir()
    with published(tmp_path / "empty"):
        with pytest.raises(ScopeUndecidable, match="Cannot read"):
            target_model_active()
    with published(model_dir(tmp_path, QWEN), json_model_override_args="{not json"):
        with pytest.raises(ScopeUndecidable, match="Cannot read"):
            target_model_active()


def test_undecidable_escapes_exception_handlers(tmp_path):
    (tmp_path / "empty").mkdir()
    with published(tmp_path / "empty"):
        with pytest.raises(PluginActivationError):
            try:
                target_model_active()
            except Exception:  # A generic hook's or SGLang loader's handler.
                pytest.fail("ScopeUndecidable was swallowed")
