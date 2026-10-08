"""Row U02: the attached field behaves like production's ``Envs`` field."""

import pytest

from sglang.srt.environ import Envs, envs, exportable_env_vars

pytestmark = pytest.mark.usefixtures("numa_rows")


def test_interleave_field_is_read_and_exported_like_a_declared_field(monkeypatch):
    monkeypatch.delenv("SGLANG_NUMA_INTERLEAVE", raising=False)
    assert envs.SGLANG_NUMA_INTERLEAVE.get() is False
    assert "SGLANG_NUMA_INTERLEAVE" not in exportable_env_vars()
    monkeypatch.setenv("SGLANG_NUMA_INTERLEAVE", "1")
    assert envs.SGLANG_NUMA_INTERLEAVE.get() is True
    assert exportable_env_vars()["SGLANG_NUMA_INTERLEAVE"] == "1"
    assert vars(Envs)["SGLANG_NUMA_INTERLEAVE"].name == "SGLANG_NUMA_INTERLEAVE"
