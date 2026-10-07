"""W2's hooks are applied only when activated, and only through HookRegistry."""

import builtins
import dis
import inspect
import json
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import sglang_qsa_hisparse
from sglang_qsa_hisparse import fingerprint, patching

PINNED_TESTS = (
    "test/registered/unit/managers/test_batch_result_processor_hidden_states.py",
    "test/registered/unit/managers/test_scheduler_chunked_req_gate.py",
)


def test_w2_hooks_apply_and_teardown_restores_pinned_sglang(w2):
    from sglang.srt.managers import schedule_batch
    from sglang.srt.managers.scheduler_components import batch_result_processor
    from sglang.srt.mem_cache import allocation, common
    from sglang.srt.plugins.hook_registry import HookRegistry
    from sglang_qsa_hisparse.patches.hisparse import scheduler

    originals = {t: patching._raw_attribute(t) for t in w2.targets}
    pinned_release = common.release_kv_cache
    with w2.active():
        assert all(patching._raw_attribute(t) is not originals[t] for t in w2.targets)
        # Copies reach other patched targets through propagated bindings (G3).
        assert scheduler.release_kv_cache is common.release_kv_cache
        assert common.release_kv_cache is not pinned_release
        assert schedule_batch.alloc_for_extend is allocation.alloc_for_extend
    assert all(patching._raw_attribute(t) is originals[t] for t in w2.targets)
    assert scheduler.release_kv_cache is batch_result_processor.release_kv_cache
    assert batch_result_processor.release_kv_cache is pinned_release
    assert schedule_batch.alloc_for_extend is allocation.alloc_for_extend
    assert not HookRegistry._hooks and patching._activated is None


def test_plugin_off_keeps_pinned_definitions_and_upstream_tests_pass(w2):
    # A fresh process: plugin.load() with no switch, then every W2 target must
    # unwrap to its pinned definition through no HookRegistry or plugin
    # wrapper, and the pinned originals of the two ported tests must pass.
    script = textwrap.dedent(
        """
        import inspect, json, os, pkgutil, sys
        from sglang_qsa_hisparse import fingerprint, plugin
        plugin.load()
        from sglang.srt.plugins.hook_registry import HookRegistry
        assert not HookRegistry._hooks
        assert "sglang_qsa_hisparse.patching" not in sys.modules
        root, pinned = fingerprint.installed_source_root(), fingerprint.load_pinned()
        problems = []
        for target in json.loads(sys.argv[1]):
            owner_path, name = target.rsplit(".", 1)
            owner = pkgutil.resolve_name(owner_path)
            attr = owner.__dict__[name] if isinstance(owner, type) else getattr(owner, name)
            chain = [attr]
            while hasattr(chain[-1], "__wrapped__"):
                chain.append(chain[-1].__wrapped__)
            files = [os.path.realpath(f.__code__.co_filename) for f in chain]
            where = (files[-1], chain[-1].__code__.co_firstlineno)
            record = pinned[target]
            if (
                inspect.unwrap(attr) is not chain[-1]
                or any("hook_registry" in f or "sglang_qsa_hisparse" in f for f in files)
                or where != (os.path.realpath(root / record["file"]), record["first_line"])
            ):
                problems.append(target)
        print("PROBLEMS", json.dumps(problems))
        import pytest
        sys.exit(pytest.main(["-q", "-p", "no:cacheprovider", *sys.argv[2:]]))
        """
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SGLANG_QSA_") and key != "SGLANG_PLUGINS"
    }
    root = fingerprint.installed_source_root()
    plugin_src = Path(sglang_qsa_hisparse.__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join([str(plugin_src), str(root)])
    result = subprocess.run(
        [sys.executable, "-c", script, json.dumps(w2.targets), *PINNED_TESTS],
        cwd=root.parent,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert "PROBLEMS []" in result.stdout
    assert "10 passed" in result.stdout


def _global_names(code):
    names = {i.argval for i in dis.get_instructions(code) if i.opname == "LOAD_GLOBAL"}
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= _global_names(const)
    return names


def test_copied_bodies_resolve_every_global_name(w2):
    # Copies run with the plugin module's globals (inventory 6, G3); branches
    # no CPU test reaches must not fail with NameError on the GPU.
    pending = [
        inspect.unwrap(spec.hook) for spec in patching._declared if spec.row in w2.rows
    ]
    seen, missing = set(), []
    while pending:
        fn = pending.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for name in _global_names(fn.__code__):
            if name in fn.__globals__:
                value = fn.__globals__[name]
                if inspect.isfunction(value) and value.__module__ == fn.__module__:
                    pending.append(value)  # Plugin-local helpers (_add_one_req).
            elif not hasattr(builtins, name):
                missing.append(f"{fn.__module__}.{fn.__name__}: {name}")
    assert len(seen) > len(w2.rows) and not missing
