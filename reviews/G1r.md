1. **G1-1 — Resolved.** [plugin_probe.py:47](/home/zyk/projects/interests/ai-video/qwen/qsa-hisparse-plugin/tools/evidence/plugin_probe.py:47) activates compat, redirects Marlin, and supplies the plugin alignment helper. Source overrides are refused. The [whole-K test:410](/home/zyk/projects/interests/ai-video/qwen/qsa-hisparse-plugin/tests/model_compat/test_marlin_deterministic_alignment.py:410) now fails when the fork script is missing. Test names and parameter matrices match the pinned fork.

2. **G1-2 — Resolved.** [compare.py:59](/home/zyk/projects/interests/ai-video/qwen/qsa-hisparse-plugin/tools/evidence/compare.py:59) includes required ledger summaries and per-stage Marlin hashes. Read-only mutations of restore counts, maximum entries, Marlin hashes, top-k exactness and token IDs were rejected.

3. **G1-3 — Resolved for the required HiSparse invocation, `--require-observer 2`.** [compare.py:301](/home/zyk/projects/interests/ai-video/qwen/qsa-hisparse-plugin/tools/evidence/compare.py:301) requires capture and restore records on both ranks in both arms. In-memory checks rejected identically missing, empty and capture-only streams, plus identical restore corruption in both arms.

4. **G1-4 — Resolved.** [run_compat.py:128](/home/zyk/projects/interests/ai-video/qwen/qsa-hisparse-plugin/tools/evidence/run_compat.py:128) waits for launcher `ready.json` before accepting HTTP health. Mocked checks confirmed the wait and timeout behavior.

5. **G1-5 — Resolved.** [launch.py:64](/home/zyk/projects/interests/ai-video/qwen/qsa-hisparse-plugin/src/sglang_qsa_hisparse/launch.py:64) reads the package-local lock, included by [pyproject.toml:23](/home/zyk/projects/interests/ai-video/qwen/qsa-hisparse-plugin/pyproject.toml:23). The lock is readable.

**New findings:** None within the requested scope and threat model.

No files modified. GPU runs, full pytest and a fresh wheel build were not performed; verification used source inspection and read-only in-memory checks.

G1: cleared