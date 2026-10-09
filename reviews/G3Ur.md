Seven branches can advance; the AutoRound split branch still needs a runnable GPU validation plan.

| Branch | Earlier finding and evidence | Classification |
|---|---|---|
| `qsa/U1-hisparse-coordinator-gating` | **Resolved.** `b8d2e16836` explicitly sets `enable_hisparse=False` in both gate tests and supplies the old merge-path stub. Behavioral pre-fix failures are documented. [U1.md](/docs/upstream/U1.md:51) | **ready for owner publication decision** |
| `qsa/U9-monotonic-req-generation` | **Resolved; assessment unchanged.** `9bf148ed86` preserves generations in both pools; the CPU regression distinguishes the reset behavior. Environment failures remain disclosed. [U9.md](/docs/upstream/U9.md:85) | **ready for owner publication decision** |
| `qsa/U9-hisparse-decode-mm-inputs` | **Resolved.** `445e71bb34` changes the new class to `CustomTestCase`; the metadata repair and regression remain intact. [U9.md](/docs/upstream/U9.md:88) | **ready for owner publication decision** |
| `qsa/U7-gptq-moe-w13-scale-k` | **Resolved.** The CPU permutation evidence remains sufficient for offering the incremental commit. The doc explicitly restricts publication to a contribution to #35955 and makes no GPU accuracy claim. [U7.md](/docs/upstream/U7.md:229) | **ready for owner publication decision**, as that contribution |
| `qsa/U7-autoround-moe-marlin-group-split` | **Partially resolved.** Full versus per-rank widths are corrected in the doc and `5b20ecdda8`’s message. GPU output, accuracy and performance evidence remains pending; the validation section supplies TODO requirements rather than a runnable affected-shard comparison. [U7.md](/docs/upstream/U7.md:295) | **needs changes** |
| `qsa/U6-qsa-sm8x-varlen-fallback` | **Partially resolved.** The resolver regression is now CPU-registered, and fork evidence is clearly identified as historical. The new head-dimension-256 GPU reference test asserts vendored FA3 selection; documented commands include a main-source control. GPU execution remains pending. [U6.md](/docs/upstream/U6.md:203) | **ready after the listed GPU validation** |
| `qsa/U6-qsa-fp8-kv-scales` | **Partially resolved.** CPU registration is fixed. The new GPU test covers cached-prefix prefill and decode, distinct K/V scales, a dequantized-cache reference, and BF16/unit-scale controls. Commands include a failing unscaled-source control; TRTLLM/HIP limits are explicit. [U6.md](/docs/upstream/U6.md:221) | **ready after the listed GPU validation** |
| `qsa/U8-marlin-moe-batch-invariant` | **Partially resolved.** `5e9921bba7` adds the dequantized numerical-reference assertion while deterministic mode is enabled, alongside BF16/FP16 invariance and graph replay. The documented commands cover compilation, a pre-fix control, representative performance and model accuracy. Execution remains pending. [U8.md](/docs/upstream/U8.md:178) | **ready after the listed GPU validation** |

The smallest remaining U7 change is to provide a runnable GPU comparison that exercises the actual repeated-group loading on a 320- or 192-wide partition, plus concrete serving/evaluation and prefill/decode benchmark commands. Preserve the documented parent-merge and rebase prerequisites.

The other earlier findings are addressed:

- **U4 — resolved:** recognizes #35594 as merged and reassesses K02 through the existing pool-class seam, with equivalence limits. [U4.md](/docs/upstream/U4.md:19)
- **U5 — resolved:** separates U1’s gating repair from #35488’s protocol. [U5.md](/docs/upstream/U5.md:27)
- **U23 — resolved:** preserves the after-row-free function hook and explains normal, exceptional and early-return behavior; `on_release` is no longer treated as equivalent. [U23.md](/docs/upstream/U23.md:198)
- **INT8-row PLE — resolved:** deferral remains sound; #41624 is described as related plumbing requiring a later format addition.
- **Stable HC — resolved:** the conclusion is scoped to the compiled path; SM100’s low-M CuTe path remains explicitly unverified.
- **Stable top-k — partially resolved:** avoiding duplication remains sound; fork equivalence, graph replay and numerical-deviation acceptance remain deferred plugin-removal prerequisites.
- **Shared test-admission reasoning — resolved:** deferrals are now scope/evidence/maintenance judgments.
- **PR template/pre-commit — partially resolved:** bodies include the template sections and disclose missing checks; pinned pre-commit execution remains pending.

Read-only verification: all eight heads match the requested revisions; Python AST and whitespace checks passed, and eight isolated U8 alignment cases passed against the committed Python oracle. Full suites were not independently reproduced; no GPU validation ran. No additional production-code defect was established.

G3-U: not cleared