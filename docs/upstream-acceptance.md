# What sgl-project/sglang merges, and what stalls (2026-10-08)

Read-only research. `gh` was authenticated, so GraphQL and REST GET calls were used. No WebFetch fallback was needed.
Rules come from the upstream checkout `../.worktrees/upstream-UA` (`b7b2975b57`).
Dataset: all **11,488 PRs created 2026-05-01..2026-08-31**, with each PR's state as of 2026-10-08. Per-month totals match GitHub search exactly: May 2,315, Jun 2,560, Jul 2,943, Aug 3,670.
Author classes:
- **maint**: listed in CODEOWNERS or MAINTAINER.md, or `authorAssociation` is COLLABORATOR/MEMBER/OWNER.
- **ext**: everyone else, excluding bots.
- **non-vendor ext**: ext minus PRs whose titles or labels mark AMD/NPU/XPU/MLX/CPU work. Those vendors have their own merge paths (`HaiShaw`, `sglang-npu-bot`, `mingfeima`).

Two kinds of numbers appear in the tables. "Observed" numbers are counts. Classifications based on titles or keywords are marked *approx*. Inferences are marked *inferred*.

## Recommendations

**General rules (each backed by the evidence below)**
1. **Make each PR one upstream bug, in one CODEOWNERS area, with at most about 100 changed source lines.** Present the seam the plugin needs as a side effect of that fix. For non-vendor ext PRs in our areas, merge rates are 33-34% at 100 changed lines or fewer and 21-24% above that. Median time to merge is 3.2 days at 100 lines or fewer and 13.1 days above. A PR that touches several protected areas needs one owner approval per area: #38749 and #38751 are approved but blocked because the kernels owner never approved.
2. **The bottleneck is getting a maintainer to engage, not the code itself.** In our areas, 73% of ext PRs that received the `run-ci` label merged, against 6% of those that never did. 47% of PRs with at least one review merged, against 4% with none. Only users listed in `CI_PERMISSIONS.json` can add `run-ci`. Each PR body should name one area reviewer and the Merge Oncall. Note that pinging anyone is publication, so the owner must confirm first under the GOAL.md rule.
3. **Never build on code an oncall is actively refactoring.** In the last 30 days `mem_cache/` got 224 commits and `scheduler.py` got 81. hnyls2002 self-merges lifecycle refactors within a day: #42823/24/25 and #42923/#43023 (opened 10-06..08, all merged by 10-08). Of 92 closed ext PRs in our areas whose last comment came from an owner or oncall, at least 28 were superseded or re-implemented (*approx*, keyword match). Examples: #35793 was replaced by #40075, #29262 was fixed on main by #27193, and in #30738 the reporter was added as co-author on #31443.
4. **Keep at most 3 PRs open at once and touch each one weekly. Do not leave drafts open.** The stale bot (#34380, since 2026-08-11) closes three kinds of ext PRs:
   - drafts or WIP idle for more than 21 days;
   - any PR idle for more than 90 days;
   - every PR of an author with more than 5 open PRs when none of them was updated in 7 days.
   Approved PRs are exempt. 992 of 2,497 closed ext PRs (40%) were closed by this bot. Of the 483 ext PRs that are still drafts, 445 (92%) have been closed.
5. **Do not file an RFC first for a small change.** Of 76 ext `[RFC]` issues opened Apr-Sep, 20 got no comment at all, and only 14 (18%) got a comment from a CODEOWNER or oncall. Open an issue only when a reviewer must choose between designs, and then name that reviewer.
6. **Tests: follow the admission rule, nothing more.** A test should be a bug regression, a derived property, or bookkeeping. Coverage-only test PRs were closed in bulk (#33909: "no longer accepting coverage-driven UT PRs"). Having tests showed no positive link to merging (25% with tests vs 35% without; confounded by size).
7. **Hand-check PR text for an AI-generated look.** Monthly PR volume rose from 2.3k to 4.2k (Sep). A maintainer closed #24342 with: "this Roadmap is not for public contributions ... a vibe coding PR is not valuable."

**Per U-task** (sizes are the local branches against `b7b2975b57`)

| Task | Action | Size / split | Ask | Main risk |
| --- | --- | --- | --- | --- |
| U1 | **New PRs, split in two.** PR-A: the staging-abort fix (commit `6b53a9ac4d` plus its test), now. PR-B: the coordinator gating (`604dc106f3`, `b8d2e16836`), after A merges. Pitch B as making `TpModelWorker.register_hisparse_coordinator` actually work. Do not join #35488 (+4,265 lines, no human review since 08-19, never got `run-ci`). | Whole branch is +103/-38 in 10 files, mostly tests. Keep A under about 50 source lines. | `managers` owners xiezhq-hermann or hnyls2002; cc alphabetc1 (HiSparse owner who fixed the related abort path in #41346, +58 lines, 5 days) | B is a seam with no in-tree user. It may be met with "delete the unused hook" (*inferred*). |
| U23 | **Wait.** #42923 and #43023 merged on 10-08, so the interface is settling. Design the plugin's Phase-4 radix backend first. Then propose only the 2 missing hooks, with that backend as the stated consumer. | 1 PR, no more than about 100 lines | hnyls2002 (KV-cache owner, active) | Superseded by the next refactor (#35793 pattern). |
| U4 | **No PR.** The #35594 seam (merged 10-07) is enough. | - | - | - |
| U5 | **Wait.** When revisited, copy the #40222 pattern: subclass seams on `DecodeCudaGraphRunner`, +81/-7 lines, merged in 1 day. That avoids protocol methods that depend on #35488. | 1 PR, no more than about 100 lines | merrymercy (merged #40222) or Fridge003 | `model_runner.py` is frozen. |
| U6a | **New small PR now** (8 source lines, +126 total). Cite stale ext #36968 (opened 08-29, never reviewed). Add SM89 GPU evidence to the body first: the branch has not been GPU-tested. | 1 PR | Qiaolin-Yu or YAMY1234 (QSA activity); `layers/attention` owners Fridge003, ispobock | Low. Tiny ext QSA fixes merged in 0-5 days (#36649 +4 lines, #38346 +1 line). |
| U6b | **Wait for #41933** (Qiaolin-Yu, updated 10-08) to merge. Then rebase, credit #36644, and add a gsm8k run on an FP8 checkpoint. | +342/-43; the source part is about 110 lines. One PR. | Qiaolin-Yu | Overlap with #41933. The size band above 300 lines merges at about 24%. |
| U7 | **Contribute to #35955; open no competing PR.** Post one review comment with independent CPU test results and offer the w13 `size_k` commit. Hold the AutoRound split until #35955 merges. Drop INT8 PLE. | w13 is +46 lines on top of #35955 | quant owners mmangkad or b8zhong. The PR also touches `layers/moe` (owners ch-wan, BBuf) and `hardware_backend/gpu/quantization` (Alisehen). | #35955 has had no review and no `run-ci` since 08-22. Ext Marlin/GPTQ titles merged 6 of 44. |
| U8 | **Not ready, then follow #42087.** Support #42087 with SM89 results; it covers QSA top-k and HC. Open the Marlin PR only after GPU validation and a cold-L2 benchmark (`.claude/rules/kernel-benchmark.md`), ideally after #42087 lands, framed as its follow-up. | +280/-20 across 3 owner areas (kernels, layers/moe, triton utils). Cannot be split usefully. | Fridge003 or hebiao064 (`batch_invariant_ops`), BBuf (kernel oncall) | Ext determinism titles merged 9 of 55. Kernel review is slow. |
| U9a | **New small PR now.** | +54/-2. `memory_pool.py` plus `disaggregation/decode.py`, both owned by hnyls2002. | hnyls2002 | Low. Mention #36093. |
| U9b | **New small PR now.** | +50/-19 (3 source lines) | `managers` owners; cc alphabetc1 | Low |
| Identity (I1) | #41792 was closed on 10-08 and **taken over by maintainer mickqian in #43043** (open, keeps the original commit). Align with #43043's interface and propose Qwen-VL wiring as a follow-up after it lands. | - | mickqian | - |

Suggested order, never more than 3 open at once:
1. U9b, U9a, U1-A
2. U6a
3. U1-B, U6b (after #41933 merges)
4. U8, U5, U23

## Evidence

### 1. Written rules
- **Merge process** (`.github/MAINTAINER.md`, PR template):
  - a bot assigns a Merge Oncall;
  - each modified protected file needs one CODEOWNER approval;
  - anyone with write access merges once CI is green and approvals are in;
  - a Merge Oncall can bypass. Observed: all six hnyls2002 lifecycle PRs merged with `reviewDecision=REVIEW_REQUIRED`.
- **CI**:
  - CI runs only with the `run-ci` label. The label can be added only by the 267 users in `CI_PERMISSIONS.json`, via `/tag-and-rerun-ci` or `/tag-run-ci-label`.
  - Authors may only `/rerun-failed-ci` their own PRs.
  - `pr-gate` fails on drafts and on PRs without the label. The "CI red" status of #35955 and #42087 is this gate (`call-gate / pr-gate: FAILURE`); their test suites never ran.
  - A failing lint job stops all CI.
- **Lint**: `pre-commit run --all-files`. Hooks: isort, ruff, ruff-format, codespell, clang-format, plus local checks (Chinese characters, registered tests, bare `pytest.main`).
- **Tests**:
  - CPU tests go in `test/registered/unit/<mirror of srt>/` with `register_cpu_ci(est_time=..., suite="base-a-test-cpu")`, `CustomTestCase`, and a standard `__main__`. Prefer extending an existing file.
  - GPU kernel tests go in `test/registered/kernels/ops/<group>/` with `register_cuda_ci`.
  - Changed-line coverage of at least 60% (`diff-cover`).
  - Admission criteria: `.claude/rules/unit-test-admission.md` (#30608, 2026-07-08).
- **Style**:
  - from `.claude/rules`: no defensive `getattr`/`hasattr`; `msgspec.Struct` instead of dataclasses; keyword arguments; no mixins;
  - `model_runner.py` is frozen (`large-class-style`). Changes to `Scheduler.__init__` must follow that skill too.
  - For new features: "do not drastically change existing code", and keep the common path first.
- **CODEOWNERS for our areas**:

  | Path | Owners |
  | --- | --- |
  | `managers` | merrymercy, Ying1123, hnyls2002, xiezhq-hermann |
  | `managers/hisparse_coordinator.py` | the `managers` owners plus hzh0425, ispobock, alphabetc1, huangtingwei9988 (added 09-13, #38682) |
  | `mem_cache` | 10 owners incl. hnyls2002, ispobock, alphabetc1 |
  | `model_executor` | merrymercy, Ying1123, hnyls2002, Fridge003, ispobock |
  | `layers/attention` | merrymercy, Fridge003, ispobock, Qiaolin-Yu, hebiao064, HaiShaw |
  | `layers/quantization` | 12 owners |
  | `python/sglang/kernels` | DarkSharpness, HaiShaw, BBuf, celve, HydraQYH, yuan-luo |
  | `batch_invariant_ops` | Fridge003, hebiao064 |
  | `srt/multimodal` | mickqian et al. |
  | `srt/plugins` | **no entry, so no CODEOWNER gate** |
- **Plugin docs** (`docs/docs/hardware-platforms/plugin.mdx`) ask for this kind of contribution: "Reach for a general plugin only when the Platform interface has no seam ... and please report that gap."

### 2. Empirical patterns (May-Aug 2026 cohort, n=11,488)

| Class | n | merged | closed unmerged | still open |
| --- | ---: | ---: | ---: | ---: |
| maint | 5,650 | 71% | 22% | 7% |
| ext | 5,766 | 30% | 43% | 27% |
| ext, author in `CI_PERMISSIONS` | 612 | 64% | 28% | 8% |
| non-vendor ext, non-draft, our areas* | 1,582 | 27% | 40% | 34% |

\*Our areas: any changed file in scheduler or batch-result code, `mem_cache`, hisparse, `model_executor`, `layers/attention`, quantization/MoE/Marlin, or plugin/registry/hook files. Per area: sched 23%, mem_cache 25%, hisparse 18% (n=38), attn 24%, mexec 28%, quant/MoE 29%.

- **Size.** For ext PRs merged by size band in our areas: 34% (≤20 lines), 33% (21-100), 24% (101-300), 24% (301-1,000), 21% (more than 1,000). Across all ext PRs, merged PRs have a median of 70 changed lines; closed ones 128, open ones 148. PRs touching 21 or more files: 19% merged.
- **Speed.** Merged ext PRs: median 4.9 days to merge, p75 15 days, p90 33 days. Our areas: median 7.6 days, p75 20.6 days. Maint PRs: median 0.6 days. 62% of maint PRs merge within a day.
- **Engagement dominates.** In our areas, ext PRs that received `run-ci` merged 73% of the time (n=498), against 6% for those that did not (n=1,084). With at least one review: 47%; with none: 4%. Of 1,517 still-open ext PRs, 1,034 have no review at all.
- **Who merges ext PRs in our areas**: Fridge003 60, hzh0425 44, BBuf 38, ch-wan 37, ispobock 36, hnyls2002 36, ShangmingCai 25, merrymercy 21.
- **Not predictive or negative** (*confounded, observed only*):
  - a linked closing issue: 15% merged vs 32% without;
  - title wording (fix 28%, feature 30%, refactor 35%);
  - containing tests (see rule 6).
- **What closes ext PRs.** In our areas, 808 ext PRs were closed: 306 by the stale bot, 221 by the author, 88 with no comments at all, and the rest after a maintainer comment. Closing reasons, with examples:
  - superseded or fixed on main: #29262, #29491, #26703, #35793 to #40075, #41792 to #43043;
  - maintainer roadmap work done by outsiders: #24342;
  - not a real bug or the wrong fix: #24745, #25901;
  - coverage-only tests: #33909.
- **Small ext fixes merge fast.** Sample from mem_cache, sched, and mexec: #25770 (+9 lines, 1 day), #32208 (+31 lines, `ReqToTokenPool.alloc` O(1)), #35204 (+91 lines, with a test, merged by hnyls2002), #42467 (+4 lines, 2 days, no formal review).

### 3. Plugin and extension-point PRs
- **The general plugin system** (#21388, Baidu-AIAK) was merged by merrymercy on 2026-04-20: 26 days open, +2,811 lines, labeled `high priority`. It made the ext PR #23840 unnecessary, and #23840 was closed. `srt/plugins/` has had only one formatting commit since (#37210).
- **`register_radix_cache_backend`** (#25101, Jialin, Meta) was merged 2026-05-20: 8 days, +516/-80, approved by merrymercy and rainj-me.
- **Ext seam PRs** found by title search for plugin, hook, registry, pluggable, out-of-tree, and extensible, created Apr-Oct (*approx*, hand-classified, n=52):

  | Outcome | Count | Detail |
  | --- | ---: | --- |
  | merged | 23 | median 101 changed lines; median 3 days to merge |
  | closed | 16 | 6 by stale bot; 3 superseded by another PR (#36319 by #37547, #27515 by #23969, #23840 by #21388); 2 replaced by the author's own PR; 5 unclear |
  | open | 13 | most have no review. #38464 (load plugins in all processes) has had none since 09-08. #38749 and #38751 are approved by alexnails but lack a kernels-owner approval. |

  - Merged examples: #40222 (decode CUDA graph hooks, +81 lines, 1 day), #40227 (GDN prefill hooks, 12 days), #38740 (out-of-tree DFlash extension points, 11 days), #37969 (out-of-tree graph backends, 14 days), #33426 (FA backend subclassable, 42 days), #24937 (CUSTOM enum, +19 lines, 1 day), #25050 (config-parser registry, 2 days).
  - What the merged ones share: small; a concrete consumer named (an out-of-tree platform or a model); a single owner area. Several authors list Meta or OpenAI as their company (#37969, #25337/#25347, #25807). *Inferred*: corporate authors have an internal champion. An unaffiliated contributor needs to recruit one, per rule 2.

### 4. State of tracked PRs (2026-10-08)

| PR | State | Author | Opened to merged | Size | Notes |
| --- | --- | --- | --- | --- | --- |
| #35488 | open | bingps | 08-19, idle since 09-22 | +4,265/-131, 27 files | assigned alphabetc1 and huangtingwei9988; only a Copilot review; no `run-ci` |
| #35955 | open | EanWang211123 | 08-22, force-pushed 10-08 | +130/-18, 3 files | no reviews or comments; no `run-ci`; 14 reviewers requested |
| #42087 | open | Misaka9468 (Tencent) | 10-01, idle since 10-03 | +191/-2, 7 files | one community test comment; no `run-ci` |
| #41792 | closed 10-08 (draft) | rchalamala | - | +613 | superseded by #43043 (mickqian, open, +801) |
| #42354 | merged | hnyls2002 | same day (10-03) | +81/-80 | self-merged, oncall bypass |
| #42823 / #42824 / #42825 | merged | hnyls2002 | 10-06 to 10-07 | 6 / 45 / 40 files | self-merged |
| #42923 / #43023 | merged | hnyls2002 | 10-07/08 to 10-08 | 43 / 16 files | self-merged |
| #39928 | merged | Jiminator | 09-17 to 09-20 | +74/-18 | approved and merged by YAMY1234 |

## Unverified or inferred
- Author affiliation comes from public GitHub profiles. The "corporate champion" effect is inferred, not measured.
- Keyword classification (superseded share, seam PR set, area tags from the first 40 files of each PR) is approximate.
- The outcome rates are correlations. `run-ci` and reviews are markers of maintainer engagement, not independent causes.
- Whether maintainers would accept the U1 gating seam, or prefer deleting `register_hisparse_coordinator`, is unknown.
