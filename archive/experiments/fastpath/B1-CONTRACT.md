# HiSparse B1 ragged-FA2 acceptance

Test source `a8987a6fd3057edab2bef029fa36ac420e550dba` against its parent
`5401b6e65925422acd3c9a015935267970acbf4b` with the matched baseline
model and runtime. Keep the baseline argv unchanged except for the test port,
and change only `PYTHONPATH` and cwd/PWD between arms.

Before either service arm, run one SM89 FP8 component check with independent
K/V scales. Then run baseline followed by candidate, each with one warmup and
two measured copies of the historical 38,185-token request and 768 generated
tokens. Preserve cumulative output IDs and per-token monotonic timestamps.

Acceptance requires component numerical agreement, complete B1 output,
TP0/TP1 full decode-graph counters with no eager decode, recovered request/KV/
Mamba metrics, no fatal server log pattern, and the candidate fast-path marker
from the executed branch on both ranks. The integration performance gate is at
least 10% higher mean token-65-to-768 rate than the parent; this is well below
the historical path's 27.2% gain while remaining above measured run drift.
Report full-response and token-65-to-768 rates. Treat output-ID
equality as descriptive because this native runtime is not deterministic.
Report both relative gain and distance from the historical 104.931 tok/s; do
not claim historical parity from a different model or prompt length.

Use the 3-second baseline idle gate, run only on the test endpoint, stop each test service,
and restore the baseline with exact argv, cwd, full environment, model identity, and
source commit in the single `finally` path. No profile, retry, B4/B8 arm, source
deployment, or production-GO decision belongs to this run.
