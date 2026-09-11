# HiSparse B>1 ragged-FA2 component gate

Validate source `687a39bbe0af00240fecb61872e8643ccb076cb7` on one RTX 4090 before any
service run.  Capture one default split-KV ragged-FA2 graph for B2 and one for
B8, each planned at 2051 K/V rows per request.  Within the same captured graph,
replay two mixed vectors spanning every real P2 valid count, 2048 through 2051.

The graph must recompute `cu_seqlens_k`, compact independent FP8 K/V rows with
non-unit K/V scales, and run the candidate wrapper.  Every replay must produce
the expected prefix sum, finite output, bitwise-stable repeated output, and
BF16 agreement with both an independent FP32 reference and the current classic
FA2 varlen path.  Do not test arbitrary short rows: P2 handoff rejects prompts
below 2048 and its 2051-wide selection consists of 2048 selected tokens plus a
0..3-token tail.

Use the baseline 3-second idle gate, stop it only after the gate passes, run no
model service or request, and restore exact argv, cwd, full environment, model,
source commit, test-endpoint availability, and the two baseline GPU owners.  This gate
does not establish service throughput, B8 production readiness, or deployment.
