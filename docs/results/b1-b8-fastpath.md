# HiSparse B1–B8 fast-path result

Source `687a39bbe0af00240fecb61872e8643ccb076cb7` passes the bounded component and service gates against production source `5401b6e65925422acd3c9a015935267970acbf4b`.

## Correctness and routing

- The component gate replayed one captured graph with dynamic compact lengths 2048–2051 for B2 and B8. Packed gather error was zero, ragged and fallback maximum absolute error against FP32 were at most `0.0009765625`, and repeated replays were bitwise stable.
- The service gate completed 2/2 B2 requests and 8/8 B8 requests. Both TP ranks used CUDA graphs with no eager decode or fatal log pattern.
- Candidate logs contain `QSA HiSparse SM89 ragged FA2` markers on TP0 and TP1 for B2 and B8; baseline logs contain none.
- Cross-arm generated token IDs are descriptive only and differed under concurrent service scheduling. Kernel correctness is established by the component oracle above, not by cross-arm service token equality.

## Performance

The service comparison used identical sanitized launch settings, frozen request-A with the authoritative 9-token chat tail, 256 greedy output tokens, and the natural common token-65→256 steady interval.

| Batch | Baseline aggregate | Fast path aggregate | Gain |
|---|---:|---:|---:|
| B2 | 130.4991 tok/s | 158.3351 tok/s | 21.3304% |
| B8 | 400.1555 tok/s | 464.1457 tok/s | 15.9914% |

The earlier B1 service gate measured 69.3809 → 86.0660 tok/s, a 24.0485% gain.

## Recovery

Each arm recovered full KV/Mamba capacity. The final restore matched the pre-test health, launch settings, source and resource ownership snapshot.

## Deployment

The guarded deployment validation advanced source `5401b6e659` to `687a39bbe0`. The 2048/8 smoke used the B1 graph on both TP ranks, completed 8/8 tokens with 17.0149 ms TPOT, used no eager decode, and recovered full capacity. The validated commit was pushed to the public fork.

Verdict: `PASS_B1_B8_FAST_PATH_DEPLOYMENT_VALIDATION`.
