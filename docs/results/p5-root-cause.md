# P5c 256K/768 root-cause result

## Verdict

**PASS_ROOTCAUSE_STALE_PRE_ALIGNMENT_ORACLE.** The P5c endpoint did not expose
a HiSparse graph or KV-offload numerical defect. Its final-source output was
compared with a resident-eager oracle produced at source `f04c0d7955`, before
the accepted eager/full-graph score-width alignment `114ac4b89a`.

No candidate runtime source change is justified by this failure. The fixes are
to bind P5c to the independently collected aligned resident-eager vector and to
correct one resident-storage assertion in the diagnostic reducer.

## Decisive output evidence

All paths used the same 261,120-token request, deterministic seed, TP2, 4,096
token prefill chunks, no MTP, and final candidate source `7c37f8ee93`.

- Fresh resident-eager and preserved P5c offload-full graph outputs match at
  all 768 positions.
- Fresh offload-eager matches both of those paths at all 16 collected
  positions, including the original first mismatch.
- The pre-alignment V3 vector first differs at zero-based output 9
  (`264` versus the aligned value `279`) and differs at 752 of 768 positions.

The accepted P4 replay already established the mechanism: FlashInfer's
deterministic top-k collection order changes with score width even when the
valid score prefix and selected set are the same, and that ordering changes
the downstream FA2 output. P4 then showed that aligned resident-eager matches
full graph at model level. The fresh 768-token resident run extends that
model-level agreement through the complete P5c output.

## Lifecycle evidence

The offload-eager arm records 64 gap-free prefill chunks and 15 consecutive
decode forwards on each rank. The resident-eager arm records the same 64
prefill chunks and 767 consecutive decode forwards on each rank. TP schedules
match across both ranks in both arms.

Both arms complete the terminal release chain and restore logical KV, Mamba,
lease, staging, host, hot, and pending-release ownership. Raw and index storage
remain stable, both GC gates pass, final idle passes, and the pre-test service
restoration matches its health, launch, source and environment snapshot.

## Reducer failure and fix

The original diagnostic ended `FAIL_RESTORED` after both requests completed
because its resident assertion treated `raw_pool_size_tokens` as the physical
backing capacity. In this implementation that field describes compact slot
geometry: `262144 + 5 * 2 = 262154`. The independent
`raw_backing_size_tokens` field records the resident physical backing and is
`524288`.

Reducer commit `66b1aa6` corrects that one expectation. Offline replay of the
preserved GPU evidence then passes output, TP lifecycle, ownership, GC, NVML,
SSE, and final-idle validation; no GPU rerun was needed. The corrected
reduction is committed at `319c77c`, while the original manifest remains
unchanged as an honest record of the reducer failure.

P5c driver commit `7745170` now binds the aligned resident artifact and this
accepted root-cause audit before any service mutation. Replaying the preserved
P5c GPU evidence with that binding returns
`PASS_P5C_FINAL_768_B1_REGRESSION` at `e02bfec`; the original failed manifest is
preserved unchanged. Independent root-cause and measurement reviews passed at
`ed88dc0` and `d4a1f6f`.

This closes the P5c numerical regression. It does not qualify B8, a fixed SLO,
performance, or production GO.
