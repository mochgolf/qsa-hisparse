# P4 final-acceptance-01 main audit

**FUNCTIONAL_BRIDGE_ACCEPTED; THREE_ARM_ACCEPTANCE_INCOMPLETE.** The resident
reference is numerically and operationally valid. The original driver remains
`FAIL_RESTORED`, because its literal historical cadence check required
`73 B1 / 31 B2 / 23 B1` while both new TP ranks produced
`70 B1 / 31 B2 / 26 B1`. The two light treatments were never started, so this
audit does not report `PASS_P4_B1_PREFILL_INTERLEAVING`, a scheduling
performance comparison, or production readiness.

The split check was overconstrained. Both schedules have the same B1 -> B2 ->
B1 topology, exactly 31 joint B2 forwards, 96 total B1 forwards and 127 total
decode forwards. The new run moved three initial-only forwards from before the
replacement became decode-ready to after it completed. That placement depends
on elapsed service work; it does not change a prompt boundary, request row,
generation, sequence progression or output. The frozen helpers pass every
semantic gate when only this literal split equality is removed.

The resident bridge is exact. Long B completed 128 tokens and replacement A
completed 32 tokens, position-for-position equal to execution-01 strict. B's
first 49 tokens also equal diagnostic-06. Incoming A was admitted between B
tokens four and five, consumed exactly two 4096-token chunks through 8192,
observed a real B1 decode, terminated by HTTP 200 abort, drained, and reused
the same request row and lease slot at generation two. Both ranks agree on all
64-chunk completed prompts and ordered decode rows.

Resident ownership also passes: no graph or D2H layer event, no host/hot or
workspace allocation, stable 524288-token raw backing and index storage,
identity-consistent release for all three RIDs, 63 decode opportunities between
replacement chunks, complete logical/Mamba/lease closure, four-process GC,
674 numeric NVML device samples with no errors, and final idle. The pre-test
service was restored with health, launch arguments, working tree, source and
environment identity matching its snapshot; an independent check also passed.

This bridge makes the previously accepted execution-01 strict offload result a
valid functional oracle: that run already passed cancellation, generation
reuse, B1/B2 graph, copy ordering, ownership, interleaved progress and closure,
and the final source's strict graph path is unchanged. It establishes the P4
functional prerequisite for advancing. It does not replace the missing light
interval-0/1 measurements. Per the no-retry instruction, those arms remain
`NOT_RUN_FAIL_CLOSED`; P5 online-arrival measurements will directly cover the
chosen interval-1 policy instead of adding another P4 campaign.

Resident timing is descriptive only. During replacement prefill, B's output
gap was 702.67 ms at P50, 884.46 ms at P95 and 900.61 ms maximum; pure resident
decode P95 was 99.46 ms and incoming TTFT was 46.04 s. These strict resident
numbers are neither offload cost nor a production SLO result, but they confirm
that P5 must preserve per-request gap reporting while scaling capacity.

Main now advances to the existing P5 B4-first capacity implementation. B1/B2
correctness and long B2 ownership are reused; B4 and then B8 must each pass
real decode membership, output, capacity and release gates before the next
size runs. No production deployment is authorized by this audit.
