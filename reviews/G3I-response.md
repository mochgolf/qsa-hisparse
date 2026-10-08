# G3-I (CPU) response

Review: `reviews/G3I.md`.

| # | Resolution |
| --- | --- |
| 1 (high) caller-supplied processor outputs | Fixed: `identity_for` admits only `MultimodalInputFormat.NORMAL` items (built by the processor in the server); `PROCESSOR_OUTPUT` and precomputed items bypass. Counterexample added to `tests/image/test_identity.py` bypass cases. |
| 2 (medium) retraction | Not applicable to supported configurations: QSA HiSparse does not support retraction or preemption (QSA_HISPARSE.md; the runtime's `retract_req` raises), and session requests bypass host prefixes. A retracted image request can only miss, never hit wrongly. No change. |
| 3 (medium) I-C criterion coverage | Forwarded to I-C: frozen expected restore length per hit case, inside-image assertion `B.start < L < B.stop` with `cached_tokens == L`, named page-aligned (logits tail) and input-logprob cases. |
| 4 (medium) ViT work and batch invariance | Forwarded to I-C: record encoded image identities per request, require skips for images ending before L, and a focused GPU embedding-bytes comparison of one image encoded alone vs batched with the per-image cache disabled. |
| 5 (low) multi-tokenizer | PLAN corrected: multi-tokenizer image items carry no artifact key and bypass (no wrong hit possible), so it is documented, not rejected (rule 10). |
