You are an independent reviewer re-checking gate G3-U (Track U). Work
read-only; no GitHub writes. Be proportionate (owner's instruction).

Read `reviews/G3U.md` (your earlier findings) and the updated
`docs/upstream/U*.md`. Branches are local in `../qsa-hisparse` (inspect with
`git -C ../qsa-hisparse log/show/diff upstream/main..<branch>`; base
`b7b2975b57`):
- `qsa/U1-hisparse-coordinator-gating` (now ends `b8d2e16836`)
- `qsa/U9-monotonic-req-generation` (`9bf148ed86`), `qsa/U9-hisparse-decode-mm-inputs` (`445e71bb34`)
- `qsa/U7-gptq-moe-w13-scale-k` (`2796e71123`), `qsa/U7-autoround-moe-marlin-group-split` (`5b20ecdda8`)
- `qsa/U6-qsa-sm8x-varlen-fallback` (`6fa6778afc`), `qsa/U6-qsa-fp8-kv-scales` (`20509a88ca`)
- `qsa/U8-marlin-moe-batch-invariant` (`5e9921bba7`)

For each earlier finding: resolved / partially / not, with evidence. Then
classify each branch: "ready for owner publication decision" (CPU evidence
sufficient and the doc states remaining limits honestly), "ready after the
listed GPU validation" (GPU commands in its doc are sufficient), or
"needs changes". Report new findings only if they would make a PR wrong or
misleading. Final line: "G3-U: cleared" (classification complete, no branch
needs changes) or "G3-U: not cleared".
