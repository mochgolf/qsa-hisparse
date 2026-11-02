Read-only follow-up to reviews/G7CPU-remediation.md. Use the prescribed
GOAL.md/PLAN.md protocol; no delegation, GPU/service work, environment
changes, credential/config reads or publication. Do not repeat the completed
source review, inspect raw GPU data or reread large evidence files.

Prior review found no code defects and recorded final complete CPU passes:
772 unit; integration4/1skip and11. It withheld clearance solely for P3:
docs/patch-inventory.md's maintenance paragraph wrongly said plugin kernels
use main's fixed batch-invariant selector. Read only that paragraph around
lines814-820, src/sglang_qsa_hisparse/kernels/qsa_sparse_attn.py lines1-38,
and reviews/G7CPU-remediation-response.md. It now describes local legacy
selection and module-fingerprinted imported table data. No product source
changed since the cleared code review. Confirm whether P3 is resolved;
report any remaining directly related issue, otherwise clear the CPU gate.
End with G7-CPU-remediation: cleared or not cleared. Full GPU qualification
remains pending and is a separate gate.
