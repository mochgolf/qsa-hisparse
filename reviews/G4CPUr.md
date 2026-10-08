No new findings. The finding is resolved: exact three-way-merge equality rejects moved or misplaced fork lines for mechanically checked copies, including renamed bodies.

M03 genuinely conflicts and remains a documented manual exception. Its hand merge preserves the fork’s release order: capture and runtime release before handoff/logical free; `after_release` after row free and `mark_kv_released`; streaming claims return before `after_release`. Removing the fork additions restores the v0.5.21 AST.

All 35 focused source tests passed using memory-backed temporary files. Repository unchanged; full CPU suite not rerun.

G4-CPU: cleared