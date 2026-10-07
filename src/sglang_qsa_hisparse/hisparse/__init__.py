"""QSA HiSparse CPU KV offload integrated into SGLang.

Read config and layout for the supported geometry, slots for ownership, runtime
for multi-request execution, and coordinator for scheduler/TP handoff. The
single_request module also provides the original eager comparison runtime.

Modules are imported explicitly so that lease bookkeeping needs no GPU runtime.
"""
