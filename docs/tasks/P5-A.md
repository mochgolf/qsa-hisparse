# P5-A scheduler, lifecycle and tree-cache selection at 35f3c96ff4

Read `P5-common.md`. Rows of P4-A (S*, P*, B*, M*). Port to the new pin with
production as the reference: changed REPLACE targets B03, B05, M03, P02, S07
and the missing `maybe_cache_unfinished_req` dependency. Map the `?` hunks
of `mem_cache/common.py` (4) and `mem_cache/registry.py` (4): production keeps
`ChunkCache` when the QSA host prefix is active under `--disable-radix-cache`
on hybrid-SSM models (#42354 adaptation) and changes the release path. The
release order (restore → async copy → logical release → free-group flush →
physical reuse) must hold.
