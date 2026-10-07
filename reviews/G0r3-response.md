# G0 round 4 response

Review: `reviews/G0r3.md`. All findings accepted; the in-place mutation
class is now bounded by an explicit threat model (PLAN.md, "Activation
guarantees").

| Finding | Resolution |
| --- | --- |
| 1 (blocker) shallow behavior records | Every function level, internal or external, records closure, defaults and keyword defaults, recursively for function-valued captures (depth-limited; deeper or opaque state is `Undescribable`). Enum values are identified. Class levels record descriptions of all function/descriptor members. Operator levels record normalized dispatcher registrations (`torch._C._dispatch_dump`). Chain format is versioned (`chain_format`); stale records are refused. `verify_final` recomputes descriptions of the wrapped originals of patched targets and of dependencies and fails on any change (`Protected bindings changed in place`). Tests: set_grad_enabled(True/False) and partials over different closures differ; an operator kernel override changes the chain; kwdefaults and class-member mutations fail before activation and at final verification. Out of scope (documented): mutations after final verification, same-location kernel re-registration, unchecked bindings, native libraries (launcher version lock), other plugins (launcher sets `SGLANG_PLUGINS=qsa_hisparse`). |
| 2 (blocker) unmanifested same-source hooks | Any registry entry on or around a protected name before this activation registers is rejected regardless of source; after registration each target's registry list must equal the declared hooks exactly (type, hook object, full source identity). Test reproduces the reviewer's QSA-sourced AROUND case. |
| 3 (major) feature ownership | Manifest keys include the feature. Test swaps feature tags with an independent manifest. |
| 4 (minor) lambda defaults, global redirects | Lambda defaults are scanned in the enclosing scope; bindings in any nested function or class that declares the name `global` count. Tests for lambda default, class-body global and function-body global. |

CUDA chains: the chain format changed, so framework records currently carry
CPU chains only; activation in CUDA mode fails ("no pinned binding chain")
until they are regenerated in an agreed GPU window (the production service
is running). Framework tests: 80 passed.
