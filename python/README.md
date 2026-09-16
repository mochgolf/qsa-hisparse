# QSA HiSparse runtime

This distribution provides the QSA HiSparse fork of SGLang. It includes the
complete runtime and request-scoped CPU KV offload for Qwen Sparse Attention.
Its distribution name, import namespace, and CLI remain `sglang`.

Install from a complete checkout with `python -m pip install -e ./python`.
Use one SGLang checkout per Python environment. Native extensions require the
SGLang Rust/CUDA toolchain; runtime dependencies are declared in `pyproject.toml`.

See the [project documentation](https://github.com/mochgolf/qsa-hisparse) for
the supported TP2 model geometry, launch configuration, architecture, upstream
merge workflow, and validation status. Historical performance measurements
apply only to their recorded source revisions.

Licensed under Apache-2.0; SGLang's source attribution and Git history are retained.
