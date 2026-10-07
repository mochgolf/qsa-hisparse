"""Source fingerprints of patch targets, computed without importing them.

Activation is gated on the raw bytes of every module that contains a patch
target or a declared dependency: the plugin is pinned to one SGLang commit,
so any edit to such a module fails closed (conditional redefinitions,
constants, helpers, line endings, indentation).

Each record also hashes the definition itself (raw bytes, decorators, and the
decorator/header lines of every enclosing class) to report which definitions
changed when the pin is upgraded; the module hash alone gates activation.

Pinned records live in ``fingerprints/*.json`` (one file per patch module, so
parallel work does not share a file). See PLAN.md, "Activation guarantees",
for what activation does and does not detect.
"""

import ast
import hashlib
import json
from importlib import resources
from pathlib import Path

from sglang_qsa_hisparse.errors import FingerprintMismatch

_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _locate(tree: ast.Module, qualname: str) -> tuple[ast.AST, list[ast.ClassDef]]:
    """Last direct definition along ``qualname`` (the one Python binds last)."""
    node: ast.AST = tree
    enclosing: list[ast.ClassDef] = []
    for part in qualname.split("."):
        direct = [
            child
            for child in getattr(node, "body", ())
            if isinstance(child, _DEFINITIONS) and child.name == part
        ]
        if not direct:
            raise LookupError(f"{qualname}: no direct definition of {part!r}")
        if isinstance(node, ast.ClassDef):
            enclosing.append(node)
        node = direct[-1]
    return node, enclosing


def _first_line(node: ast.AST) -> int:
    return min([node.lineno] + [d.lineno for d in node.decorator_list])


def _header(lines: list[bytes], klass: ast.ClassDef) -> bytes:
    return b"".join(lines[_first_line(klass) - 1 : klass.body[0].lineno - 1])


def definition_record(path: str | Path, qualname: str) -> dict:
    data = Path(path).read_bytes()
    node, enclosing = _locate(ast.parse(data), qualname)
    lines = data.splitlines(keepends=True)
    first = _first_line(node)
    segment = b"".join(_header(lines, k) for k in enclosing)
    segment += b"".join(lines[first - 1 : node.end_lineno])
    return {
        "qualname": qualname,
        "kind": "class" if isinstance(node, ast.ClassDef) else "function",
        "first_line": first,
        "def_line": node.lineno,
        "sha256": _sha256(segment),
        "module_sha256": _sha256(data),
    }


# Records ------------------------------------------------------------------------


def resolve_file(source_root: str | Path, target: str) -> tuple[str, str]:
    """Map ``pkg.module.Qual.name`` to (relative module file, qualname) by path."""
    root = Path(source_root)
    parts = target.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        stem = Path(*parts[:cut])
        for candidate in (stem.with_suffix(".py"), stem / "__init__.py"):
            if (root / candidate).is_file():
                return candidate.as_posix(), ".".join(parts[cut:])
    raise LookupError(f"No module file under {root} for {target}")


def record(source_root: str | Path, target: str) -> dict:
    file, qualname = resolve_file(source_root, target)
    return {"file": file, **definition_record(Path(source_root) / file, qualname)}


def installed_source_root() -> Path:
    import sglang  # Already imported whenever SGLang loads this plugin.

    return Path(sglang.__file__).resolve().parent.parent


def load_pinned() -> dict[str, dict]:
    pinned: dict[str, dict] = {}
    directory = resources.files("sglang_qsa_hisparse").joinpath("fingerprints")
    for entry in sorted(directory.iterdir(), key=lambda e: e.name):
        if not entry.name.endswith(".json"):
            continue
        for target, data in json.loads(entry.read_text(encoding="utf-8")).items():
            if target in pinned and pinned[target] != data:
                raise FingerprintMismatch(f"{target}: conflicting pinned records")
            pinned[target] = data
    return pinned


def verify(
    names: list[str],
    *,
    source_root: str | Path | None = None,
    pinned: dict[str, dict] | None = None,
) -> dict[str, dict]:
    """Return the pinned records of ``names`` after checking their modules.

    Raises FingerprintMismatch if a name has no record or the bytes of its
    module differ from the pinned revision. Each module file is read once;
    definitions are parsed only to report which ones changed.
    """
    root = installed_source_root() if source_root is None else Path(source_root)
    pinned = load_pinned() if pinned is None else pinned
    problems = []
    records = {}
    hashes: dict[str, str | None] = {}
    for name in names:
        expected = pinned.get(name)
        if expected is None:
            problems.append(f"{name}: no pinned fingerprint")
            continue
        file = expected["file"]
        if file not in hashes:
            try:
                hashes[file] = _sha256((root / file).read_bytes())
            except OSError:
                hashes[file] = None
        if hashes[file] is None:
            problems.append(f"{name}: cannot read {file}")
            continue
        if hashes[file] != expected["module_sha256"]:
            try:
                actual = definition_record(root / file, expected["qualname"])
                changed = "definition" if actual["sha256"] != expected["sha256"] else "module"
            except (LookupError, SyntaxError):
                changed = "definition (missing)"
            problems.append(f"{name}: {changed} differs from the pinned revision")
            continue
        records[name] = expected
    if problems:
        raise FingerprintMismatch("; ".join(problems))
    return records
