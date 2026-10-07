"""Source fingerprints of patch targets, computed without importing them.

Activation is gated on the raw bytes of every module that contains a patch
target or a declared dependency: the plugin is pinned to one SGLang commit,
so any edit to such a module fails closed, including new conditional
redefinitions, module constants, helpers, line endings and indentation.

Each record also hashes the definition itself (raw bytes, decorators, and
the decorator/header lines of every enclosing class) for upgrade
diagnostics, and stores its line numbers so activation can check that the
live object is the definition that was fingerprinted. Writing a record
requires exactly one binding of each name along the qualified path.

Pinned records live in ``fingerprints/*.json`` (one file per patch module,
so parallel work does not share a file) and map a dotted target to
``{"file", "qualname", "kind", "first_line", "def_line", "sha256",
"module_sha256"}``.
"""

import ast
import hashlib
import json
from importlib import resources
from pathlib import Path

from sglang_qsa_hisparse.errors import FingerprintMismatch

_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bound_names(statement: ast.stmt) -> list[str]:
    names: list[str] = []
    if isinstance(statement, _DEFINITIONS):
        names.append(statement.name)
    elif isinstance(statement, (ast.Import, ast.ImportFrom)):
        for alias in statement.names:
            names.append(alias.asname or alias.name.split(".")[0])
    else:
        targets: list[ast.AST] = []
        if isinstance(statement, ast.Assign):
            targets = list(statement.targets)
        elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
            targets = [statement.target]
        elif isinstance(statement, ast.Delete):
            targets = list(statement.targets)
        elif isinstance(statement, (ast.For, ast.AsyncFor)):
            targets = [statement.target]
        elif isinstance(statement, (ast.With, ast.AsyncWith)):
            targets = [i.optional_vars for i in statement.items if i.optional_vars]
        for target in targets:
            for node in ast.walk(target):
                if isinstance(node, ast.Name):
                    names.append(node.id)
    return names


def _statements(scope: ast.AST):
    """Statements executed in ``scope``, entering compound statements only."""
    pending = list(getattr(scope, "body", ()))
    while pending:
        statement = pending.pop()
        yield statement
        if isinstance(statement, _SCOPES):
            continue
        for field in ("body", "orelse", "finalbody", "handlers", "cases"):
            for child in getattr(statement, field, ()) or ():
                if isinstance(child, (ast.ExceptHandler, ast.match_case)):
                    pending.extend(child.body)
                else:
                    pending.append(child)


def _locate(tree: ast.Module, qualname: str) -> tuple[ast.AST, list[ast.ClassDef]]:
    node: ast.AST = tree
    enclosing: list[ast.ClassDef] = []
    for part in qualname.split("."):
        bindings = [s for s in _statements(node) if part in _bound_names(s)]
        direct = [
            child
            for child in getattr(node, "body", ())
            if isinstance(child, _DEFINITIONS) and child.name == part
        ]
        if len(bindings) != 1 or len(direct) != 1:
            raise LookupError(
                f"{qualname}: {part!r} must have exactly one binding, a direct "
                f"definition; found {len(bindings)} binding(s), {len(direct)} direct"
            )
        if isinstance(node, ast.ClassDef):
            enclosing.append(node)
        node = direct[0]
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
    module (and therefore its definition) differ from the pinned revision.
    """
    root = installed_source_root() if source_root is None else Path(source_root)
    pinned = load_pinned() if pinned is None else pinned
    problems = []
    records = {}
    for name in names:
        expected = pinned.get(name)
        if expected is None:
            problems.append(f"{name}: no pinned fingerprint")
            continue
        try:
            actual = definition_record(root / expected["file"], expected["qualname"])
        except (LookupError, OSError, SyntaxError) as error:
            problems.append(f"{name}: {error}")
            continue
        if actual["module_sha256"] != expected["module_sha256"]:
            changed = "definition" if actual["sha256"] != expected["sha256"] else "module"
            problems.append(f"{name}: {changed} differs from the pinned revision")
            continue
        records[name] = {**expected, "path": str(root / expected["file"])}
    if problems:
        raise FingerprintMismatch("; ".join(problems))
    return records
