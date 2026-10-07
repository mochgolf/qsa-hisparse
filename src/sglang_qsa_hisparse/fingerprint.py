"""Source fingerprints of patch targets, computed without importing them.

Activation is gated on the raw bytes of every module that contains a patch
target or a declared dependency: the plugin is pinned to one SGLang commit,
so any edit to such a module fails closed (conditional redefinitions,
constants, helpers, line endings, indentation).

Each record also hashes the definition itself (raw bytes, decorators, and the
decorator/header lines of every enclosing class) to report which definitions
changed when the pin is upgraded. Writing a record requires exactly one
binding of each name along the qualified path, so the fingerprinted
definition is the one the module binds.

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
_NEW_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_BODY_FIELDS = ("body", "orelse", "finalbody", "handlers", "cases")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# Binding analysis -------------------------------------------------------------


def _expression_bindings(node: ast.AST, names: list[str], in_comprehension: bool) -> None:
    """Names an expression binds in the enclosing scope."""
    if isinstance(node, ast.NamedExpr):
        names.append(node.target.id)  # Walrus binds outside comprehensions too.
        _expression_bindings(node.value, names, in_comprehension)
        return
    if isinstance(node, ast.Lambda):
        # Defaults are evaluated in the enclosing scope; the body is not.
        arguments = node.args
        for default in arguments.defaults + [d for d in arguments.kw_defaults if d]:
            _expression_bindings(default, names, in_comprehension)
        return
    if isinstance(node, _NEW_SCOPES):
        return
    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
        if not in_comprehension:
            names.append(node.id)
        return
    inner = in_comprehension or isinstance(node, _COMPREHENSIONS)
    for child in ast.iter_child_nodes(node):
        _expression_bindings(child, names, inner)


def _pattern_bindings(pattern: ast.AST, names: list[str]) -> None:
    for node in ast.walk(pattern):
        if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            names.append(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.append(node.rest)


def _statement_bindings(statement: ast.stmt) -> list[str]:
    """Names bound in the current scope by ``statement`` itself (not its body)."""
    names: list[str] = []
    if isinstance(statement, _DEFINITIONS):
        names.append(statement.name)
        # Expressions evaluated in the enclosing scope when the definition runs.
        header = list(statement.decorator_list)
        if isinstance(statement, ast.ClassDef):
            header += statement.bases + [k.value for k in statement.keywords]
        else:
            arguments = statement.args
            header += arguments.defaults + [d for d in arguments.kw_defaults if d]
            every = arguments.posonlyargs + arguments.args + arguments.kwonlyargs
            every += [a for a in (arguments.vararg, arguments.kwarg) if a]
            header += [a.annotation for a in every if a.annotation is not None]
            if statement.returns is not None:
                header.append(statement.returns)
        for expression in header:
            _expression_bindings(expression, names, False)
        return names
    if isinstance(statement, (ast.Import, ast.ImportFrom)):
        for alias in statement.names:
            if alias.name != "*":
                names.append(alias.asname or alias.name.split(".")[0])
        return names
    if isinstance(statement, (ast.Global, ast.Nonlocal)):
        return names
    if isinstance(statement, ast.AnnAssign) and statement.value is None:
        # Annotation only: the target is declared, not bound, but the
        # annotation expression is evaluated (and may bind via walrus).
        _expression_bindings(statement.annotation, names, False)
        return names
    if isinstance(statement, ast.Try) or type(statement).__name__ == "TryStar":
        for handler in statement.handlers:
            if handler.name:
                names.append(handler.name)
            if handler.type is not None:
                _expression_bindings(handler.type, names, False)
        return names
    if isinstance(statement, ast.Match):
        _expression_bindings(statement.subject, names, False)
        for case in statement.cases:
            _pattern_bindings(case.pattern, names)
            if case.guard is not None:
                _expression_bindings(case.guard, names, False)
        return names
    for field, value in ast.iter_fields(statement):
        if field in _BODY_FIELDS:
            continue
        for node in value if isinstance(value, list) else [value]:
            if isinstance(node, ast.AST):
                _expression_bindings(node, names, False)
    return names


def _statements(scope: ast.AST):
    """Statements executed in ``scope``, entering compound statements only."""
    pending = list(getattr(scope, "body", ()))
    while pending:
        statement = pending.pop()
        yield statement
        if isinstance(statement, _NEW_SCOPES):
            continue
        for field in _BODY_FIELDS:
            for child in getattr(statement, field, ()) or ():
                if isinstance(child, (ast.ExceptHandler, ast.match_case)):
                    pending.extend(child.body)
                else:
                    pending.append(child)


def _global_rebindings(module: ast.Module, name: str) -> int:
    """Bindings of ``name`` in nested scopes that declare it ``global``."""
    count = 0
    for node in ast.walk(module):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        statements = list(_statements(node))
        if any(isinstance(s, ast.Global) and name in s.names for s in statements):
            count += sum(_statement_bindings(s).count(name) for s in statements)
    return count


def binding_count(scope: ast.AST, name: str) -> int:
    count = sum(_statement_bindings(s).count(name) for s in _statements(scope))
    if isinstance(scope, ast.Module):
        count += _global_rebindings(scope, name)
    return count


def _locate(tree: ast.Module, qualname: str) -> tuple[ast.AST, list[ast.ClassDef]]:
    node: ast.AST = tree
    enclosing: list[ast.ClassDef] = []
    for part in qualname.split("."):
        count = binding_count(node, part)
        direct = [
            child
            for child in getattr(node, "body", ())
            if isinstance(child, _DEFINITIONS) and child.name == part
        ]
        if count != 1 or len(direct) != 1:
            raise LookupError(
                f"{qualname}: {part!r} must have exactly one binding, a direct "
                f"definition; found {count} binding(s), {len(direct)} direct"
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
    module differ from the pinned revision.
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
        records[name] = expected
    if problems:
        raise FingerprintMismatch("; ".join(problems))
    return records
