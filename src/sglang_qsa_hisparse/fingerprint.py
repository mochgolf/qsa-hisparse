"""Source and binding fingerprints of patch targets.

Activation is gated on the raw bytes of every module that contains a patch
target or a declared dependency: the plugin is pinned to one SGLang commit,
so any edit to such a module fails closed, including conditional
redefinitions, module constants, helpers, line endings and indentation.

Each record also has
- ``sha256``: raw bytes of the definition, its decorators, and the
  decorator/header lines of every enclosing class (upgrade diagnostics);
- ``chains``: per binding mode (``cpu``/``cuda``, because some SGLang modules
  choose implementations at import time from CUDA availability), the live
  binding as imported from the pinned checkout, from the attribute itself
  through every ``__wrapped__`` level (type and code location per level).
  Activation recomputes the chain for the current mode from the installed
  objects and requires an exact match, so a foreign wrapper, even one using
  ``functools.wraps``, cannot pass as the pinned definition. A missing chain
  for the current mode fails activation; regenerate it in that mode.

Writing a record requires exactly one binding of each name along the
qualified path, counting every Python binding form in that scope.

Pinned records live in ``fingerprints/*.json`` (one file per patch module, so
parallel work does not share a file).
"""

import ast
import hashlib
import importlib
import inspect
import json
import pkgutil
import sys
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
        for expression in statement.decorator_list:
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
        return names  # Annotation only: declares, does not bind.
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


def binding_count(scope: ast.AST, name: str) -> int:
    return sum(_statement_bindings(s).count(name) for s in _statements(scope))


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


# Live bindings ------------------------------------------------------------------


def _type_name(value: object) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def _location(filename: str, root: Path) -> str:
    path = Path(filename).resolve()
    try:
        return path.relative_to(root.resolve()).as_posix()
    except ValueError:
        return "<external>/" + path.name


def raw_attribute(target: str) -> object:
    owner_path, name = target.rsplit(".", 1)
    owner = pkgutil.resolve_name(owner_path)
    if isinstance(owner, type) and name in owner.__dict__:
        return owner.__dict__[name]
    return getattr(owner, name)


def live_chain(value: object, root: Path) -> list[dict]:
    """Describe ``value`` and every ``__wrapped__`` level beneath it."""
    chain: list[dict] = []
    seen: set[int] = set()
    current = value
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        entry: dict = {"type": _type_name(current)}
        if isinstance(current, type):
            entry["class"] = current.__qualname__
            try:
                entry["file"] = _location(inspect.getsourcefile(current), root)
            except TypeError:
                entry["file"] = None
        code = getattr(current, "__code__", None)
        if code is not None:
            entry["code"] = [_location(code.co_filename, root), code.co_firstlineno]
        chain.append(entry)
        if isinstance(current, (staticmethod, classmethod)):
            current = current.__func__
        elif isinstance(current, property):
            current = current.fget
        else:
            current = getattr(current, "__wrapped__", None)
    return chain


def binding_mode() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


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


def record(source_root: str | Path, target: str, previous: dict | None = None) -> dict:
    """Pinned record of ``target`` in the current binding mode.

    Imports ``target`` from ``source_root``. Chains recorded earlier for other
    modes are kept when the module bytes are unchanged.
    """
    root = Path(source_root).resolve()
    file, qualname = resolve_file(root, target)
    module_name = target[: -len(qualname) - 1]
    imported = sys.modules.get(module_name) or importlib.import_module(module_name)
    if Path(imported.__file__).resolve() != root / file:
        raise LookupError(
            f"{module_name} imports from {imported.__file__}, not {root / file}; "
            "put the pinned checkout first on sys.path"
        )
    data = {"file": file, **definition_record(root / file, qualname)}
    chains = {}
    if previous and previous.get("module_sha256") == data["module_sha256"]:
        chains.update(previous.get("chains", {}))
    chains[binding_mode()] = live_chain(raw_attribute(target), root)
    data["chains"] = dict(sorted(chains.items()))
    return data


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
    check_bindings: bool = True,
) -> dict[str, dict]:
    """Check module bytes and live bindings of ``names`` against the pin.

    Raises FingerprintMismatch on a missing record, different module bytes,
    or a live binding chain that differs from the pinned one.
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
        if check_bindings:
            mode = binding_mode()
            pinned_chain = expected.get("chains", {}).get(mode)
            if pinned_chain is None:
                problems.append(f"{name}: no pinned binding chain for mode {mode}")
                continue
            try:
                owner_path, member = name.rsplit(".", 1)
                owner = pkgutil.resolve_name(owner_path)
                chain = live_chain(raw_attribute(name), root)
            except (AttributeError, ImportError, ValueError) as error:
                problems.append(f"{name}: cannot resolve live binding: {error!r}")
                continue
            if isinstance(owner, type) and member not in owner.__dict__:
                problems.append(
                    f"{name}: inherited, not defined on {owner.__qualname__}; "
                    "live binding is not the pinned definition"
                )
                continue
            if chain != pinned_chain:
                problems.append(f"{name}: live binding is not the pinned definition")
                continue
        records[name] = expected
    if problems:
        raise FingerprintMismatch("; ".join(problems))
    return records
