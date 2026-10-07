"""Source and binding fingerprints of patch targets.

Activation is gated on the raw bytes of every module that contains a patch
target or a declared dependency: the plugin is pinned to one SGLang commit,
so any edit to such a module fails closed, including conditional
redefinitions, module constants, helpers, line endings and indentation.

Each record also has
- ``sha256``: raw bytes of the definition, its decorators, and the
  decorator/header lines of every enclosing class (upgrade diagnostics);
- ``chains``: per binding mode (``cpu``/``cuda``, because some SGLang modules
  choose implementations at import time from CUDA availability), a
  description of the live binding as imported from the pinned checkout: the
  attribute and every level beneath it (descriptor, ``property.fget``,
  ``__wrapped__``), with each function's code location, closure, defaults
  and keyword defaults (recursively), class members, ``lru_cache``
  parameters, ``partial`` arguments, and operator dispatcher registrations.
  Levels or values that cannot be identified raise ``Undescribable`` when the
  record is written. Activation recomputes the chain for the current mode and
  requires an exact match; ``patching.verify_final`` recomputes it again.

Threat model: see ``docs/PLAN.md`` ("Activation guarantees"). These checks
detect SGLang source drift from the pin and replacement, wrapping or
identifiable in-place mutation of protected bindings up to final
verification. They do not detect later mutations, native-library changes
(versions are recorded by the launcher), or state marked unchecked.

Writing a record requires exactly one binding of each name along the
qualified path, counting every Python binding form in that scope.

Pinned records live in ``fingerprints/*.json`` (one file per patch module, so
parallel work does not share a file).
"""

import ast
import enum
import functools
import re
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


# Live bindings ------------------------------------------------------------------
#
# A binding chain describes the attribute and every level beneath it
# (descriptor ``__func__``, ``property.fget``, ``__wrapped__``). Each level must
# be a kind whose behavior is identified by the description; anything else is
# rejected when the record is written (patches may opt out per name with
# ``unchecked_bindings``, keeping module bytes pinned). Functions whose code is
# outside the pinned tree (library decorators) also describe their closure and
# defaults, which is where decorator factories keep their configuration.

CHAIN_FORMAT = 2  # Bump whenever live_chain's description changes.
_SCALARS = (type(None), bool, int, float, complex, str, bytes)
_MAX_ITEMS = 64


class Undescribable(LookupError):
    """A binding level whose behavior the chain cannot identify."""


def _type_name(value: object) -> str:
    kind = value if isinstance(value, type) else type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def _location(filename: str, root: Path) -> tuple[str, bool]:
    path = Path(filename).resolve()
    try:
        return path.relative_to(root.resolve()).as_posix(), True
    except ValueError:
        return "<external>/" + path.name, False


def _code(function, root: Path) -> dict:
    code = function.__code__
    location, internal = _location(code.co_filename, root)
    entry = {
        "code": [location, code.co_firstlineno],
        "function": f"{function.__module__}.{function.__qualname__}",
    }
    if internal:
        entry["file_sha256"] = _sha256(Path(code.co_filename).read_bytes())
    return entry, internal


def _function_state(function, root: Path, depth: int) -> dict:
    """Code identity plus everything a function object carries besides code."""
    entry, _ = _code(function, root)
    entry["closure"] = [
        [name, describe_value(cell.cell_contents, root, depth)]
        for name, cell in zip(function.__code__.co_freevars, function.__closure__ or ())
    ]
    entry["defaults"] = describe_value(function.__defaults__, root, depth)
    entry["kwdefaults"] = describe_value(function.__kwdefaults__ or {}, root, depth)
    return entry


def describe_value(value: object, root: Path, depth: int = 0) -> object:
    """Identity of a closure cell, default or partial argument."""
    if depth > 4:
        raise Undescribable("nested too deeply")
    if isinstance(value, _SCALARS):
        return repr(value)
    try:
        import torch

        if isinstance(value, (torch.dtype, torch.device)):
            return repr(value)
    except ImportError:
        pass
    if isinstance(value, type):
        return {"type": _type_name(value)}
    if inspect.ismodule(value):
        return {"module": value.__name__}
    if isinstance(value, enum.Enum):
        return {"enum": f"{_type_name(value)}.{value.name}"}
    if inspect.isfunction(value):
        return _function_state(value, root, depth + 1)
    if inspect.ismethod(value):
        return {
            "method": f"{_type_name(value.__self__)}.{value.__func__.__qualname__}",
            "self": describe_value(getattr(value.__self__, "__dict__", {}), root, depth + 1),
        }
    if inspect.isbuiltin(value):
        return {"builtin": f"{getattr(value, '__module__', None)}.{value.__qualname__}"}
    if isinstance(value, (tuple, list, frozenset, set)) and len(value) <= _MAX_ITEMS:
        items = [describe_value(v, root, depth + 1) for v in value]
        return {_type_name(value): sorted(map(repr, items)) if isinstance(value, (set, frozenset)) else items}
    if isinstance(value, dict) and len(value) <= _MAX_ITEMS and all(isinstance(k, str) for k in value):
        return {"dict": {k: describe_value(v, root, depth + 1) for k, v in sorted(value.items())}}
    if isinstance(value, functools.partial):
        return {
            "partial": describe_value(value.func, root, depth + 1),
            "args": describe_value(value.args, root, depth + 1),
            "keywords": describe_value(value.keywords, root, depth + 1),
        }
    raise Undescribable(f"cannot identify value of type {_type_name(value)}")


def _level(current: object, root: Path) -> tuple[dict, object]:
    """Describe one level and return the next level beneath it (or None)."""
    entry: dict = {"type": _type_name(current)}
    if isinstance(current, (staticmethod, classmethod)):
        return entry, current.__func__
    if isinstance(current, property):
        return entry, current.fget
    if isinstance(current, type):
        location = None
        try:
            location, internal = _location(inspect.getsourcefile(current), root)
            if internal:
                entry["file_sha256"] = _sha256((root / location).read_bytes())
        except TypeError:
            pass
        entry.update(type_name=_type_name(current), file=location)
        entry["members"] = {
            name: live_chain(member, root)
            for name, member in sorted(current.__dict__.items())
            if inspect.isfunction(member)
            or isinstance(member, (staticmethod, classmethod, property, functools.partial))
        }
        return entry, None
    if inspect.isfunction(current):
        entry.update(_function_state(current, root, 0))
        return entry, getattr(current, "__wrapped__", None)
    if isinstance(current, functools.partial):
        entry["partial"] = describe_value(current, root)
        return entry, None
    if type(current).__name__ == "_lru_cache_wrapper":
        entry["cache_parameters"] = current.cache_parameters()
        return entry, current.__wrapped__
    if type(current).__qualname__ in ("OpOverloadPacket", "OpOverload"):
        name = getattr(current, "_qualified_op_name", None) or getattr(current, "name", None)
        if callable(name):
            name = name()
        if not name:
            raise Undescribable("operator without a qualified name")
        entry["op"] = str(name)
        entry["dispatch"] = _dispatch_registrations(str(name), root)
        return entry, None
    raise Undescribable(f"cannot identify binding level of type {_type_name(current)}")


def _dispatch_registrations(op: str, root: Path) -> list[str]:
    """Dispatcher kernel registrations of ``op`` with normalized locations."""
    import torch

    def normalize(match):
        location, _ = _location(match.group(1), root)
        return f"registered at {location}:{match.group(2)}"

    try:
        dump = torch._C._dispatch_dump(op)
    except RuntimeError as error:
        raise Undescribable(f"no dispatcher entry for {op}: {error}") from error
    lines = []
    for line in dump.splitlines():
        if line.startswith(("name:", "schema:", "debug:", "alias analysis")):
            lines.append(line)
        elif "registered at" in line:
            lines.append(re.sub(r"registered at (\S+?):(\d+)", normalize, line))
    return lines


def raw_attribute(target: str) -> object:
    owner_path, name = target.rsplit(".", 1)
    owner = pkgutil.resolve_name(owner_path)
    if isinstance(owner, type) and name in owner.__dict__:
        return owner.__dict__[name]
    return getattr(owner, name)


def live_chain(value: object, root: Path) -> list[dict]:
    """Describe ``value`` and every level beneath it; raise if unidentifiable."""
    chain: list[dict] = []
    seen: set[int] = set()
    current = value
    while current is not None:
        if id(current) in seen:
            raise Undescribable("cyclic wrapper chain")
        seen.add(id(current))
        entry, current = _level(current, root)
        chain.append(entry)
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
    if (
        previous
        and previous.get("module_sha256") == data["module_sha256"]
        and previous.get("chain_format") == CHAIN_FORMAT
    ):
        chains.update(previous.get("chains", {}))
    chains[binding_mode()] = live_chain(raw_attribute(target), root)
    data["chain_format"] = CHAIN_FORMAT
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
            if expected.get("chain_format") != CHAIN_FORMAT:
                problems.append(f"{name}: binding chain format is stale; refresh records")
                continue
            pinned_chain = expected.get("chains", {}).get(mode)
            if pinned_chain is None:
                problems.append(f"{name}: no pinned binding chain for mode {mode}")
                continue
            try:
                owner_path, member = name.rsplit(".", 1)
                owner = pkgutil.resolve_name(owner_path)
                chain = live_chain(raw_attribute(name), root)
            except (AttributeError, ImportError, ValueError, LookupError) as error:
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
