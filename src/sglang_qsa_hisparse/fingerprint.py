"""Source fingerprints of patch targets, computed without importing them.

A fingerprint is the SHA-256 of a definition's dedented source text,
including decorators, located by qualified name in the parsed module file.
The same routine runs offline against the pinned checkout and at activation
against the installed SGLang, so any upstream edit to a patched definition
fails activation instead of silently mixing old and new code.

Pinned records live in ``fingerprints/*.json`` (one file per patch module, so
parallel work does not share a file) and map a dotted target to
``{"file": "sglang/...py", "qualname": "...", "sha256": "..."}``.
"""

import ast
import hashlib
import json
import textwrap
from importlib import resources
from pathlib import Path

from sglang_qsa_hisparse.errors import FingerprintMismatch

_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _locate(tree: ast.Module, qualname: str) -> ast.AST:
    node: ast.AST = tree
    for part in qualname.split("."):
        matches = [
            child
            for child in getattr(node, "body", ())
            if isinstance(child, _DEFINITIONS) and child.name == part
        ]
        if len(matches) != 1:
            raise LookupError(
                f"{qualname}: expected one definition of {part!r} directly in "
                f"its parent, found {len(matches)}"
            )
        node = matches[0]
    return node


def source_fingerprint(path: str | Path, qualname: str) -> str:
    text = Path(path).read_text(encoding="utf-8")
    node = _locate(ast.parse(text), qualname)
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    lines = text.splitlines()[start - 1 : node.end_lineno]
    segment = textwrap.dedent("\n".join(lines))
    return hashlib.sha256(segment.encode("utf-8")).hexdigest()


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


def record(source_root: str | Path, target: str) -> dict[str, str]:
    file, qualname = resolve_file(source_root, target)
    return {
        "file": file,
        "qualname": qualname,
        "sha256": source_fingerprint(Path(source_root) / file, qualname),
    }


def installed_source_root() -> Path:
    import sglang  # Already imported whenever SGLang loads this plugin.

    return Path(sglang.__file__).resolve().parent.parent


def load_pinned() -> dict[str, dict[str, str]]:
    pinned: dict[str, dict[str, str]] = {}
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
    pinned: dict[str, dict[str, str]] | None = None,
) -> None:
    """Raise FingerprintMismatch unless every named definition matches the pin."""
    root = installed_source_root() if source_root is None else Path(source_root)
    pinned = load_pinned() if pinned is None else pinned
    problems = []
    for name in names:
        expected = pinned.get(name)
        if expected is None:
            problems.append(f"{name}: no pinned fingerprint")
            continue
        try:
            actual = source_fingerprint(root / expected["file"], expected["qualname"])
        except (LookupError, OSError, SyntaxError) as error:
            problems.append(f"{name}: {error}")
            continue
        if actual != expected["sha256"]:
            problems.append(f"{name}: source differs from the pinned revision")
    if problems:
        raise FingerprintMismatch("; ".join(problems))
