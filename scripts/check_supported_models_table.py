#!/usr/bin/env python3
"""Keep the lazy adapter manifest, registrations, and support tables in sync."""
import ast
import pathlib
import re
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
API_REFERENCE_MD = REPO_ROOT / "docs" / "api_reference.md"
API_REFERENCE_ZH_MD = REPO_ROOT / "docs" / "api_reference.zh.md"
PYTHON_ROOT = REPO_ROOT / "python"
ADAPTERS_ROOT = REPO_ROOT / "python" / "rpu_backend" / "adapters"
MANIFEST_PATH = ADAPTERS_ROOT / "_manifest.py"
START_FENCE = "<!-- supported-models-start -->"
END_FENCE = "<!-- supported-models-end -->"


class CheckError(RuntimeError):
    """A statically checked source of truth is malformed or incomplete."""


def _parse(path: pathlib.Path) -> ast.Module:
    try:
        return ast.parse(path.read_text(), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise CheckError(f"cannot parse {path}: {exc}") from exc


def _extract_table_archs(path: pathlib.Path) -> list[str]:
    """Return list of HF arch strings from the first column of the table.

    The table looks like:
        | HF arch | Model | device adapter | Status | Notes |
        |---|---|---|---|---|
        | Qwen3ForCausalLM | ... |
    Skips header + separator rows.
    """
    try:
        content = path.read_text()
    except OSError as exc:
        raise CheckError(f"cannot read {path}: {exc}") from exc
    if START_FENCE not in content:
        # The public API reference uses a compact architecture list.
        anchor = "`rpu_backend.adapters._manifest.BUILTIN_ADAPTERS`"
        if anchor in content:
            body = content.split(anchor, 1)[1]
            paragraphs = body.split("\n\n", 2)
            declaration = next((part for part in paragraphs[:2] if "`" in part), "")
            archs = re.findall(r"`([^`]+)`", declaration)
            if not archs:
                raise CheckError(f"architecture list missing in {path}")
            if len(archs) != len(set(archs)):
                raise CheckError(f"duplicate HF architecture rows in {path}: {archs}")
            return archs
        raise CheckError(f"{START_FENCE!r} fence missing in {path}")
    if END_FENCE not in content:
        raise CheckError(f"{END_FENCE!r} fence missing in {path}")
    body = content.split(START_FENCE, 1)[1].split(END_FENCE, 1)[0]
    archs: list[str] = []
    for line in body.splitlines():
        line = line.strip()
        # markdown table rows start with "|"; skip header + separator
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not cells:
            continue
        first = cells[0]
        # Skip the header row + separator row.
        if first.lower() in {"hf arch", ""} or set(first) <= set("- :"):
            continue
        archs.append(first)
    if len(archs) != len(set(archs)):
        raise CheckError(f"duplicate HF arch rows in {path}: {archs}")
    return archs


def _manifest_entries() -> list[tuple[str, str, str]]:
    """Read ``BUILTIN_ADAPTERS`` without importing the package."""
    tree = _parse(MANIFEST_PATH)
    value_node: ast.expr | None = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "BUILTIN_ADAPTERS"
            for target in node.targets
        ):
            value_node = node.value
            break
    if value_node is None:
        raise CheckError(f"BUILTIN_ADAPTERS assignment missing in {MANIFEST_PATH}")
    try:
        raw_entries = ast.literal_eval(value_node)
    except (ValueError, SyntaxError) as exc:
        raise CheckError(f"BUILTIN_ADAPTERS is not pure literal data: {exc}") from exc

    entries: list[tuple[str, str, str]] = []
    for raw in raw_entries:
        if (
            not isinstance(raw, (tuple, list))
            or len(raw) != 3
            or not all(isinstance(item, str) and item for item in raw)
        ):
            raise CheckError(f"invalid BUILTIN_ADAPTERS entry: {raw!r}")
        entries.append((raw[0], raw[1], raw[2]))
    if not entries:
        raise CheckError("BUILTIN_ADAPTERS must not be empty")
    for index, label in enumerate(("HF arch", "module", "adapter target")):
        values = [entry[index] for entry in entries]
        if len(values) != len(set(values)):
            raise CheckError(f"duplicate {label} in BUILTIN_ADAPTERS: {values}")
    return entries


def _string_constants(tree: ast.Module) -> dict[str, str]:
    constants: dict[str, str] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = node.value.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            constants[node.target.id] = node.value.value
    return constants


def _resolve_string(node: ast.expr, constants: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


def _target_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _module_name(path: pathlib.Path) -> str:
    parts = list(path.relative_to(PYTHON_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _registered_builtin_entries() -> list[tuple[str, str, str]]:
    """Resolve public in-tree ``register_adapter`` calls statically."""
    entries: list[tuple[str, str, str]] = []
    for path in sorted(ADAPTERS_ROOT.rglob("*.py")):
        rel_parts = path.relative_to(ADAPTERS_ROOT).parts
        # Skip private modules/packages, but not a public package's
        # ``__init__.py``: package adapters register from that file.
        if any(part.startswith("_") for part in rel_parts[:-1]):
            continue
        if path.name.startswith("_") and path.name != "__init__.py":
            continue
        tree = _parse(path)
        constants = _string_constants(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_register = (
                isinstance(func, ast.Name) and func.id == "register_adapter"
            ) or (
                isinstance(func, ast.Attribute) and func.attr == "register_adapter"
            )
            if not is_register or len(node.args) < 2:
                continue
            arch = _resolve_string(node.args[0], constants)
            target = _target_name(node.args[1])
            if arch is None or target is None:
                raise CheckError(
                    f"registration in {path}:{node.lineno} must use a literal or "
                    "module string constant and a named adapter target"
                )
            entries.append((arch, _module_name(path), target))
    if len(entries) != len(set(entries)):
        raise CheckError(f"duplicate public adapter registrations: {entries}")
    return sorted(entries)


def main() -> int:
    try:
        manifest_entries = sorted(_manifest_entries())
        registration_entries = _registered_builtin_entries()
        tables = {
            API_REFERENCE_MD: sorted(_extract_table_archs(API_REFERENCE_MD)),
            API_REFERENCE_ZH_MD: sorted(_extract_table_archs(API_REFERENCE_ZH_MD)),
        }
    except CheckError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    ok = True
    if registration_entries != manifest_entries:
        ok = False
        print(
            "FAIL: lazy manifest and public adapter registrations differ.\n"
            f"  Manifest only:     {sorted(set(manifest_entries) - set(registration_entries))}\n"
            f"  Registration only: {sorted(set(registration_entries) - set(manifest_entries))}",
            file=sys.stderr,
        )

    manifest_archs = sorted(entry[0] for entry in manifest_entries)
    for path, table_archs in tables.items():
        if table_archs == manifest_archs:
            continue
        ok = False
        print(
            f"FAIL: supported-models table drift in {path}.\n"
            f"  Table archs:       {table_archs}\n"
            f"  Manifest archs:    {manifest_archs}\n"
            f"  Missing in table:  {sorted(set(manifest_archs) - set(table_archs))}\n"
            f"  Extra table rows:  {sorted(set(table_archs) - set(manifest_archs))}",
            file=sys.stderr,
        )

    if not ok:
        return 1
    print(
        "PASS: lazy manifest, public registrations, and EN/ZH support tables "
        f"match ({len(manifest_archs)} archs: {manifest_archs})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
