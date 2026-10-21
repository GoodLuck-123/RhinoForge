"""Resolve manifest-bound runtime libraries without importing the backend."""

from __future__ import annotations

import hashlib
from importlib import metadata
import os
import json
from pathlib import Path
import re
import sysconfig


LIBRARY_ROLES = (
    ("RPU_KERNEL_LIB_PATH", "rhino_ops_main", "rhinoOpLib_current.ref"),
    ("RPU_KERNEL_LIB_PATH_EXPANSION", "rhino_ops_expansion", "rhinoExpansionOpLib_current.ref"),
)


def file_sha256(path: Path | str) -> str:
    """Hash a file in bounded memory, including on supported Python 3.10."""
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def verify_backend_source(root: Path, origin: str | Path) -> Path:
    """Accept this checkout or its matching installed wheel, never an overlay."""
    root, imported = Path(root).resolve(), Path(origin).resolve()
    source_package = root / "python/rpu_backend"
    if imported == (source_package / "__init__.py").resolve():
        return imported

    def fail(detail: str) -> None:
        raise RuntimeError(f"wrong source checkout or installed wheel imported: {imported}; {detail}")

    package_dirs = {Path(sysconfig.get_path(kind)).resolve() / "rpu_backend"
                    for kind in ("purelib", "platlib")}
    if imported.parent not in package_dirs or imported.name != "__init__.py":
        fail("expected this checkout or the current interpreter's site-packages")
    try:
        distribution = metadata.distribution("rhinoforge")
    except metadata.PackageNotFoundError:
        fail("rhinoforge distribution metadata is missing")
    files = list(distribution.files or ())
    entry = next((item for item in files if str(item) == "rpu_backend/__init__.py"), None)
    if entry is None or Path(distribution.locate_file(entry)).resolve() != imported:
        fail("the imported package is not owned by the rhinoforge distribution")
    if distribution.version != (root / "VERSION").read_text().strip():
        fail("rhinoforge distribution version differs from the example source")
    # RECORD lists only installed runtime modules, so excluded tests/debug files
    # do not make a matching wheel fail. Source snapshots bind the Python bytes
    # too; version alone cannot distinguish different builds of one release.
    if source_package.is_dir():
        for item in files:
            relative = Path(str(item))
            if relative.parts[:1] != ("rpu_backend",) or relative.suffix != ".py":
                continue
            installed = Path(distribution.locate_file(item)).resolve()
            expected = root / "python" / relative
            if (not installed.is_file() or not expected.is_file()
                    or file_sha256(installed) != file_sha256(expected)):
                fail(f"installed Python differs from source: {relative}")
    return imported


def verify_libraries(root: Path, selected: dict) -> tuple[dict, dict]:
    """Require each selected role to contain its manifest-published bytes.

    Installed copies may have different paths. Neither explicit paths nor a
    complete environment override can replace the checkout's role identities.
    """
    manifest = json.loads((Path(root) / "release/runtime-dependencies.json").read_text())
    artifacts = manifest["artifacts"]
    expected = {}
    for key, role, filename in LIBRARY_ROLES:
        artifact = artifacts[role]
        if artifact["path"] != filename:
            raise ValueError(f"{role} must name its published current library: {filename}")
        digest = artifact.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{role} requires a published SHA-256")
        if key not in selected:
            raise ValueError(f"missing runtime library role: {key}")
        expected[key] = digest
    if len(set(expected.values())) != len(LIBRARY_ROLES):
        raise ValueError("main and expansion libraries must use distinct payloads")
    env, records = {}, {}
    for key, value in selected.items():
        path = Path(value).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"runtime library must be a file: {path}")
        digest = file_sha256(path)
        if key in expected and digest != expected[key]:
            raise ValueError(f"{key} library checksum mismatch: {path}")
        env[key] = str(path)
        records[key] = {"path": str(path), "sha256": digest}
    return env, records


def current_libraries(root: Path) -> tuple[dict, dict]:
    """Resolve the two external assets selected for this source release."""
    selected = {}
    for key, role, filename in LIBRARY_ROLES:
        value = os.environ.get(key)
        if not value:
            raise ValueError(f"set {key} to the matching external {filename}")
        selected[key] = value
    return verify_libraries(root, selected)
