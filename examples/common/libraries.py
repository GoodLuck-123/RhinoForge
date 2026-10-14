"""Resolve manifest-bound runtime libraries without importing the backend."""

from __future__ import annotations

import hashlib
import os
import json
from pathlib import Path
import re


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
