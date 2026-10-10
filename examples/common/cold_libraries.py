"""Cold runtime-library binding for the G05 examples, without backend imports."""

from __future__ import annotations

import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
LIBRARY_KEYS = {
    "native": "RPU_BACKEND_SO",
    "main": "RPU_KERNEL_LIB_PATH",
    "expansion": "RPU_KERNEL_LIB_PATH_EXPANSION",
}
AUTOLOAD = "TORCH_DEVICE_BACKEND_AUTOLOAD"
from examples.common import libraries as _runtime_libraries
_RUNTIME_LIBRARIES = vars(_runtime_libraries)


def validate_libraries(table):
    """Validate TOML structure only; --check-config accepts placeholder paths."""
    if table is None:
        return
    if (not isinstance(table, dict) or set(table) != set(LIBRARY_KEYS) or
            any(not isinstance(value, str) or not value.strip()
                for value in table.values())):
        raise ValueError("[libraries] must explicitly name native, main, and expansion together")


def _file_path(value):
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"runtime library must be a file: {path}")
    return str(path)


def _digest(path):
    return _RUNTIME_LIBRARIES["file_sha256"](path)


def bind_libraries(config, root=ROOT):
    """Bind explicit paths and cold overrides to the two published role hashes."""
    table = config.get("libraries")
    validate_libraries(table)
    if table is not None:
        selected = {LIBRARY_KEYS[role]: value for role, value in table.items()}
    else:
        selected = {key: os.environ[key] for key in LIBRARY_KEYS.values()
                    if key in os.environ}
        missing = [LIBRARY_KEYS[role] for role in ("main", "expansion")
                   if LIBRARY_KEYS[role] not in selected]
        if missing:
            defaults, _ = _RUNTIME_LIBRARIES["current_libraries"](Path(root))
            selected.update({key: defaults[key] for key in missing})
    return _RUNTIME_LIBRARIES["verify_libraries"](Path(root), selected)


def bind_cold_environment(env):
    """Set selectors atomically before torch can autoload the device backend."""
    if "rpu_backend" in sys.modules or "rpu_backend._cpp_ext" in sys.modules:
        raise ValueError("G05 configuration requires a fresh process before backend import")
    if (os.environ.get(AUTOLOAD, "0") != "0" or
            env.get(AUTOLOAD, "0") != "0"):
        raise ValueError(f"G05 configuration requires {AUTOLOAD}=0 before torch import")
    if "torch" in sys.modules and os.environ.get(AUTOLOAD) != "0":
        raise ValueError(f"torch was imported before {AUTOLOAD}=0; use a fresh process")
    if (os.environ.get("RPU_CHUNK_FORCE_UNSAFE", "0") != "0" or
            env.get("RPU_CHUNK_FORCE_UNSAFE", "0") != "0"):
        raise ValueError("RPU_CHUNK_FORCE_UNSAFE is incompatible with this entry")
    pending = dict(env)
    for key, value in env.items():
        if key in LIBRARY_KEYS.values():
            pending[key] = _file_path(value)
        if key in os.environ:
            existing = (_file_path(os.environ[key]) if key in LIBRARY_KEYS.values()
                        else os.environ[key])
            if existing != pending[key]:
                raise ValueError(f"conflicting cold environment: {key}")
    pending[AUTOLOAD] = "0"
    os.environ.update(pending)


def runtime_payloads(libraries):
    """Verify loaded REF snapshots and the native selected before backend import."""
    from rpu_backend import _cpp_ext

    actual = _cpp_ext.get_runtime_payload_paths()
    for role in ("main", "expansion"):
        selected = libraries.get(LIBRARY_KEYS[role], {})
        path = actual.get("oplib_roles", {}).get(role)
        if (not path or str(Path(path).expanduser().resolve()) != selected.get("path") or
                actual.get("oplib_payloads", {}).get(path, {}).get("sha256") != selected.get("sha256")):
            raise RuntimeError(f"loaded {role} payload does not match configuration")
    extension = _file_path(actual["extension"])
    native = {"path": extension, "sha256": _digest(extension)}
    if libraries.get(LIBRARY_KEYS["native"], native) != native:
        raise RuntimeError("loaded native extension does not match configuration")
    libraries[LIBRARY_KEYS["native"]] = native
    return actual
