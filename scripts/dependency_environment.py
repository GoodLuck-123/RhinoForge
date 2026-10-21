#!/usr/bin/env python3
"""Check third-party dependencies before building RhinoForge (no device access).

Run with the environment's Python. --print-requirements emits only third-party
requirements; --audit-package additionally rejects private/application payloads.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import ipaddress
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from urllib.parse import urlsplit

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
import tomllib

SCHEMA = "rhinoforge-dependency-env/v1"
ENVIRONMENT_ID = "deps-2026.10.1"
EXTRAS = ["audio", "dev", "vision", "vla"]
TOOL_REQUIREMENTS = ["cmake>=3.22", "ninja", "conda-pack>=0.8", "pip", "wheel",
                     "packaging>=24.2,<26", "setuptools>=71,<81"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_requirement(value: str) -> str:
    req = Requirement(value)
    if req.url:
        raise ValueError(f"Dependency recipe must use version constraints, not URLs: {req.name}")
    result = canonicalize_name(req.name)
    if req.extras:
        result += "[" + ",".join(sorted(canonicalize_name(x) for x in req.extras)) + "]"
    result += str(req.specifier)
    if req.marker:
        result += "; " + str(req.marker)
    return result


def requirement_contract(source: Path) -> dict:
    with (source / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)
    optional = project["project"].get("optional-dependencies", {})
    missing = sorted(set(EXTRAS) - optional.keys())
    if missing:
        raise ValueError("Missing full-profile extras: " + ", ".join(missing))
    constraint_path = source / "constraints/security.txt"
    constraints = []
    for line in constraint_path.read_text().splitlines():
        line = line.split(" #", 1)[0].strip()
        if line and not line.startswith("#"):
            constraints.append(normalized_requirement(line))
    normalize = lambda items: sorted(set(normalized_requirement(x) for x in items))
    return {
        "profile": "full", "extras": EXTRAS,
        "requires_python": str(SpecifierSet(project["project"]["requires-python"])),
        "runtime": normalize(project["project"].get("dependencies", [])),
        "optional": {extra: normalize(optional[extra]) for extra in EXTRAS},
        "build": normalize(project["build-system"]["requires"]),
        "tools": normalize(TOOL_REQUIREMENTS),
        "constraints": sorted(set(constraints)),
    }


def requirements_sha256(contract: dict) -> str:
    return hashlib.sha256(json.dumps(contract, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def selected_requirements(contract: dict) -> list[str]:
    return sorted(set(contract["runtime"] + contract["build"] + contract["tools"] +
                      [value for values in contract["optional"].values() for value in values]))


def check_requirements(contract: dict, installed: dict[str, str], *,
                       python_version: str | None = None, marker_environment: dict | None = None) -> None:
    installed = {canonicalize_name(name): version for name, version in installed.items()}
    python_version = python_version or platform.python_version()
    if python_version not in SpecifierSet(contract["requires_python"]):
        raise ValueError(f"Python {python_version} does not satisfy {contract['requires_python']}")
    markers = default_environment()
    if marker_environment:
        markers.update(marker_environment)
    errors = []
    for group, values in (("requirement", selected_requirements(contract)),
                          ("constraint", contract["constraints"])):
        for value in values:
            req = Requirement(value)
            if req.marker and not any(req.marker.evaluate({**markers, "extra": extra})
                                      for extra in [""] + contract["extras"]):
                continue
            found = installed.get(canonicalize_name(req.name))
            if found is None:
                if group == "requirement":
                    errors.append(f"missing {req}")
            elif found not in req.specifier:
                errors.append(f"{req}: installed {found}")
    if errors:
        raise ValueError("Unsatisfied RhinoForge dependencies: " + "; ".join(errors))


def public_url(value: str) -> bool:
    """Local paths/credential URLs cannot enter public provenance records."""
    parsed = urlsplit(value)
    if parsed.scheme not in ("https", "http") or not parsed.hostname or parsed.username or parsed.password:
        return False
    host = parsed.hostname.lower()
    if host == "localhost" or "." not in host or host.endswith((".local", ".internal")):
        return False
    try:
        if not ipaddress.ip_address(host).is_global:
            return False
    except ValueError:
        pass
    return not parsed.query and not parsed.fragment


def forbidden_path(relative: str) -> bool:
    parts = Path(relative).parts
    lower = [part.lower() for part in parts]
    base = lower[-1] if lower else ""
    if any(part in {"rpu_backend", "rhinoforge", "rhino_launch", "launch_kernel",
                    ".git", ".ssh", ".aws", ".huggingface"} for part in lower):
        return True
    if base in {".netrc", ".pypirc", ".env", ".bash_history", "credentials", "token",
                ".condarc", "pip.conf", ".npmrc"}:
        return True
    if any(part.startswith(("rhinoforge-", "rpu_backend-", "__editable__")) for part in lower):
        return True
    if base.startswith(("libhxmem.so", "libhxil.so", "librpu_api.so", "librhino_launch.so",
                        "rpu_backend.", "rhinooplib", "rhinoexpansionoplib")):
        return True
    if base.endswith((".safetensors", ".gguf", ".egg-link")):
        return True
    if lower and lower[0] in {"source", "runtime", "models", "checkpoints", ".cache"}:
        return True
    return base in {"pytorch_model.bin", "model.safetensors.index.json", "id_rsa", "id_ed25519"}


def audit_prefix(prefix: Path) -> None:
    """Reject application/private files, external links and editable path hooks."""
    prefix = prefix.resolve(strict=True)
    for root, dirs, files in os.walk(prefix):
        for name in dirs + files:
            path = Path(root) / name
            relative = path.relative_to(prefix).as_posix()
            if forbidden_path(relative):
                raise ValueError(f"Excluded payload in dependency environment: {relative}")
            if path.is_symlink() and not path.resolve().is_relative_to(prefix):
                raise ValueError(f"External environment symlink: {relative}")
            if path.is_file() and path.suffix == ".pth":
                for line in path.read_text(errors="replace").splitlines():
                    if "__editable__" in line or "rpu_backend" in line or "rhinoforge" in line:
                        raise ValueError(f"Editable/application path hook: {relative}")
                    if line and not line.startswith(("#", "import ", "import\t")):
                        target = path.parent / line.strip()
                        if not target.resolve().is_relative_to(prefix):
                            raise ValueError(f"External Python path hook: {relative}")


def profile_runtime_files(prefix: Path) -> list[str]:
    files = [prefix / "bin/sox", prefix / "bin/ffmpeg"]
    missing = [p.relative_to(prefix).as_posix() for p in files
               if not p.is_file() or not os.access(p, os.X_OK)]
    libraries = sorted(p for p in (prefix / "lib").glob("libsndfile.so*") if p.is_file())
    if not libraries:
        missing.append("lib/libsndfile.so*")
    if missing:
        raise ValueError("Full profile requires SoX, FFmpeg and libsndfile: " + ", ".join(missing))
    return [p.relative_to(prefix).as_posix() for p in files + libraries]


def native_inventory(path: Path | None, prefix: Path, *, verify_files: bool) -> list[dict]:
    if path is None:
        return []
    data = json.loads(path.read_text())
    if data.get("schema") != "rhinoforge-native-dependencies/v1":
        raise ValueError("Unsupported native third-party inventory schema")
    packages = data.get("packages", [])
    names = set()
    for package in packages:
        name = canonicalize_name(package["name"])
        if name in names or name in {"rhinoforge", "rpu-backend", "rhino-launch"}:
            raise ValueError("Invalid/duplicate native dependency name")
        names.add(name)
        if not package.get("version") or not public_url(package.get("source_url", "")):
            raise ValueError(f"Public native dependency source/version required: {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", package.get("source_sha256", "")):
            raise ValueError(f"Native dependency source SHA256 required: {name}")
        if not package.get("files"):
            raise ValueError(f"Native dependency file inventory is empty: {name}")
        for relative, digest in package["files"].items():
            target = prefix / relative
            if (Path(relative).is_absolute() or ".." in Path(relative).parts
                    or "\\" in relative or forbidden_path(relative)
                    or not target.resolve().is_relative_to(prefix.resolve())):
                raise ValueError(f"Excluded native dependency file: {relative}")
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"Invalid native dependency file digest: {relative}")
            if verify_files and (not target.is_file() or sha256(target) != digest):
                raise ValueError(f"Native dependency file SHA256 mismatch: {relative}")
    return packages


def audit_managed_files(prefix: Path, native: list[dict] | None = None) -> None:
    """Refuse untracked payloads even when their names look innocuous.

    Conda owns files in its package records; pip owns wheel RECORD entries.
    Bytecode and Conda's prefix/history bookkeeping are generated locally.
    """
    prefix = prefix.resolve(strict=True)
    owned = {"conda-meta/history", "conda-meta/pinned", ".nonadmin"}
    for package in native or []:
        owned.update(package["files"])
    for path in (prefix / "conda-meta").glob("*.json"):
        owned.add(path.relative_to(prefix).as_posix())
        data = json.loads(path.read_text())
        for name in data.get("files", []):
            location = Path(os.path.abspath(prefix / name))
            if not location.is_relative_to(prefix):
                raise ValueError(f"Conda package record escapes environment: {path.name}")
            owned.add(location.relative_to(prefix).as_posix())
    for dist in metadata.distributions():
        for entry in dist.files or []:
            location = Path(os.path.abspath(dist.locate_file(entry)))
            if not location.is_relative_to(prefix):
                raise ValueError(f"Python package record escapes environment: {dist.metadata['Name']}")
            owned.add(location.relative_to(prefix).as_posix())
    for root, _, files in os.walk(prefix):
        for name in files:
            path = Path(root) / name
            relative = path.relative_to(prefix).as_posix()
            if relative in owned:
                continue
            if path.suffix == ".pyc" and "__pycache__" in path.parts:
                continue
            raise ValueError(f"Unmanaged file in dependency environment: {relative}")


def package_inventory(prefix: Path, *, audit: bool = False) -> dict:
    packages = []
    seen = set()
    for dist in metadata.distributions():
        name = canonicalize_name(dist.metadata["Name"])
        if audit and not Path(dist.locate_file("")).resolve().is_relative_to(prefix.resolve()):
            raise ValueError(f"Distribution is outside the selected environment: {name}")
        if name in seen:
            raise ValueError(f"Duplicate Python distribution: {name}")
        seen.add(name)
        direct = dist.read_text("direct_url.json")
        origin = json.loads(direct) if direct else None
        if audit and (name in {"rhinoforge", "rpu-backend"} or
                      (origin and origin.get("dir_info", {}).get("editable"))):
            raise ValueError(f"Application/editable distribution cannot be packed: {name}")
        item = {"name": name, "version": dist.version,
                "installer": (dist.read_text("INSTALLER") or "").strip(),
                "requires": sorted(dist.requires or [])}
        if origin and origin.get("url") and public_url(origin["url"]):
            item["origin"] = origin
        packages.append(item)
    conda = []
    for path in sorted((prefix / "conda-meta").glob("*.json")):
        data = json.loads(path.read_text())
        item = {key: data[key] for key in ("name", "version", "build", "build_number", "subdir",
                                          "sha256", "md5", "license", "license_family", "depends")
                if key in data}
        for key in ("url", "channel"):
            if key in data:
                if not public_url(data[key]):
                    raise ValueError(f"Non-public Conda {key}: {path.name}")
                item[key] = data[key]
        conda.append(item)
    return {"pip": sorted(packages, key=lambda item: item["name"]), "conda": conda}


def check_binding(contract: dict, manifest: dict, binding: dict | None = None,
                  manifest_path: Path | None = None) -> None:
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Expected a dependency-only environment manifest")
    digest = requirements_sha256(contract)
    if manifest.get("requirements_sha256") != digest:
        raise ValueError("Source dependency requirements differ from the environment recipe")
    if manifest.get("profile") != "full" or sorted(manifest.get("extras", [])) != EXTRAS:
        raise ValueError("Environment does not contain the full dependency profile")
    if binding is not None:
        for field in ("schema", "environment_id", "profile", "requirements_sha256"):
            if not binding.get(field) or binding[field] != manifest.get(field):
                raise ValueError(f"Source environment binding differs: {field}")
        expected = binding.get("manifest_sha256")
        if expected and (manifest_path is None or sha256(manifest_path) != expected):
            raise ValueError("Published environment manifest SHA256 mismatch")


def inspect_environment(source: Path, prefix: Path, *, audit: bool = False,
                        native_path: Path | None = None) -> dict:
    if Path(sys.prefix).resolve() != prefix.resolve():
        raise ValueError("Run this checker with --prefix/bin/python")
    if sys.version_info[:2] != (3, 12):
        raise ValueError("Dependency environment requires Python 3.12")
    contract = requirement_contract(source)
    inventory = package_inventory(prefix, audit=audit)
    inventory["native"] = native_inventory(native_path, prefix, verify_files=audit)
    check_requirements(contract, {p["name"]: p["version"] for p in inventory["pip"]})
    tools = profile_runtime_files(prefix)
    if audit:
        if not inventory["conda"]:
            raise ValueError("A clean Conda environment with package metadata is required")
        audit_prefix(prefix)
        audit_managed_files(prefix, inventory["native"])
    subprocess.run([str(prefix / "bin/python"), "-m", "pip", "check"], check=True,
                   stdout=sys.stderr)
    return {"requirements": contract, "requirements_sha256": requirements_sha256(contract),
            "python": platform.python_version(),
            "platform": {"system": platform.system(), "machine": platform.machine(),
                         "glibc_minimum": platform.libc_ver()[1]},
            "packages": inventory, "profile_runtime_files": tools}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--prefix", type=Path, default=Path(sys.prefix))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--audit-package", action="store_true")
    parser.add_argument("--native-inventory", type=Path,
                        help="Reviewed native third-party source and pre-relocation file hashes")
    parser.add_argument("--no-binding", action="store_true", help="Recipe preparation only")
    parser.add_argument("--print-requirements", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        source = args.source.resolve(strict=True)
        if args.print_requirements:
            print("\n".join(selected_requirements(requirement_contract(source))))
            return 0
        native_path = args.native_inventory
        default_native = args.prefix.parent / "locks/native-packages.json"
        if native_path is None and default_native.is_file():
            native_path = default_native
        result = inspect_environment(source, args.prefix, audit=args.audit_package,
                                     native_path=native_path)
        manifest_path = args.manifest or args.prefix.parent / "environment-manifest.json"
        if not args.no_binding:
            if not manifest_path.is_file():
                raise ValueError("Dependency manifest missing; use --no-binding for a manually created environment")
            binding_path = source / "release/environment.json"
            binding = json.loads(binding_path.read_text()) if binding_path.is_file() else None
            check_binding(result["requirements"], json.loads(manifest_path.read_text()),
                          binding, manifest_path)
        print(json.dumps(result, indent=2) if args.json else
              "Dependency requirements and environment binding passed; RhinoForge can now be built.")
        return 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Dependency check failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
