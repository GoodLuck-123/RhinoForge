#!/usr/bin/env python3
"""Install a verified RhinoForge dependency-only environment into a new, final directory.

Uses only Python's standard library (Python >= 3.10). The SHA256 must come from
the provider's trusted delivery channel. This installer does not acquire a
device or initialize the RPU backend.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
from urllib.parse import urlsplit
from urllib.request import urlopen


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_name(name: str) -> str:
    if not name or "\\" in name or name.startswith("/"):
        raise ValueError(f"Unsafe archive path: {name!r}")
    parts = PurePosixPath(name).parts
    if not parts or ".." in parts:
        raise ValueError(f"Unsafe archive path: {name!r}")
    return str(PurePosixPath(*parts))


def validated_members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    """Reject traversal, link pivots, devices and ambiguous duplicate entries."""
    members: dict[str, tarfile.TarInfo] = {}
    for member in archive.getmembers():
        name = archive_name(member.name)
        if name in members:
            raise ValueError(f"Duplicate archive path: {name}")
        if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
            raise ValueError(f"Unsupported archive entry: {name}")
        members[name] = member
    for name, member in members.items():
        for parent in PurePosixPath(name).parents:
            ancestor = members.get(str(parent))
            if ancestor is not None and not ancestor.isdir():
                raise ValueError(f"Archive parent is not a directory: {parent}")
        if member.issym():
            link = member.linkname
            if not link or link.startswith("/") or "\\" in link:
                raise ValueError(f"Unsafe symlink: {name}")
            # Resolve components in order: lexical normalization alone misses
            # a link to "x/../../outside" when x itself points to ".".
            pending = list(PurePosixPath(name).parent.parts) + link.split("/")
            resolved: list[str] = []
            expansions = 0
            while pending:
                part = pending.pop(0)
                if part in ("", "."):
                    continue
                if part == "..":
                    if not resolved:
                        raise ValueError(f"Symlink escapes installation: {name}")
                    resolved.pop()
                    continue
                resolved.append(part)
                target = members.get("/".join(resolved))
                if target is not None and target.issym():
                    expansions += 1
                    if expansions > 40:
                        raise ValueError(f"Symlink cycle or excessive chain: {name}")
                    if target.linkname.startswith("/") or "\\" in target.linkname:
                        raise ValueError(f"Unsafe symlink chain: {name}")
                    resolved.pop()
                    pending = target.linkname.split("/") + pending
        elif member.islnk():
            target = archive_name(member.linkname)
            if target not in members or not members[target].isfile():
                raise ValueError(f"Hardlink must target a regular archive file: {name}")
    return members


def safe_extract(archive: tarfile.TarFile, destination: Path,
                 members: dict[str, tarfile.TarInfo]) -> None:
    """Write regular files before links; never follow an archive-provided link."""
    for name, member in members.items():
        path = destination / name
        if member.isdir():
            path.mkdir(parents=True, exist_ok=True)
        elif member.isfile():
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Missing archive contents: {name}")
            with stream, path.open("xb") as output:
                shutil.copyfileobj(stream, output, 1024 * 1024)
            path.chmod(member.mode & 0o777)
    for name, member in members.items():
        path = destination / name
        if member.issym() or member.islnk():
            path.parent.mkdir(parents=True, exist_ok=True)
            if member.issym():
                path.symlink_to(member.linkname)
            else:
                os.link(destination / archive_name(member.linkname), path)


def check_platform(manifest: dict) -> None:
    if manifest.get("schema") != "rhinoforge-dependency-env/v1":
        raise ValueError("Unsupported or missing bundle schema")
    expected = manifest["platform"]
    if expected["system"] != platform.system() or expected["machine"] != platform.machine():
        raise ValueError(f"Bundle requires {expected['system']}/{expected['machine']}")
    libc, version = platform.libc_ver()
    minimum = expected.get("glibc_minimum")
    if minimum and (libc != "glibc" or
                    tuple(map(int, version.split("."))) < tuple(map(int, minimum.split(".")))):
        raise ValueError(f"Bundle requires glibc >= {minimum}; found {libc} {version}")



def check_dependency_archive(members: dict[str, tarfile.TarInfo], manifest: dict) -> None:
    if manifest.get("profile") != "full" or not re.fullmatch(
            r"deps-[0-9]{4}\.[0-9]{2}\.[0-9]+", manifest.get("environment_id", "")):
        raise ValueError("Invalid dependency environment identity/profile")
    if sorted(manifest.get("extras", [])) != ["audio", "dev", "vision", "vla"]:
        raise ValueError("Dependency environment is missing full-profile extras")
    if not re.fullmatch(r"[0-9a-f]{64}", manifest.get("requirements_sha256", "")):
        raise ValueError("Invalid source dependency recipe digest")
    if any(manifest.get("contents", {}).get(name) is not False
           for name in ("rhinoforge", "sdk", "launch", "operators", "model_weights")):
        raise ValueError("Dependency manifest must exclude application/runtime payloads")
    materials = manifest.get("redistribution_sha256", {})
    for required in ("locks/environment-lock.json", "locks/requirements.json"):
        if required not in materials:
            raise ValueError(f"Dependency manifest is missing {required}")
    if manifest.get("lock_sha256") != materials["locks/environment-lock.json"]:
        raise ValueError("Dependency lock digest is inconsistent")
    for group in ("licenses/", "sources/"):
        if not any(name.startswith(group) for name in materials):
            raise ValueError(f"Dependency manifest is missing redistribution {group}")
    for name in members:
        parts = [part.lower() for part in PurePosixPath(name).parts]
        base = parts[-1]
        if parts[0] in {"runtime", "source", "models", "checkpoints"}:
            raise ValueError(f"Not a dependency-only archive: {name}")
        if parts[0] != "env":
            continue
        if (any(part in {"rpu_backend", "rhinoforge", "rhino_launch", "launch_kernel",
                         ".git", ".ssh", ".aws", ".huggingface"} for part in parts)
                or any(part.startswith(("rhinoforge-", "rpu_backend-", "__editable__"))
                       for part in parts)
                or base.startswith(("libhxmem.so", "libhxil.so", "librpu_api.so",
                                    "librhino_launch.so", "rpu_backend.", "rhinooplib",
                                    "rhinoexpansionoplib"))
                or base.endswith((".safetensors", ".gguf", ".egg-link"))
                or base in {".netrc", ".pypirc", ".env", "id_rsa", "id_ed25519",
                            ".condarc", "pip.conf", ".npmrc"}):
            raise ValueError(f"Excluded payload in dependency archive: {name}")
    for name, digest in materials.items():
        if archive_name(name) != name or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid redistribution file identity")
        member = members.get(name)
        if member is None or not member.isfile():
            raise ValueError(f"Missing redistribution material: {name}")


def install(archive_path: Path, expected_sha256: str, prefix: Path) -> dict:
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("--sha256 must contain exactly 64 hexadecimal characters")
    if os.path.lexists(prefix):
        raise FileExistsError(f"Installation directory already exists: {prefix}")
    if sha256(archive_path) != expected_sha256.lower():
        raise ValueError("Archive SHA256 mismatch")
    created = False
    try:
        with tarfile.open(archive_path, "r:*") as archive:
            members = validated_members(archive)
            manifest_member = members.get("environment-manifest.json")
            if manifest_member is None or not manifest_member.isfile():
                raise ValueError("Bundle has no regular environment-manifest.json")
            if manifest_member.size > 4 * 1024 * 1024:
                raise ValueError("Bundle manifest is unexpectedly large")
            manifest = json.load(archive.extractfile(manifest_member))
            check_platform(manifest)
            check_dependency_archive(members, manifest)
            for required in ("env/bin/python", "env/bin/conda-unpack", "activate.sh"):
                if required not in members:
                    raise ValueError(f"Bundle is missing {required}")
            # This atomic mkdir also rejects a directory created since the check.
            for name, digest in manifest.get("redistribution_sha256", {}).items():
                stream = archive.extractfile(members[name])
                actual = hashlib.sha256()
                with stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        actual.update(block)
                if actual.hexdigest() != digest:
                    raise ValueError(f"Redistribution material SHA256 mismatch: {name}")
            prefix.mkdir(mode=0o755)
            created = True
            safe_extract(archive, prefix, members)
        child_env = {
            "PATH": f"{prefix}/env/bin:/usr/bin:/bin",
            "HOME": os.environ.get("HOME", str(prefix)),
            "PYTHONNOUSERSITE": "1",
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        }
        subprocess.run([str(prefix / "env/bin/python"),
                        str(prefix / "env/bin/conda-unpack")],
                       env=child_env, cwd=prefix, check=True)
        subprocess.run([str(prefix / "env/bin/python"), "-m", "pip", "check"],
                       env=child_env, cwd=prefix, check=True)
        (prefix / "installation.json").write_text(json.dumps({
            "archive_sha256": expected_sha256.lower(),
            "environment_id": manifest["environment_id"],
            "relocation_complete": True,
            "device_validation": "not_run",
        }, indent=2) + "\n")
        return manifest
    except BaseException:
        if created:
            shutil.rmtree(prefix)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, help="Local archive or HTTP(S) URL")
    parser.add_argument("--sha256", required=True, help="Provider-published archive SHA256")
    parser.add_argument("--prefix", required=True, type=Path,
                        help="New final directory; its parent must already exist")
    args = parser.parse_args()
    prefix = Path(os.path.abspath(args.prefix.expanduser()))
    if not prefix.parent.is_dir():
        parser.error("--prefix parent directory must already exist")
    if os.path.lexists(prefix):
        parser.error("--prefix already exists; select a new directory")
    parsed = urlsplit(args.archive)
    try:
        with tempfile.TemporaryDirectory(prefix=".rhinoforge-download-", dir=prefix.parent) as temporary:
            if parsed.scheme in ("http", "https"):
                if parsed.username or parsed.password:
                    raise ValueError("Use a delivery URL without embedded credentials")
                archive_path = Path(temporary) / "bundle.tar.gz"
                print("Downloading bundle...", flush=True)
                with urlopen(args.archive, timeout=60) as response, archive_path.open("xb") as out:
                    if urlsplit(response.url).scheme not in ("http", "https"):
                        raise ValueError("Download redirected outside HTTP(S)")
                    shutil.copyfileobj(response, out, 1024 * 1024)
            elif parsed.scheme:
                raise ValueError("Only local files and HTTP(S) URLs are supported")
            else:
                archive_path = Path(args.archive).expanduser().resolve(strict=True)
            manifest = install(archive_path, args.sha256, prefix)
        print(f"Installed dependency environment {manifest['environment_id']} ({manifest['profile']}).")
        print("RhinoForge is not installed: obtain its source, check dependencies, then build it.")
        print(f"Activate in bash: source {shlex_quote(str(prefix / 'activate.sh'))}")
        print("Keep this final directory in place; reinstall from the archive to change paths.")
        return 0
    except (OSError, ValueError, KeyError, tarfile.TarError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Installation failed: {error}\n")


def shlex_quote(value: str) -> str:
    import shlex
    return shlex.quote(value)


if __name__ == "__main__":
    raise SystemExit(main())
