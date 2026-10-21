#!/usr/bin/env python3
"""Pack an installed RhinoForge conda environment and matching external runtime.

Run this with the builder's Python containing conda-pack and scikit-build-core's
pathspec dependency (also tomli on Python 3.10). The target environment must
contain a regular wheel installation, never an editable installation. External
runtime inputs stay outside the source repository.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile

from install_conda_bundle import sha256, validated_members


ACTIVATE = r'''# Source this file in bash; it also works from a packed environment without conda.
export RHINOFORGE_BUNDLE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export RHINOFORGE_ENV="$RHINOFORGE_BUNDLE/env"
export RHINOFORGE_SOURCE="$RHINOFORGE_BUNDLE/source"
export RHINOFORGE_WORKDIR="$RHINOFORGE_BUNDLE"
export RHINOFORGE_BUILD="$RHINOFORGE_BUNDLE/build"
unset PYTHONPATH PYTHONHOME RPU_BACKEND_SO
while IFS= read -r _rf_key; do
  case "$_rf_key" in
    RPU_MODEL_CACHE|RPU_DEVICE_LOCK) ;;
    RPU_*|LKN_*) unset "$_rf_key" ;;
  esac
done < <(compgen -v)
unset _rf_key
export PYTHONNOUSERSITE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
# conda-pack's activation script reads unset variables such as CONDA_PREFIX.
# Relax nounset only while sourcing it, then restore the caller's shell mode.
_rf_had_nounset=0
case $- in *u*) _rf_had_nounset=1; set +u ;; esac
if . "$RHINOFORGE_ENV/bin/activate"; then
  _rf_activate_status=0
else
  _rf_activate_status=$?
fi
if [ "$_rf_had_nounset" = 1 ]; then set -u; else set +u; fi
unset _rf_had_nounset
if [ "$_rf_activate_status" != 0 ]; then return "$_rf_activate_status"; fi
unset _rf_activate_status
export RHINO_LAUNCH_ROOT="$RHINOFORGE_BUNDLE/runtime/launch"
export RPU_KERNEL_LIB_PATH="$RHINOFORGE_BUNDLE/runtime/operators/rhinoOpLib_current.ref"
export RPU_KERNEL_LIB_PATH_EXPANSION="$RHINOFORGE_BUNDLE/runtime/operators/rhinoExpansionOpLib_current.ref"
export CMAKE_PREFIX_PATH="$RHINO_LAUNCH_ROOT"
export LD_LIBRARY_PATH="$RHINO_LAUNCH_ROOT/lib:$RHINOFORGE_BUNDLE/runtime/sdk:$RHINOFORGE_ENV/lib"
# Set model/results paths after activation. Additional platform SDK paths, if
# required by the provider, must be added explicitly to LD_LIBRARY_PATH.
'''

RUN_EXAMPLE = r'''#!/usr/bin/env bash
set -euo pipefail
_rf_bundle="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
. "$_rf_bundle/activate.sh"
cd "$RHINOFORGE_SOURCE"
exec "$RHINOFORGE_ENV/bin/python" -m examples.run "$@"
'''

# Metadata imports only: this probe never imports torch or rpu_backend.
ENV_PROBE = r'''
import importlib.metadata as m
import json, pathlib, platform, sys
from packaging.requirements import Requirement
dist = m.distribution("rhinoforge")
extras = json.loads(sys.argv[1])
missing = []
for value in dist.requires or []:
    req = Requirement(value)
    if req.marker and not any(req.marker.evaluate({"extra": x}) for x in [""] + extras):
        continue
    try:
        found = m.version(req.name)
    except m.PackageNotFoundError:
        missing.append(str(req)); continue
    if req.specifier and found not in req.specifier:
        missing.append(f"{req}: installed {found}")
if missing:
    raise SystemExit("Unsatisfied profile dependencies: " + "; ".join(missing))
native = pathlib.Path(dist.locate_file("rpu_backend/rpu_backend.so"))
if not native.is_file():
    raise SystemExit("Regular wheel installation with rpu_backend/rpu_backend.so required")
if sys.version_info[:2] != (3, 12):
    raise SystemExit("RhinoForge requires Python 3.12")
print(json.dumps({
    "python": platform.python_version(), "rhinoforge_version": dist.version,
    "platform": {"system": platform.system(), "machine": platform.machine(),
                 "glibc_minimum": platform.libc_ver()[1]},
    "packages": sorted([{"name": d.metadata["Name"], "version": d.version}
                       for d in m.distributions()], key=lambda x: x["name"].lower()),
    "native_sha256": __import__("hashlib").sha256(native.read_bytes()).hexdigest(),
    "package_directory": str(native.parent),
}))
'''


def git(source: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(source), *args])


def add_bytes(archive: tarfile.TarFile, name: str, content: bytes, mode: int = 0o644) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(content)
    info.mode = mode
    archive.addfile(info, io.BytesIO(content))


def clean_header(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mode &= 0o777
    return info


def source_files(source: Path) -> list[Path]:
    paths = [Path(os.fsdecode(x)) for x in git(source, "ls-files", "-z").split(b"\0") if x]
    return sorted(path for path in paths if (source / path).is_file())


def verify_installed_python(source: Path, installed: Path, tracked: list[Path]) -> None:
    """Match the source mapping and final wheel filters used by scikit-build-core."""
    from pathspec import GitIgnoreSpec
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10 builder with scikit-build-core.
        import tomli as tomllib
    with (source / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)
    patterns = project["tool"]["scikit-build"]["wheel"].get("exclude", [])
    exclude = GitIgnoreSpec.from_lines(patterns)
    for relative in tracked:
        if relative.parts[:2] != ("python", "rpu_backend") or relative.suffix != ".py":
            continue
        package_path = Path(*relative.parts[2:])
        wheel_path = Path(*relative.parts[1:])
        # 0.12.2's package mapping filters relative to the package directory;
        # its wheel writer applies the same patterns to the final wheel path.
        if exclude.match_file(package_path.as_posix()) or exclude.match_file(wheel_path.as_posix()):
            continue
        target = installed / package_path
        if not target.is_file() or sha256(target) != sha256(source / relative):
            raise ValueError(f"Installed wheel differs from source: {relative}")


def verify_ref(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise ValueError(f"Operator asset does not match source dependency lock: {path.name}")
    sidecar = Path(str(path) + ".kernels")
    lines = sidecar.read_text(encoding="ascii").splitlines()
    if len(lines) < 3 or lines[:2] != ["rhinoforge-kernels-v1", f"asset-size={path.stat().st_size}"]:
        raise ValueError(f"Invalid operator sidecar: {sidecar.name}")
    if lines[2:] != sorted(set(lines[2:])) or not all(lines[2:]):
        raise ValueError(f"Invalid operator kernel list: {sidecar.name}")


def verify_launch(launch: Path, notices: Path, expected: dict) -> dict:
    """Bind the copied SDK to the source lock and provider's release identity."""
    config = (launch / "lib/cmake/rhino_launch/rhino_launchConfig.cmake").read_text()
    version = (launch / "include/version.h").read_text()
    features = (launch / "include/rhino_launch_features.h").read_text()
    release = {}
    for line in (notices / "RELEASE.txt").read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator:
            release[key] = value
    for key in ("commit", "version", "cmake_version", "abi_version"):
        if release.get("launch_" + key) != str(expected[key]):
            raise ValueError(f"Provider RELEASE.txt Launch {key} differs from source lock")
    for key in ("release_version", "cmake_version", "abi_version"):
        pattern = r'set\s*\(\s*RHINO_LAUNCH_' + key.upper() + r'\s+"?([^"\s)]+)"?\s*\)'
        match = re.search(pattern, config)
        if not match or match[1] != str(expected[key]):
            raise ValueError(f"Installed Launch {key} differs from source lock")
    commit = re.search(r'GIT_COMMIT_HASH\s*=\s*"([0-9a-f]{7,40})"', version)
    if not commit or not expected["commit"].startswith(commit[1]):
        raise ValueError("Installed Launch commit differs from source lock")
    if not re.search(r'BUILD_TYPE\s*=\s*"Release"', version):
        raise ValueError("Installed Launch must be a Release build")
    if not re.search(r'^\s*#define\s+RHINO_LAUNCH_HAS_DEV_PROGRAM_API\s+0\s*$',
                     features, re.MULTILINE):
        raise ValueError("Installed Launch must expose the sanitized API")
    if not (notices / "DISTRIBUTION_TERMS.txt").is_file():
        raise ValueError("Provider DISTRIBUTION_TERMS.txt is required")
    libraries = sorted((launch / "lib").glob("librhino_launch.so*"))
    if not libraries:
        raise ValueError("Installed Launch shared library is missing")
    return {path.relative_to(launch).as_posix(): sha256(path)
            for path in libraries if path.is_file()}


def publish_archive(packed: Path, output: Path, digest: str) -> None:
    """Publish complete files atomically without overwriting existing paths."""
    checksum_path = Path(str(output) + ".sha256")
    temporary_checksum = packed.with_suffix(".sha256")
    temporary_checksum.write_text(f"{digest}  {output.name}\n")
    owned: list[Path] = []
    try:
        # Staging is on the destination filesystem, so hard links provide an
        # atomic exclusive publish without a partial multi-gigabyte copy.
        os.link(packed, output)
        owned.append(output)
        os.link(temporary_checksum, checksum_path)
        owned.append(checksum_path)
    except BaseException:
        for path in reversed(owned):
            path.unlink()
        raise


def profile_runtime_files(prefix: Path, profile: str) -> list[str]:
    """The audio Python wrappers also need the packaged CLI tools and library."""
    if profile != "full":
        return []
    files = [prefix / "bin/sox", prefix / "bin/ffmpeg"]
    missing = [path.relative_to(prefix).as_posix() for path in files
               if not path.is_file() or not os.access(path, os.X_OK)]
    libraries = sorted(path for path in (prefix / "lib").glob("libsndfile.so*")
                       if path.is_file())
    if not libraries:
        missing.append("lib/libsndfile.so*")
    if missing:
        raise ValueError("Full profile requires packaged SoX, FFmpeg and libsndfile; missing or "
                         "not executable: " + ", ".join(missing))
    return [path.relative_to(prefix).as_posix() for path in files + libraries]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("env", "source", "launch", "main-ref", "expansion-ref", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--sdk-version", required=True, help="Provider SDK version tested for this build")
    parser.add_argument("--profile", choices=("text", "full"), default="full")
    parser.add_argument("--sdk-libs-dir", type=Path,
                        help="Optional provider-approved redistributable user-space SDK libraries")
    parser.add_argument("--runtime-notices", "--runtime-notices-dir", required=True, type=Path,
                        help="Provider RELEASE/distribution terms directory, copied without modification")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    prefix = args.env.resolve(strict=True)
    output = args.output.absolute()
    checksum_path = Path(str(output) + ".sha256")
    if output.exists() or checksum_path.exists():
        parser.error("Output archive or checksum already exists")
    if not (prefix / "conda-meta").is_dir():
        parser.error("--env must be a conda environment")
    profile_files = profile_runtime_files(prefix, args.profile)
    if not output.parent.is_dir():
        parser.error("--output parent directory must already exist")
    launch = args.launch.resolve(strict=True)
    if not (launch / "lib/cmake/rhino_launch/rhino_launchConfig.cmake").is_file():
        parser.error("--launch must be an installed sanitized Launch prefix")
    notices = args.runtime_notices.resolve(strict=True)
    if not notices.is_dir() or not any(notices.iterdir()):
        parser.error("--runtime-notices must be a nonempty provider notices directory")
    lock = json.loads((source / "release/runtime-dependencies.json").read_text())
    launch_hashes = verify_launch(launch, notices, lock["launch_kernel"])
    verify_ref(args.main_ref, lock["artifacts"]["rhino_ops_main"]["sha256"])
    verify_ref(args.expansion_ref, lock["artifacts"]["rhino_ops_expansion"]["sha256"])
    sdk_hashes = ({path.relative_to(args.sdk_libs_dir).as_posix(): sha256(path)
                   for path in sorted(args.sdk_libs_dir.rglob("*")) if path.is_file()}
                  if args.sdk_libs_dir else {})
    extras = ["vision", "vla", "audio"] if args.profile == "full" else []
    child_env = dict(os.environ, PYTHONNOUSERSITE="1", TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    for key in ("PYTHONPATH", "PYTHONHOME", "RPU_BACKEND_SO"):
        child_env.pop(key, None)
    subprocess.run([str(prefix / "bin/python"), "-m", "pip", "check"], env=child_env, check=True)
    metadata = json.loads(subprocess.check_output(
        [str(prefix / "bin/python"), "-c", ENV_PROBE, json.dumps(extras)], env=child_env))
    installed = Path(metadata.pop("package_directory"))
    tracked = source_files(source)
    verify_installed_python(source, installed, tracked)
    snapshot_digest = hashlib.sha256()
    for relative in tracked:
        path = source / relative
        if path.is_symlink():
            raise ValueError(f"Source snapshot symlinks are not supported: {relative}")
        digest = sha256(path)
        snapshot_digest.update(relative.as_posix().encode() + b"\0" + digest.encode() + b"\n")
    manifest = {
        "schema": "rhinoforge-conda-bundle/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "extras": extras,
        "profile_runtime_files": profile_files,
        "source_commit": git(source, "rev-parse", "HEAD").decode().strip(),
        "source_dirty": bool(git(source, "status", "--porcelain", "--untracked-files=no")),
        "source_diff_sha256": hashlib.sha256(git(source, "diff", "HEAD", "--binary")).hexdigest(),
        "source_snapshot_sha256": snapshot_digest.hexdigest(),
        "runtime_dependencies": lock,
        "launch_libraries_sha256": launch_hashes,
        "sdk_version": args.sdk_version,
        "sdk_libraries_sha256": sdk_hashes,
        "system_prerequisites": (["compatible Linux kernel and RPU driver", "device access and platform lock"]
                                 + ([] if sdk_hashes else ["matching provider SDK user-space libraries"])),
        "model_weights_included": False,
        "hardware_validation": "not_run_by_packaging",
        **metadata,
    }
    import conda_pack
    with tempfile.TemporaryDirectory(prefix="rhinoforge-pack-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        env_archive = temporary / "env.tar"
        # Remove installation-local wheel URLs and bytecode. conda-pack still
        # retains its own required prefix relocation data until conda-unpack.
        conda_pack.pack(prefix=str(prefix), output=str(env_archive), format="tar", arcroot="env",
                        filters=[("exclude", "**/direct_url.json"), ("exclude", "**/__pycache__/*")])
        packed = temporary / "bundle.tar.gz"
        with tarfile.open(packed, "w:gz", compresslevel=4, dereference=False) as archive:
            with tarfile.open(env_archive, "r:") as environment:
                for member in environment:
                    stream = environment.extractfile(member) if member.isfile() else None
                    archive.addfile(clean_header(member), stream)
                    if stream is not None:
                        stream.close()
            for relative in tracked:
                archive.add(source / relative, arcname="source/" + relative.as_posix(),
                            recursive=False, filter=clean_header)
            archive.add(launch, arcname="runtime/launch", filter=clean_header)
            archive.add(notices, arcname="runtime/notices", filter=clean_header)
            if args.sdk_libs_dir:
                archive.add(args.sdk_libs_dir, arcname="runtime/sdk", filter=clean_header)
            for path, name in ((args.main_ref, "rhinoOpLib_current.ref"),
                               (args.expansion_ref, "rhinoExpansionOpLib_current.ref")):
                archive.add(path, arcname="runtime/operators/" + name, filter=clean_header)
                archive.add(Path(str(path) + ".kernels"),
                            arcname="runtime/operators/" + name + ".kernels", filter=clean_header)
            add_bytes(archive, "activate.sh", ACTIVATE.encode())
            add_bytes(archive, "run-example.sh", RUN_EXAMPLE.encode(), 0o755)
            add_bytes(archive, "bundle-manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
        # Validate the archive with the same rules used on destination machines.
        with tarfile.open(packed, "r:gz") as archive:
            validated_members(archive)
        digest = sha256(packed)
        publish_archive(packed, output, digest)
    print(f"Created {output.name}\nSHA256 {digest}")
    print("Validate relocation and the intended device case before distributing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
