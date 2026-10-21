#!/usr/bin/env python3
"""Pack a clean, public third-party dependency environment; never include RF/SDK.

Run with a builder Python containing conda-pack. The target environment must be
Python 3.12 and satisfy the source recipe without installing RhinoForge itself.
Redistribution materials must already have been reviewed for the actual builds.
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

from dependency_environment import ENVIRONMENT_ID, EXTRAS, SCHEMA, forbidden_path
from install_dependency_env import check_dependency_archive, sha256, validated_members

ACTIVATE = r'''# Source this file in bash. The SDK and runtime assets are configured separately.
export RHINOFORGE_DEPENDENCIES="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export RHINOFORGE_ENV="$RHINOFORGE_DEPENDENCIES/env"
export RHINOFORGE_ENVIRONMENT_ID=__ENVIRONMENT_ID__
unset PYTHONPATH PYTHONHOME RPU_BACKEND_SO
export PYTHONNOUSERSITE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
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
# Preserve the provider's SDK/Launch paths; prepend only this environment's libs.
case ":${LD_LIBRARY_PATH-}:" in
  *":$RHINOFORGE_ENV/lib:"*) ;;
  *) export LD_LIBRARY_PATH="$RHINOFORGE_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac
'''


def add_bytes(archive: tarfile.TarFile, name: str, content: bytes, mode: int = 0o644) -> None:
    info = tarfile.TarInfo(name)
    info.size, info.mode = len(content), mode
    archive.addfile(info, io.BytesIO(content))


def clean_header(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mode &= 0o777
    return info


def publish_archive(packed: Path, output: Path, digest: str) -> None:
    checksum = packed.with_suffix(".sha256")
    checksum.write_text(f"{digest}  {output.name}\n")
    owned = []
    try:
        for original, target in ((packed, output), (checksum, Path(str(output) + ".sha256"))):
            os.link(original, target)
            owned.append(target)
    except BaseException:
        for path in reversed(owned):
            path.unlink()
        raise


def redistribution_files(root: Path) -> list[Path]:
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(f"Redistribution material must be regular files/directories: {relative}")
        if path.is_file():
            if relative.parts[0] in {"env", "runtime", "source", "models"} or relative.as_posix() in {
                    "activate.sh", "environment-manifest.json", "installation.json",
                    "locks/environment-lock.json", "locks/requirements.json"}:
                raise ValueError(f"Reserved redistribution path: {relative}")
            if forbidden_path(relative.as_posix()):
                raise ValueError(f"Excluded redistribution material: {relative}")
            files.append(relative)
    for required in ("licenses", "sources", "locks"):
        if not any(path.parts[0] == required for path in files):
            raise ValueError(f"Redistribution materials require nonempty {required}/")
    return files


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("env", "source", "output", "redistribution"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--environment-id", default=ENVIRONMENT_ID)
    args = parser.parse_args()
    try:
        if not re.fullmatch(r"deps-[0-9]{4}\.[0-9]{2}\.[0-9]+", args.environment_id):
            raise ValueError("Environment ID must be deps-YYYY.MM.N")
        prefix = args.env.resolve(strict=True)
        source = args.source.resolve(strict=True)
        materials = args.redistribution.resolve(strict=True)
        output = args.output.absolute()
        if not output.parent.is_dir():
            raise ValueError("Output parent directory must already exist")
        if output.exists() or Path(str(output) + ".sha256").exists():
            raise FileExistsError("Output archive/checksum already exists")
        if output.is_relative_to(prefix) or materials.is_relative_to(prefix):
            raise ValueError("Output and redistribution materials must be outside the environment")
        files = redistribution_files(materials)
        checker = Path(__file__).with_name("dependency_environment.py").resolve()
        child_env = {"PATH": f"{prefix}/bin:/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp"),
                     "PYTHONNOUSERSITE": "1", "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}
        probe_command = [
            str(prefix / "bin/python"), str(checker), "--source", str(source), "--prefix", str(prefix),
            "--audit-package", "--no-binding", "--json"]
        native_path = materials / "locks/native-packages.json"
        if native_path.is_file():
            probe_command += ["--native-inventory", str(native_path)]
        probe = subprocess.check_output(probe_command, env=child_env)
        metadata = json.loads(probe)
        commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        # A dirty source tree is allowed during release preparation, but the exact
        # normalized recipe hash always binds what was actually used.
        recipe_dirty = bool(subprocess.check_output([
            "git", "-C", str(source), "status", "--porcelain", "--", "pyproject.toml",
            "constraints/security.txt", "scripts/dependency_environment.py"]))
        inventory = json_bytes(metadata["packages"])
        contract = json_bytes(metadata.pop("requirements"))
        generated = {"locks/environment-lock.json": inventory, "locks/requirements.json": contract}
        materials_hashes = {path.as_posix(): sha256(materials / path) for path in files}
        materials_hashes.update({name: hashlib.sha256(data).hexdigest() for name, data in generated.items()})
        manifest = {
            "schema": SCHEMA, "environment_id": args.environment_id, "profile": "full", "extras": EXTRAS,
            "created_at": datetime.now(timezone.utc).isoformat(), "recipe_source_commit": commit,
            "recipe_dirty": recipe_dirty, "lock_sha256": hashlib.sha256(inventory).hexdigest(),
            "native_file_hashes_stage": "before_conda_pack_relocation",
            "redistribution_sha256": materials_hashes,
            "contents": {"rhinoforge": False, "sdk": False, "launch": False,
                         "operators": False, "model_weights": False},
            "hardware_validation": "not_run_by_packaging", **metadata,
        }
        import conda_pack
        with tempfile.TemporaryDirectory(prefix="rhinoforge-deps-pack-", dir=output.parent) as temporary:
            temporary = Path(temporary)
            env_archive = temporary / "env.tar"
            conda_pack.pack(prefix=str(prefix), output=str(env_archive), format="tar", arcroot="env",
                            filters=[("exclude", "**/direct_url.json"),
                                     ("exclude", "**/__pycache__/*")])
            staged = temporary / "dependencies.tar.gz"
            with tarfile.open(staged, "w:gz", compresslevel=4, dereference=False) as archive:
                with tarfile.open(env_archive, "r:") as environment:
                    for member in environment:
                        stream = environment.extractfile(member) if member.isfile() else None
                        archive.addfile(clean_header(member), stream)
                        if stream is not None:
                            stream.close()
                for relative in files:
                    archive.add(materials / relative, arcname=relative.as_posix(),
                                recursive=False, filter=clean_header)
                for name, data in generated.items():
                    add_bytes(archive, name, data)
                activation = ACTIVATE.replace("__ENVIRONMENT_ID__", args.environment_id)
                add_bytes(archive, "activate.sh", activation.encode())
                add_bytes(archive, "environment-manifest.json", json_bytes(manifest))
            with tarfile.open(staged, "r:gz") as archive:
                members = validated_members(archive)
                check_dependency_archive(members, manifest)
            digest = sha256(staged)
            publish_archive(staged, output, digest)
        # External sidecars are intentionally produced from archive contents by
        # the publisher, so this operation only publishes the archive and digest.
        print(f"Created dependency environment {args.environment_id}: {output.name}\nSHA256 {digest}")
        print("Validate relocation, source build and representative hardware cases before publishing.")
        return 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Dependency packaging failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
