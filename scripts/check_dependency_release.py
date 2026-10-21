#!/usr/bin/env python3
"""Validate a separately versioned dependency-environment release."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def check(root: Path, *, tag: str = "", archive: Path | None = None,
          manifest: Path | None = None) -> dict:
    binding = json.loads((root / "release/environment.json").read_text())
    spec = importlib.util.spec_from_file_location(
        "rf_dependency_contract", Path(__file__).with_name("dependency_environment.py"))
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    if binding.get("schema") != contract.SCHEMA or binding.get("profile") != "full":
        raise ValueError("Unsupported dependency release schema/profile")
    identity = binding.get("environment_id", "")
    if not re.fullmatch(r"deps-[0-9]{4}\.[0-9]{2}\.[0-9]+", identity):
        raise ValueError("Invalid dependency environment ID")
    if tag and tag != identity:
        raise ValueError("Dependency tag must match environment_id")
    if sorted(binding.get("extras", [])) != contract.EXTRAS:
        raise ValueError("Dependency release must cover every full-profile extra")
    if binding.get("requirements_sha256") != contract.requirements_sha256(
            contract.requirement_contract(root)):
        raise ValueError("Source dependency requirements differ from release binding")
    for key in ("archive_sha256", "manifest_sha256", "lock_sha256"):
        if not isinstance(binding.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", binding[key]):
            raise ValueError(f"Missing or invalid published digest: {key}")
    platform = binding.get("platform", {})
    if platform.get("system") != "Linux" or platform.get("machine") != "aarch64":
        raise ValueError("Initial dependency release requires Linux/aarch64")
    if not re.fullmatch(r"[0-9]+\.[0-9]+", platform.get("glibc_minimum", "")):
        raise ValueError("Record the validated glibc baseline")
    if not re.fullmatch(r"3\.12\.[0-9]+", binding.get("python", "")):
        raise ValueError("Initial dependency release requires Python 3.12")
    url = urlsplit(binding.get("archive_url", ""))
    filename = f"rhinoforge-{identity}-full-linux-aarch64-py312.tar.gz"
    expected = f"/HUIXI-AI/RhinoForge/releases/download/{identity}/{filename}"
    if (url.scheme != "https" or url.netloc != "github.com" or url.path != expected
            or url.query or url.fragment):
        raise ValueError("Archive URL must identify the immutable GitHub environment release")
    if archive is not None and sha256(archive) != binding["archive_sha256"]:
        raise ValueError("Published archive SHA256 mismatch")
    if manifest is not None:
        if sha256(manifest) != binding["manifest_sha256"]:
            raise ValueError("Published manifest SHA256 mismatch")
        packed = json.loads(manifest.read_text())
        for key in ("schema", "environment_id", "profile", "requirements_sha256",
                    "lock_sha256", "platform", "python"):
            if packed.get(key) != binding[key]:
                raise ValueError(f"Manifest differs from source binding: {key}")
        if sorted(packed.get("extras", [])) != binding["extras"]:
            raise ValueError("Manifest extras differ from source binding")
    return {"environment_id": identity, "profile": "full",
            "requirements_sha256": binding["requirements_sha256"],
            "archive_sha256": binding["archive_sha256"],
            "artifact_bytes_checked": archive is not None,
            "manifest_bytes_checked": manifest is not None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--tag", default="")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(check(args.root, tag=args.tag, archive=args.archive,
                               manifest=args.manifest), sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Dependency release check failed: {error}\n")


if __name__ == "__main__":
    main()
