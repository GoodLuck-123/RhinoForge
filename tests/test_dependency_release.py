"""Independent environment releases must bind actual source requirements and bytes."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release = load_script("check_dependency_release")
contract = load_script("dependency_environment")


@pytest.fixture
def candidate(tmp_path):
    (tmp_path / "release").mkdir()
    (tmp_path / "constraints").mkdir()
    for name in ("VERSION", "pyproject.toml", "constraints/security.txt",
                 "release/runtime-dependencies.json"):
        shutil.copyfile(ROOT / name, tmp_path / name)
    archive = tmp_path / "candidate.tar.gz"
    archive.write_bytes(b"artifact bytes verified separately by installer tests")
    identity = "deps-2026.10.1"
    binding = {
        "schema": contract.SCHEMA, "environment_id": identity, "profile": "full",
        "extras": contract.EXTRAS,
        "platform": {"system": "Linux", "machine": "aarch64", "glibc_minimum": "2.35"},
        "python": "3.12.15",
        "requirements_sha256": contract.requirements_sha256(contract.requirement_contract(tmp_path)),
        "archive_url": ("https://github.com/HUIXI-AI/RhinoForge/releases/download/" + identity
                        + "/rhinoforge-" + identity + "-full-linux-aarch64-py312.tar.gz"),
        "archive_sha256": release.sha256(archive),
        "lock_sha256": "a" * 64,
    }
    manifest = tmp_path / "environment-manifest.json"
    manifest.write_text(json.dumps(binding))
    binding["manifest_sha256"] = release.sha256(manifest)
    (tmp_path / "release/environment.json").write_text(json.dumps(binding))
    return tmp_path, archive, manifest


def mutate(root, key, value):
    path = root / "release/environment.json"
    data = json.loads(path.read_text())
    data[key] = value
    path.write_text(json.dumps(data))


def test_environment_tag_and_source_version_are_independent(candidate):
    root, archive, manifest = candidate
    result = release.check(root, tag="deps-2026.10.1", archive=archive, manifest=manifest)
    assert result["artifact_bytes_checked"] and result["manifest_bytes_checked"]
    run = subprocess.run([sys.executable, str(ROOT / "scripts/check_release_version.py"),
                          "--root", str(root), "--tag", "deps-2026.10.1"],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["version"] == (ROOT / "VERSION").read_text().strip()
    # A dependency release never makes a development RF source tag acceptable.
    wrong = subprocess.run([sys.executable, str(ROOT / "scripts/check_release_version.py"),
                            "--root", str(root), "--tag", "v1.3.0.dev0"],
                           capture_output=True, text=True)
    assert wrong.returncode != 0


@pytest.mark.parametrize("key,value", [
    ("environment_id", "latest"),
    ("schema", "rhinoforge-conda-bundle/v1"),
    ("profile", "text"),
    ("extras", ["audio"]),
    ("archive_sha256", ""),
    ("manifest_sha256", "not-a-checksum"),
    ("lock_sha256", "b" * 63),
    ("requirements_sha256", "0" * 64),
    ("python", "3.13.1"),
    ("platform", {"system": "Linux", "machine": "x86_64", "glibc_minimum": "2.35"}),
    ("archive_url", "https://github.com/HUIXI-AI/RhinoForge/releases/latest/download/env.tar.gz"),
    ("archive_url", "https://example.com/archive.tar.gz"),
])
def test_rejects_incomplete_or_incompatible_release_binding(candidate, key, value):
    root, _, _ = candidate
    mutate(root, key, value)
    with pytest.raises(ValueError):
        release.check(root)


def test_requirement_changes_need_revalidation_but_rf_version_changes_do_not(candidate):
    root, _, _ = candidate
    (root / "VERSION").write_text("1.4.0.dev0\n")
    release.check(root)
    with (root / "constraints/security.txt").open("a") as stream:
        stream.write("\nnew-dependency>=2\n")
    with pytest.raises(ValueError, match="requirements"):
        release.check(root)


def test_rejects_changed_artifact_bytes_and_wrong_tag(candidate):
    root, archive, manifest = candidate
    with pytest.raises(ValueError, match="tag"):
        release.check(root, tag="deps-2026.10.2")
    archive.write_bytes(b"changed")
    with pytest.raises(ValueError, match="archive SHA256"):
        release.check(root, archive=archive)
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="manifest SHA256"):
        release.check(root, manifest=manifest)
