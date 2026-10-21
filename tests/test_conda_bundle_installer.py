"""Archive boundary checks; no conda installation or device is required."""
import importlib.util
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import tarfile

import pytest


spec = importlib.util.spec_from_file_location(
    "install_conda_bundle", Path(__file__).parents[1] / "scripts/install_conda_bundle.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)
sys.modules["install_conda_bundle"] = installer
pack_spec = importlib.util.spec_from_file_location(
    "pack_conda_bundle", Path(__file__).parents[1] / "scripts/pack_conda_bundle.py")
packer = importlib.util.module_from_spec(pack_spec)
pack_spec.loader.exec_module(packer)


def write_archive(path, entries):
    with tarfile.open(path, "w:gz") as archive:
        for name, content, kind in entries:
            member = tarfile.TarInfo(name)
            if kind == "file":
                content = content.encode()
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
            else:
                member.type = {"symlink": tarfile.SYMTYPE, "hardlink": tarfile.LNKTYPE,
                               "fifo": tarfile.FIFOTYPE}[kind]
                member.linkname = content
                archive.addfile(member)


@pytest.mark.parametrize("entries", [
    [("../outside", "bad", "file")],
    [("/outside", "bad", "file")],
    [("env/lib", "../../outside", "symlink")],
    [("env/lib", "target", "symlink"), ("env/lib/file", "bad", "file")],
    [("env/x", ".", "symlink"), ("env/link", "x/../../outside", "symlink")],
    [("env/a", "b", "symlink"), ("env/b", "a", "symlink")],
    [("env/file", "../outside", "hardlink")],
    [("env/fifo", "", "fifo")],
    [("env/file", "a", "file"), ("env/file", "b", "file")],
])
def test_rejects_unsafe_archives_before_writing(tmp_path, entries):
    archive = tmp_path / "bad.tar.gz"
    write_archive(archive, entries)
    destination = tmp_path / "install"
    with pytest.raises(ValueError):
        installer.install(archive, installer.sha256(archive), destination)
    assert not destination.exists()


def minimal_entries():
    manifest = {"schema": "rhinoforge-conda-bundle/v1",
                "platform": {"system": platform.system(), "machine": platform.machine()}}
    return [("bundle-manifest.json", json.dumps(manifest), "file"),
            ("env/bin/python", "python3", "symlink"),
            ("env/bin/python3", "binary", "file"),
            ("env/bin/conda-unpack", "unpack", "file"),
            ("activate.sh", "activate", "file")]


def test_install_relocates_at_final_path_and_preserves_internal_link(tmp_path, monkeypatch):
    archive = tmp_path / "bundle.tar.gz"
    write_archive(archive, minimal_entries())
    calls = []
    monkeypatch.setattr(installer.subprocess, "run", lambda argv, **kw: calls.append((argv, kw)))
    destination = tmp_path / "a directory with spaces"
    installer.install(archive, installer.sha256(archive), destination)
    assert (destination / "env/bin/python").read_text() == "binary"
    assert calls[0][0] == [str(destination / "env/bin/python"), str(destination / "env/bin/conda-unpack")]
    assert calls[0][1]["cwd"] == destination
    assert (destination / "installation.json").is_file()


def test_failed_relocation_removes_only_new_installation(tmp_path, monkeypatch):
    archive = tmp_path / "bundle.tar.gz"
    write_archive(archive, minimal_entries())
    sentinel = tmp_path / "other-user"
    sentinel.write_text("keep")
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "conda-unpack")
    monkeypatch.setattr(installer.subprocess, "run", fail)
    destination = tmp_path / "new"
    with pytest.raises(subprocess.CalledProcessError):
        installer.install(archive, installer.sha256(archive), destination)
    assert not destination.exists()
    assert sentinel.read_text() == "keep"


def test_existing_directory_and_bad_digest_are_never_modified(tmp_path):
    archive = tmp_path / "bundle.tar.gz"
    write_archive(archive, minimal_entries())
    destination = tmp_path / "existing"
    destination.mkdir()
    sentinel = destination / "keep"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        installer.install(archive, installer.sha256(archive), destination)
    assert sentinel.read_text() == "keep"
    with pytest.raises(ValueError, match="SHA256"):
        installer.install(archive, "0" * 64, tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_publish_checksum_conflict_removes_only_own_archive(tmp_path):
    staged = tmp_path / "staged.tar.gz"
    staged.write_bytes(b"complete archive")
    output = tmp_path / "release.tar.gz"
    existing_checksum = Path(str(output) + ".sha256")
    existing_checksum.write_text("other build")
    with pytest.raises(FileExistsError):
        packer.publish_archive(staged, output, installer.sha256(staged))
    assert not output.exists()
    assert existing_checksum.read_text() == "other build"
    assert staged.read_bytes() == b"complete archive"


def test_publish_does_not_replace_existing_archive(tmp_path):
    staged = tmp_path / "staged.tar.gz"
    staged.write_bytes(b"new")
    output = tmp_path / "release.tar.gz"
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        packer.publish_archive(staged, output, installer.sha256(staged))
    assert output.read_bytes() == b"existing"


def test_activate_uses_relocated_paths_and_preserves_model_cache(tmp_path):
    bundle = tmp_path / "bundle with spaces"
    (bundle / "env/bin").mkdir(parents=True)
    (bundle / "env/bin/activate").write_text('export PATH="$RHINOFORGE_ENV/bin:$PATH"\n')
    (bundle / "activate.sh").write_text(packer.ACTIVATE)
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c",
         '. "$1"; printf "%s\\n" "$RHINOFORGE_ENV" "$RHINO_LAUNCH_ROOT" '
         '"$RPU_MODEL_CACHE" "${RPU_BACKEND_SO-unset}" "$LD_LIBRARY_PATH" '
         '"$RHINOFORGE_WORKDIR" "$RHINOFORGE_BUILD"',
         "bash", str(bundle / "activate.sh")],
        env={"PATH": "/usr/bin:/bin", "RPU_MODEL_CACHE": "/chosen/model",
             "RPU_BACKEND_SO": "/old/backend.so", "LD_LIBRARY_PATH": "/old/launch"},
        capture_output=True, text=True, check=True)
    assert result.stdout.splitlines() == [
        str(bundle / "env"), str(bundle / "runtime/launch"), "/chosen/model", "unset",
        f"{bundle}/runtime/launch/lib:{bundle}/runtime/sdk:{bundle}/env/lib",
        str(bundle), str(bundle / "build")]


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("activation_status", [0, 7])
def test_activate_restores_nounset_and_propagates_conda_failure(
        tmp_path, strict, activation_status):
    bundle = tmp_path / "relocated bundle"
    (bundle / "env/bin").mkdir(parents=True)
    (bundle / "env/bin/activate").write_text(
        'if [ -n "$CONDA_PREFIX" ]; then return 99; fi\n'
        'export CONDA_PREFIX="$RHINOFORGE_ENV"\n'
        'export PATH="$CONDA_PREFIX/bin:$PATH"\n'
        f"return {activation_status}\n")
    (bundle / "activate.sh").write_text(packer.ACTIVATE)
    command = (
        ("set -euo pipefail; " if strict else "set +u; ")
        + 'before=$-; if . "$1"; then status=0; else status=$?; fi; '
        'printf "%s\\n" "$status" "$before" "$-" "$PATH" '
        '"${RHINO_LAUNCH_ROOT-unset}"')
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", command, "bash", str(bundle / "activate.sh")],
        env={"PATH": "/usr/bin:/bin"}, capture_output=True, text=True, check=True)
    status, before, after, path, launch = result.stdout.splitlines()
    assert status == str(activation_status)
    assert before == after
    assert ("u" in after) is strict
    assert path == f"{bundle}/env/bin:/usr/bin:/bin"
    assert launch == (str(bundle / "runtime/launch") if activation_status == 0 else "unset")


@pytest.mark.parametrize("missing", ["bin/sox", "bin/ffmpeg", "lib/libsndfile.so"])
def test_full_profile_requires_packaged_audio_runtime(tmp_path, missing):
    for relative in ("bin/sox", "bin/ffmpeg", "lib/libsndfile.so"):
        path = tmp_path / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"test fixture")
        path.chmod(0o755)
    (tmp_path / missing).unlink()
    with pytest.raises(ValueError, match="Full profile requires"):
        packer.profile_runtime_files(tmp_path, "full")


def test_full_profile_rejects_nonexecutable_tool_and_records_valid_paths(tmp_path):
    for relative in ("bin/sox", "bin/ffmpeg", "lib/libsndfile.so.1"):
        path = tmp_path / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"test fixture")
        path.chmod(0o755)
    sox = tmp_path / "bin/sox"
    sox.chmod(0o644)
    with pytest.raises(ValueError, match="bin/sox"):
        packer.profile_runtime_files(tmp_path, "full")
    sox.chmod(0o755)
    assert packer.profile_runtime_files(tmp_path, "full") == [
        "bin/sox", "bin/ffmpeg", "lib/libsndfile.so.1"]
    assert packer.profile_runtime_files(tmp_path, "text") == []


@pytest.mark.parametrize("runtime_state", ["matching", "missing", "changed"])
def test_wheel_exclusions_preserve_runtime_source_checks(tmp_path, runtime_state):
    pytest.importorskip("pathspec", reason="wheel filtering uses the optional packaging toolchain")
    source = tmp_path / "source"
    installed = tmp_path / "site-packages/rpu_backend"
    installed.mkdir(parents=True)
    files = {
        "api.py": "runtime = True\n",
        "tests/_seed.py": "test support\n",
        "debug/inspect.py": "debug helper\n",
        "nested/test_runtime.py": "test case\n",
    }
    tracked = []
    for name, text in files.items():
        relative = Path("python/rpu_backend") / name
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        tracked.append(relative)
    (source / "pyproject.toml").write_text(
        (Path(__file__).parents[1] / "pyproject.toml").read_text())
    if runtime_state != "missing":
        (installed / "api.py").write_text(
            files["api.py"] if runtime_state == "matching" else "different runtime\n")
    if runtime_state == "matching":
        packer.verify_installed_python(source, installed, tracked)
    else:
        with pytest.raises(ValueError, match=r"python/rpu_backend/api\.py"):
            packer.verify_installed_python(source, installed, tracked)
