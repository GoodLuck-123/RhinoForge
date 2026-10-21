"""Dependency-only delivery boundaries and source/environment compatibility."""
import importlib.util
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import tarfile

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


deps = load("dependency_environment")
installer = load("install_dependency_env")
packer = load("pack_dependency_env")


def contract():
    return {"profile": "full", "extras": deps.EXTRAS, "requires_python": ">=3.12,<3.13",
            "runtime": ["torch==2.10.0", 'windows-only; sys_platform == "win32"'],
            "optional": {"audio": ["soundfile==0.13.1"], "dev": [], "vision": [], "vla": []},
            "build": ["scikit-build-core>=0.12,<0.13"], "tools": [],
            "constraints": ["torch<2.11", "unused-package>=9"]}


def installed():
    return {"torch": "2.10.0+cpu", "soundfile": "0.13.1", "scikit_build_core": "0.12.2"}


def test_checks_uninstalled_rf_requirements_including_extras_build_and_markers():
    deps.check_requirements(contract(), installed(), python_version="3.12.15",
                            marker_environment={"sys_platform": "linux"})
    for missing in ("soundfile", "scikit_build_core", "torch"):
        packages = installed()
        del packages[missing]
        with pytest.raises(ValueError, match="missing"):
            deps.check_requirements(contract(), packages, python_version="3.12.15")
    with pytest.raises(ValueError, match="Python 3.13"):
        deps.check_requirements(contract(), installed(), python_version="3.13.0")
    packages = installed() | {"torch": "2.11.0"}
    with pytest.raises(ValueError, match="torch"):
        deps.check_requirements(contract(), packages, python_version="3.12.15")


def write_recipe(root, version="1.0", extra="soundfile==0.13.1"):
    root.mkdir(exist_ok=True)
    (root / "constraints").mkdir(exist_ok=True)
    (root / "constraints/security.txt").write_text("# reviewed constraint\ntorch<2.11\n")
    (root / "pyproject.toml").write_text(f'''
[build-system]
requires = ["scikit_build_core<0.13,>=0.12"]
[project]
name = "rhinoforge"
version = "{version}"
requires-python = ">=3.12,<3.13"
dependencies = ["torch==2.10.0"]
[project.optional-dependencies]
audio = ["{extra}"]
vision = []
vla = []
dev = []
''')


def test_recipe_hash_independent_of_rf_version_but_binds_extras_and_constraints(tmp_path):
    write_recipe(tmp_path)
    initial = deps.requirements_sha256(deps.requirement_contract(tmp_path))
    write_recipe(tmp_path, version="2.0")
    assert deps.requirements_sha256(deps.requirement_contract(tmp_path)) == initial
    write_recipe(tmp_path, extra="soundfile==0.14.0")
    assert deps.requirements_sha256(deps.requirement_contract(tmp_path)) != initial
    write_recipe(tmp_path)
    (tmp_path / "constraints/security.txt").write_text("torch<2.12\n")
    assert deps.requirements_sha256(deps.requirement_contract(tmp_path)) != initial
    assert deps.normalized_requirement("scikit_build_core<0.13,>=0.12") == "scikit-build-core<0.13,>=0.12"


def manifest():
    digest = hashlib.sha256(b"{}").hexdigest()
    return {"schema": deps.SCHEMA, "environment_id": deps.ENVIRONMENT_ID,
            "profile": "full", "extras": deps.EXTRAS,
            "requirements_sha256": deps.requirements_sha256(contract()),
            "lock_sha256": digest,
            "contents": {name: False for name in (
                "rhinoforge", "sdk", "launch", "operators", "model_weights")},
            "redistribution_sha256": {name: digest for name in (
                "locks/environment-lock.json", "locks/requirements.json",
                "licenses/LICENSE", "sources/SOURCE")},
            "platform": {"system": platform.system(), "machine": platform.machine()}}


def test_source_binding_rejects_changed_recipe_identity_and_manifest(tmp_path):
    value = manifest()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value))
    binding = {key: value[key] for key in ("schema", "environment_id", "profile", "requirements_sha256")}
    binding["manifest_sha256"] = deps.sha256(path)
    deps.check_binding(contract(), value, binding, path)
    with pytest.raises(ValueError, match="environment_id"):
        deps.check_binding(contract(), value, binding | {"environment_id": "deps-2026.10.2"}, path)
    changed = contract() | {"build": ["scikit-build-core>=0.13"]}
    with pytest.raises(ValueError, match="requirements differ"):
        deps.check_binding(changed, value)
    path.write_text("changed")
    with pytest.raises(ValueError, match="manifest SHA256"):
        deps.check_binding(contract(), value, binding, path)


@pytest.mark.parametrize("relative", [
    "lib/python3.12/site-packages/rpu_backend/__init__.py",
    "lib/python3.12/site-packages/rhinoforge-1.0.dist-info/METADATA",
    "lib/libHxMem.so.1", "lib/libHxil.so", "lib/librpu_api.so", "lib/librhino_launch.so",
    "share/rhinoOpLib_current.ref", "models/model.safetensors", ".ssh/id_ed25519",
    "lib/python3.12/site-packages/__editable__.pkg.pth",
])
def test_audit_rejects_application_sdk_assets_models_and_private_state(tmp_path, relative):
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text("excluded")
    with pytest.raises(ValueError, match="Excluded"):
        deps.audit_prefix(tmp_path)


def test_audit_rejects_external_links_and_path_hooks_but_allows_internal_hooks(tmp_path):
    env = tmp_path / "env"
    site = env / "lib/site-packages"
    site.mkdir(parents=True)
    (env / "external").symlink_to(tmp_path / "private")
    with pytest.raises(ValueError, match="External environment symlink"):
        deps.audit_prefix(env)
    (env / "external").unlink()
    hook = site / "example.pth"
    hook.write_text("/tmp/local-development-source\n")
    with pytest.raises(ValueError, match="External Python path hook"):
        deps.audit_prefix(env)
    hook.write_text("internal-package\nimport _distutils_hack\n")
    deps.audit_prefix(env)


def test_managed_file_audit_rejects_unlisted_payload_and_preserves_conda_metadata(tmp_path, monkeypatch):
    (tmp_path / "conda-meta").mkdir()
    (tmp_path / "conda-meta/python.json").write_text(json.dumps({"files": ["bin/python"]}))
    (tmp_path / "conda-meta/history").write_text("prefix history required by Conda")
    (tmp_path / "conda-meta/created_at").write_text("1792575245.670647")
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin/python").write_text("python")
    monkeypatch.setattr(deps.metadata, "distributions", lambda: [])
    deps.audit_managed_files(tmp_path)
    (tmp_path / "innocent-name.dat").write_text("private model or application data")
    with pytest.raises(ValueError, match="Unmanaged file"):
        deps.audit_managed_files(tmp_path)


@pytest.mark.parametrize("content", ["", "private payload", "1\n2", "1" * 40])
def test_conda_creation_metadata_must_be_a_bounded_timestamp(tmp_path, monkeypatch, content):
    (tmp_path / "conda-meta").mkdir()
    (tmp_path / "conda-meta/created_at").write_text(content)
    monkeypatch.setattr(deps.metadata, "distributions", lambda: [])
    with pytest.raises(ValueError, match="Invalid Conda creation timestamp"):
        deps.audit_managed_files(tmp_path)


def test_managed_file_audit_rejects_escaping_package_record(tmp_path, monkeypatch):
    (tmp_path / "conda-meta").mkdir()
    (tmp_path / "conda-meta/python.json").write_text(json.dumps({"files": ["../outside"]}))
    monkeypatch.setattr(deps.metadata, "distributions", lambda: [])
    with pytest.raises(ValueError, match="record escapes"):
        deps.audit_managed_files(tmp_path)


def test_reviewed_native_inventory_admits_only_matching_public_third_party_files(tmp_path, monkeypatch):
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    (env / "bin/sox").write_text("compiled public SoX")
    value = {"schema": "rhinoforge-native-dependencies/v1", "packages": [{
        "name": "sox", "version": "14.4.2",
        "source_url": "https://downloads.sourceforge.net/sox/sox-14.4.2.tar.gz",
        "source_sha256": "1" * 64, "files": {"bin/sox": deps.sha256(env / "bin/sox")}}]}
    path = tmp_path / "native.json"
    path.write_text(json.dumps(value))
    native = deps.native_inventory(path, env, verify_files=True)
    monkeypatch.setattr(deps.metadata, "distributions", lambda: [])
    deps.audit_managed_files(env, native)
    (env / "bin/sox").write_text("modified after review")
    with pytest.raises(ValueError, match="Native dependency file SHA256 mismatch"):
        deps.native_inventory(path, env, verify_files=True)
    # Relocation changes embedded prefixes; published hashes describe the
    # original reviewed build and are not compared with relocated text files.
    assert deps.native_inventory(path, env, verify_files=False)
    value["packages"][0]["source_url"] = "http://10.0.0.1/private-source.tar.gz"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Public native dependency source"):
        deps.native_inventory(path, env, verify_files=False)


@pytest.mark.parametrize("url", ["http://10.0.0.1/file", "https://name:secret@example.org/file",
                                 "file:///private/file.whl", "https://localhost/a",
                                 "https://example.org/a?token=secret"])
def test_private_provenance_urls_cannot_enter_manifest(url):
    assert not deps.public_url(url)
    assert deps.public_url("https://repo.anaconda.com/pkgs/main/linux-aarch64/python.conda")


def write_archive(path, extra=(), value=None):
    value = value or manifest()
    entries = [("environment-manifest.json", json.dumps(value), "file"),
               ("env/bin/python", "python3", "link"),
               ("env/bin/python3", "binary", "file"),
               ("env/bin/conda-unpack", "unpack", "file"),
               ("activate.sh", "activation", "file")]
    extra_names = {name for name, _, _ in extra}
    entries += [(name, "{}", "file") for name in value.get("redistribution_sha256", {})
                if name not in extra_names]
    entries += list(extra)
    with tarfile.open(path, "w:gz") as archive:
        for name, data, kind in entries:
            member = tarfile.TarInfo(name)
            if kind == "link":
                member.type, member.linkname = tarfile.SYMTYPE, data
                archive.addfile(member)
            else:
                data = data.encode()
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))


@pytest.mark.parametrize("extra", [
    [("runtime/sdk/libHxMem.so", "sdk", "file")],
    [("source/pyproject.toml", "application", "file")],
    [("env/lib/librpu_api.so", "sdk", "file")],
    [("env/lib/site-packages/rpu_backend/__init__.py", "app", "file")],
    [("env/outside", "../../private", "link")],
    [("../private", "bad", "file")],
])
def test_installer_rejects_wrong_payload_before_writing(tmp_path, extra):
    archive = tmp_path / "env.tar.gz"
    write_archive(archive, extra)
    destination = tmp_path / "installation"
    with pytest.raises(ValueError):
        installer.install(archive, installer.sha256(archive), destination)
    assert not destination.exists()


def test_installer_rejects_old_full_bundle_schema(tmp_path):
    archive = tmp_path / "old.tar.gz"
    write_archive(archive, value=manifest() | {"schema": "rhinoforge-conda-bundle/v1"})
    with pytest.raises(ValueError, match="schema"):
        installer.install(archive, installer.sha256(archive), tmp_path / "new")


@pytest.mark.parametrize("change", [
    {"contents": {"sdk": True}}, {"extras": ["audio"]},
    {"redistribution_sha256": {}}, {"lock_sha256": "0" * 64},
])
def test_installer_rejects_incomplete_or_wrong_manifest(tmp_path, change):
    archive = tmp_path / "bad.tar.gz"
    write_archive(archive, value=manifest() | change)
    with pytest.raises(ValueError):
        installer.install(archive, installer.sha256(archive), tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_installer_relocates_and_rolls_back_only_new_directory(tmp_path, monkeypatch):
    archive = tmp_path / "env.tar.gz"
    write_archive(archive)
    calls = []
    monkeypatch.setattr(installer.subprocess, "run", lambda argv, **kw: calls.append(argv))
    destination = tmp_path / "path with spaces"
    installer.install(archive, installer.sha256(archive), destination)
    assert calls[0] == [str(destination / "env/bin/python"), str(destination / "env/bin/conda-unpack")]
    assert json.loads((destination / "installation.json").read_text())["environment_id"] == deps.ENVIRONMENT_ID
    with pytest.raises(FileExistsError):
        installer.install(archive, installer.sha256(archive), destination)
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "conda-unpack")
    monkeypatch.setattr(installer.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        installer.install(archive, installer.sha256(archive), tmp_path / "failed")
    assert destination.is_dir()
    assert not (tmp_path / "failed").exists()
    with pytest.raises(ValueError, match="SHA256"):
        installer.install(archive, "0" * 64, tmp_path / "bad-digest")
    assert not (tmp_path / "bad-digest").exists()


def test_installer_verifies_redistribution_materials_before_execution(tmp_path):
    archive = tmp_path / "env.tar.gz"
    value = manifest()
    value["redistribution_sha256"]["licenses/LICENSE"] = "0" * 64
    write_archive(archive, [("licenses/LICENSE", "different", "file")], value)
    with pytest.raises(ValueError, match="Redistribution material SHA256"):
        installer.install(archive, installer.sha256(archive), tmp_path / "new")
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("status", [0, 7])
def test_activation_preserves_provider_paths_and_shell_modes(tmp_path, status):
    bundle = tmp_path / "dependency directory"
    (bundle / "env/bin").mkdir(parents=True)
    (bundle / "env/bin/activate").write_text(
        'export PATH="$RHINOFORGE_ENV/bin:$PATH"\n' + f"return {status}\n")
    (bundle / "activate.sh").write_text(packer.ACTIVATE.replace("__ENVIRONMENT_ID__", deps.ENVIRONMENT_ID))
    result = subprocess.run(["bash", "--noprofile", "--norc", "-c",
        'set -euo pipefail; before=$-; if . "$1"; then status=0; else status=$?; fi; '
        'printf "%s\\n" "$status" "$before" "$-" "$LD_LIBRARY_PATH" "$RHINO_LAUNCH_ROOT" '
        '"$RHINOFORGE_SOURCE" "${RHINOFORGE_BUILD-unset}" "$RPU_KERNEL_LIB_PATH" '
        '"${RPU_BACKEND_SO-unset}"',
        "bash", str(bundle / "activate.sh")],
        env={"PATH": "/usr/bin:/bin", "LD_LIBRARY_PATH": "/provider/sdk/lib:/provider/launch/lib",
             "RHINO_LAUNCH_ROOT": "/provider/launch", "RHINOFORGE_SOURCE": "/github/source",
             "RPU_KERNEL_LIB_PATH": "/provider/operators/main.ref",
             "RPU_BACKEND_SO": "/old-environment/backend.so"},
        text=True, capture_output=True, check=True)
    observed, before, after, libraries, launch, source, build, operators, backend = result.stdout.splitlines()
    assert observed == str(status)
    assert before == after and "u" in after
    assert backend == "unset"
    expected = "/provider/sdk/lib:/provider/launch/lib"
    assert libraries == (f"{bundle}/env/lib:{expected}" if status == 0 else expected)
    assert (launch, source, build, operators) == (
        "/provider/launch", "/github/source", "unset", "/provider/operators/main.ref")


def test_redistribution_materials_require_all_groups_and_disallow_overrides(tmp_path):
    for group in ("licenses", "sources", "locks"):
        (tmp_path / group).mkdir()
        (tmp_path / group / "public.txt").write_text("public material")
    assert len(packer.redistribution_files(tmp_path)) == 3
    (tmp_path / "activate.sh").write_text("override")
    with pytest.raises(ValueError, match="Reserved"):
        packer.redistribution_files(tmp_path)
    (tmp_path / "activate.sh").unlink()
    (tmp_path / "sources/public.txt").unlink()
    with pytest.raises(ValueError, match="sources/"):
        packer.redistribution_files(tmp_path)
