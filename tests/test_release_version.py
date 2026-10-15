"""Distribution and pinned Launch versions are independent and internally consistent."""
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_version", ROOT / "scripts/check_release_version.py")
version_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(version_check)


@pytest.fixture
def source(tmp_path):
    (tmp_path / "release").mkdir()
    for name in ("VERSION", "pyproject.toml", "release/runtime-dependencies.json", "release/version.cmake"):
        shutil.copyfile(ROOT / name, tmp_path / name)
    return tmp_path


def set_version(source, version):
    (source / "VERSION").write_text(version + "\n")
    pyproject = source / "pyproject.toml"
    classifier = "3 - Alpha" if ".dev" in version else "4 - Beta" if "rc" in version else "5 - Production/Stable"
    pyproject.write_text(re.sub(r'Development Status :: [^\"]+', "Development Status :: " + classifier, pyproject.read_text()))


def set_launch(source, **fields):
    path = source / "release/runtime-dependencies.json"
    dependencies = json.loads(path.read_text())
    dependencies["launch_kernel"].update(fields)
    path.write_text(json.dumps(dependencies))


@pytest.mark.parametrize("version,branch,tag,launch_version", [
    ("1.2.0rc1", "release/1.2", "v1.2.0rc1", "1.2.0rc1"),
    ("1.2.0", "release/1.2", "v1.2.0", "1.2.0"),
    ("1.2.1", "release/1.2", "v1.2.1", "1.2.0"),
    ("1.3.0.dev0", "main", "", "1.2.0"),
    ("1.3.0.dev0", "main", "", "1.3.0rc1"),
    ("1.3.0.dev0", "main", "", "1.3.0.dev0"),
])
def test_supported_versions_match_cmake(source, version, branch, tag, launch_version):
    set_version(source, version)
    launch_base = version_check.VERSION_PATTERN.fullmatch(launch_version)["base"]
    set_launch(source, release_version=launch_version, version=launch_base, cmake_version=launch_base)
    result = version_check.check(source, branch=branch, tag=tag)
    driver = source / "check.cmake"
    driver.write_text('include("${CMAKE_CURRENT_LIST_DIR}/release/version.cmake")\n'
                      'file(WRITE "${CMAKE_CURRENT_LIST_DIR}/result.txt" '
                      '"${RHINOFORGE_VERSION}|${RHINOFORGE_CMAKE_VERSION}|${RHINOFORGE_VERSION_STAGE}|'
                      '${RHINOFORGE_LAUNCH_RELEASE_VERSION}|${RHINOFORGE_LAUNCH_CMAKE_VERSION}")\n')
    subprocess.run(["cmake", "-P", str(driver)], check=True)
    stage = result["stage"] if result["stage"] != "stable" else ""
    assert (source / "result.txt").read_text() == (
        f'{version}|{result["cmake_version"]}|{stage}|'
        f'{result["launch_release_version"]}|{result["launch_cmake_version"]}'
    )


@pytest.mark.parametrize("version,branch,tag", [
    ("1.2.0", "main", ""),
    ("1.3.0.dev0", "release/1.3", ""),
    ("1.2.0rc1", "release/1.3", ""),
    ("1.2.0rc1", "", "v1.2.0"),
    ("1.3.0.dev0", "", "v1.3.0.dev0"),
])
def test_rejects_wrong_branch_or_tag(source, version, branch, tag):
    set_version(source, version)
    with pytest.raises(ValueError):
        version_check.check(source, branch=branch, tag=tag)


@pytest.mark.parametrize("field,value,error", [
    ("release_version", "1.02.0", "canonical"),
    ("release_version", "1.2.0rc0", "canonical"),
    ("release_version", "1.2.0.dev01", "canonical"),
    ("release_version", "1.2.0+local", "canonical"),
    ("release_version", 120, "canonical"),
    ("release_version", "1.3.0", "numeric version"),
    ("cmake_version", "1.3.0", "numeric version"),
    ("version", "1.3.0", "numeric version"),
])
def test_rejects_invalid_or_inconsistent_launch_versions(source, field, value, error):
    set_launch(source, **{field: value})
    with pytest.raises(ValueError, match=error):
        version_check.check(source)
    failed = subprocess.run(["cmake", "-P", str(source / "release/version.cmake")],
                            capture_output=True, text=True)
    assert failed.returncode != 0
    assert error in " ".join(failed.stderr.split())


@pytest.mark.parametrize("field,value,error", [
    ("abi_version", 0, "positive integer"),
    ("abi_version", "1", "positive integer"),
    ("commit", "3dd1b410", "full commit"),
])
def test_rejects_invalid_launch_abi_or_unpinned_source(source, field, value, error):
    set_launch(source, **{field: value})
    with pytest.raises(ValueError, match=error):
        version_check.check(source)


def test_existing_build_reconfigures_after_version_or_lock_changes(source):
    set_version(source, "1.2.0rc1")
    (source / "CMakeLists.txt").write_text(
        'cmake_minimum_required(VERSION 3.19)\n'
        'include(release/version.cmake)\n'
        'project(VersionCheck VERSION ${RHINOFORGE_CMAKE_VERSION} LANGUAGES NONE)\n'
        'file(WRITE "${CMAKE_BINARY_DIR}/version.txt" "${RHINOFORGE_VERSION}")\n')
    build = source / "build"
    subprocess.run(["cmake", "-S", str(source), "-B", str(build)], check=True)
    set_version(source, "1.2.0rc2")
    subprocess.run(["cmake", "--build", str(build)], check=True)
    assert (build / "version.txt").read_text() == "1.2.0rc2"
    set_launch(source, cmake_version="1.3.0")
    failed = subprocess.run(["cmake", "--build", str(build)], capture_output=True)
    assert failed.returncode != 0


def test_source_and_installed_package_resolve_their_own_version(tmp_path):
    source_version = subprocess.check_output(
        [sys.executable, "-c", "import runpy; print(runpy.run_path('python/rpu_backend/_version.py')['DISTRIBUTION_VERSION'])"],
        cwd=ROOT, text=True).strip()
    assert source_version == (ROOT / "VERSION").read_text().strip()
    package = tmp_path / "site-packages/rpu_backend"
    package.mkdir(parents=True)
    shutil.copyfile(ROOT / "python/rpu_backend/_version.py", package / "_version.py")
    metadata = package.parent / "rhinoforge-1.2.0rc1.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: rhinoforge\nVersion: 1.2.0rc1\n")
    installed_version = subprocess.check_output([
        sys.executable, "-I", "-c",
        "import runpy, sys; sys.path.insert(0, sys.argv[1]); "
        "print(runpy.run_path(sys.argv[2])['DISTRIBUTION_VERSION'])",
        str(package.parent), str(package / "_version.py"),
    ], text=True).strip()
    assert installed_version == "1.2.0rc1"
