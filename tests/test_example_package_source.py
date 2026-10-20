from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.common import libraries


@pytest.fixture
def installed_package(tmp_path, monkeypatch):
    source = tmp_path / "source"
    package = source / "python/rpu_backend"
    installed = tmp_path / "env/lib/python3.12/site-packages"
    for relative in ("__init__.py", "api/wall_oss.py"):
        for parent in (package, installed / "rpu_backend"):
            path = parent / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# matching runtime\n")
    (source / "VERSION").write_text("1.2.0\n")
    excluded = package / "tests/test_only_in_source.py"
    excluded.parent.mkdir()
    excluded.write_text("# excluded from wheel\n")
    files = [Path("rpu_backend/__init__.py"), Path("rpu_backend/api/wall_oss.py")]
    distribution = SimpleNamespace(
        files=files, version="1.2.0", locate_file=lambda relative: installed / relative)
    monkeypatch.setattr(libraries.metadata, "distribution", lambda name: distribution)
    monkeypatch.setattr(libraries.sysconfig, "get_path", lambda kind: str(installed))
    return source, installed / "rpu_backend/__init__.py", distribution


def test_matching_checkout_does_not_require_wheel_metadata(installed_package, monkeypatch):
    source, _, _ = installed_package
    def missing(name):
        raise libraries.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(libraries.metadata, "distribution", missing)
    origin = source / "python/rpu_backend/__init__.py"
    assert libraries.verify_backend_source(source, origin) == origin


def test_matching_wheel_ignores_excluded_source_tests(installed_package):
    source, origin, _ = installed_package
    assert libraries.verify_backend_source(source, origin) == origin


@pytest.mark.parametrize("mismatch", ["checkout", "metadata", "version", "python", "missing"])
def test_wheel_source_mismatch_is_rejected(installed_package, tmp_path, mismatch):
    source, origin, distribution = installed_package
    if mismatch == "checkout":
        origin = tmp_path / "other-checkout/python/rpu_backend/__init__.py"
    elif mismatch == "metadata":
        distribution.locate_file = lambda relative: tmp_path / "other-env" / relative
    elif mismatch == "version":
        distribution.version = "1.1.0"
    elif mismatch == "python":
        (origin.parent / "api/wall_oss.py").write_text("# another build\n")
    else:
        (origin.parent / "api/wall_oss.py").unlink()
    with pytest.raises(RuntimeError, match="wrong source checkout or installed wheel"):
        libraries.verify_backend_source(source, origin)
