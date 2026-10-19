from __future__ import annotations
import errno
import importlib.util
import os
import pathlib
import sys
import types
import pytest
REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
LOADER_PATH = REPO_ROOT / 'python' / 'rpu_backend' / '_native_loader.py'

def _load_source_module():
    spec = importlib.util.spec_from_file_location('_rpu_native_loader_test', LOADER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_device_preflight_requires_rpu_and_mem_and_closes_both():
    loader = _load_source_module()
    opened = []
    closed = []

    def open_fn(path, flags):
        opened.append((path, flags))
        return len(opened) + 10
    assert loader.device_nodes_accessible(rpu_paths=('/dev/rpu-test',), mem_path='/dev/mem-test', open_fn=open_fn, close_fn=closed.append)
    assert [path for (path, _) in opened] == ['/dev/rpu-test', '/dev/mem-test']
    assert closed == [12, 11]

def test_device_preflight_fails_closed_and_closes_partial_open():
    loader = _load_source_module()
    closed = []

    def open_fn(path, flags):
        if path.endswith('mem-test'):
            raise PermissionError(path)
        return 17
    assert not loader.device_nodes_accessible(rpu_paths=('/dev/rpu-test',), mem_path='/dev/mem-test', open_fn=open_fn, close_fn=closed.append)
    assert closed == [17]

def test_device_preflight_tries_other_rpu_nodes_and_uses_read_write():
    loader = _load_source_module()
    opened, closed = [], []

    def open_fn(path, flags):
        opened.append(path)
        assert flags & os.O_RDWR
        if path == '/dev/rpu0':
            raise PermissionError(errno.EACCES, 'Permission denied', path)
        return len(opened) + 10

    loader._check_device_nodes(rpu_paths=('/dev/rpu0', '/dev/rpu1'),
                               open_fn=open_fn, close_fn=closed.append)
    assert opened == ['/dev/rpu0', '/dev/rpu1', '/dev/mem']
    assert closed == [13, 12]

def test_device_preflight_reports_every_failed_rpu_candidate():
    loader = _load_source_module()
    closed = []

    def open_fn(path, flags):
        code = errno.EACCES if path == '/dev/rpu0' else errno.ENODEV
        raise OSError(code, os.strerror(code), path)

    with pytest.raises(OSError) as caught:
        loader._check_device_nodes(rpu_paths=('/dev/rpu0', '/dev/rpu1'),
                                   open_fn=open_fn, close_fn=closed.append)
    assert '/dev/rpu0' in str(caught.value) and '/dev/rpu1' in str(caught.value)
    assert 'Errno 13' in str(caught.value) and 'Errno 19' in str(caught.value)
    assert caught.value.errno == errno.ENODEV
    assert closed == []

def test_device_preflight_reports_missing_nodes():
    loader = _load_source_module()
    with pytest.raises(FileNotFoundError) as caught:
        loader._check_device_nodes(rpu_paths=())
    assert caught.value.errno == errno.ENOENT
    assert caught.value.filename == '/dev/rpu*'

def test_device_preflight_preserves_mem_errno_and_closes_partial_open():
    loader = _load_source_module()
    closed = []

    def open_fn(path, flags):
        if path == '/dev/mem':
            raise PermissionError(errno.EACCES, 'Permission denied', path)
        return 17

    with pytest.raises(PermissionError) as caught:
        loader._check_device_nodes(rpu_paths=('/dev/rpu0',),
                                   open_fn=open_fn, close_fn=closed.append)
    assert caught.value.errno == errno.EACCES
    assert caught.value.filename == '/dev/mem'
    assert closed == [17]

@pytest.mark.parametrize('deny_mem', [False, True])
def test_device_preflight_close_failure_cleans_remaining_fds_and_preserves_open_error(deny_mem):
    loader = _load_source_module()
    closed = []

    def open_fn(path, flags):
        if path == '/dev/mem':
            if deny_mem:
                raise PermissionError(errno.EACCES, 'Permission denied', path)
            return 18
        return 17

    def close_fn(fd):
        closed.append(fd)
        raise OSError(errno.EIO, 'synthetic close failure')

    with pytest.raises(OSError) as caught:
        loader._check_device_nodes(rpu_paths=('/dev/rpu0',),
                                   open_fn=open_fn, close_fn=close_fn)
    assert closed == ([17] if deny_mem else [18, 17])
    assert caught.value.errno == (errno.EACCES if deny_mem else errno.EIO)
    if deny_mem:
        assert caught.value.filename == '/dev/mem'
        assert 'cleanup also failed' in caught.value.__notes__[0]

def test_explicit_device_request_rechecks_permissions_after_native_import(monkeypatch):
    import rpu_backend
    from rpu_backend.api.errors import RPUBackendError
    loader = _load_source_module()
    monkeypatch.setattr(rpu_backend, '_cpp_loaded', True)
    allowed = True

    def check():
        if not allowed:
            raise PermissionError(errno.EACCES, 'Permission denied', '/dev/mem')

    monkeypatch.setattr(loader, '_check_device_nodes', check)
    loader.require_native_backend()
    allowed = False
    with pytest.raises(RPUBackendError, match=r'Errno 13.*?/dev/mem') as caught:
        loader.require_native_backend()
    assert isinstance(caught.value.__cause__, PermissionError)
    assert 'administrator' in str(caught.value)

def test_explicit_device_request_requires_native_registration_even_after_access_fixed(monkeypatch):
    import rpu_backend
    from rpu_backend.api.errors import RPUBackendError
    loader = _load_source_module()
    monkeypatch.setattr(rpu_backend, '_cpp_loaded', False)
    monkeypatch.setattr(loader, '_check_device_nodes', lambda: None)
    with pytest.raises(RPUBackendError, match='native backend is not loaded') as caught:
        loader.require_native_backend()
    assert 'rpu_backend.so' in str(caught.value)
    assert 'new Python process' in str(caught.value)

def test_loader_does_not_execute_native_code_without_device_access(monkeypatch, tmp_path, capsys):
    loader = _load_source_module()
    backend = tmp_path / 'rpu_backend.so'
    backend.touch()
    monkeypatch.setattr(loader, 'find_shared_object', lambda : str(backend))
    def deny_mem():
        raise PermissionError(errno.EACCES, 'Permission denied', '/dev/mem')
    monkeypatch.setattr(loader, '_check_device_nodes', deny_mem)
    monkeypatch.setattr(loader, '_load_extension', lambda *args: (_ for _ in ()).throw(AssertionError('native load attempted')))
    (cpp_ext, loaded) = loader.load_cpp_extension('rpu_backend_loader_test')
    assert not loaded
    assert cpp_ext.__name__ == 'rpu_backend_loader_test._cpp_ext'
    warning = capsys.readouterr().out
    assert '/dev/mem' in warning and 'Errno 13' in warning

def test_explicit_shared_object_precedes_package_and_build_candidates(tmp_path):
    loader = _load_source_module()
    package_dir = tmp_path / 'repo' / 'python' / 'rpu_backend'
    package_dir.mkdir(parents=True)
    packaged = package_dir / 'rpu_backend.so'
    packaged.touch()
    explicit = tmp_path / 'explicit' / 'rpu_backend.so'
    explicit.parent.mkdir()
    explicit.touch()
    found = loader.find_shared_object(package_dir=str(package_dir), executable='/opt/test-env/bin/python', environ={'RPU_BACKEND_SO': str(explicit)})  # lint-exempt: hardcoded-path; synthetic executable search fixture
    assert found == str(explicit)

def test_environment_build_precedes_generic_build_root(tmp_path):
    loader = _load_source_module()
    repo_root = tmp_path / 'repo'
    package_dir = repo_root / 'python' / 'rpu_backend'
    package_dir.mkdir(parents=True)
    preferred = repo_root / 'build_test-env' / 'nested' / 'rpu_backend.so'
    preferred.parent.mkdir(parents=True)
    preferred.touch()
    generic = repo_root / 'build' / 'rpu_backend.so'
    generic.parent.mkdir()
    generic.touch()
    found = loader.find_shared_object(package_dir=str(package_dir), executable='/opt/test-env/bin/python', environ={})  # lint-exempt: hardcoded-path; synthetic executable search fixture
    assert found == str(preferred)

def test_unrelated_custom_build_root_is_not_auto_selected(tmp_path):
    loader = _load_source_module()
    repo_root = tmp_path / 'repo'
    package_dir = repo_root / 'python' / 'rpu_backend'
    package_dir.mkdir(parents=True)
    unrelated = repo_root / 'build_other-env' / 'rpu_backend.so'
    unrelated.parent.mkdir(parents=True)
    unrelated.touch()
    found = loader.find_shared_object(package_dir=str(package_dir), executable='/opt/test-env/bin/python', environ={})  # lint-exempt: hardcoded-path; synthetic executable search fixture
    assert found is None

def test_missing_explicit_shared_object_fails_instead_of_falling_back(tmp_path):
    loader = _load_source_module()
    package_dir = tmp_path / 'repo' / 'python' / 'rpu_backend'
    package_dir.mkdir(parents=True)
    fallback = tmp_path / 'repo' / 'build' / 'rpu_backend.so'
    fallback.parent.mkdir()
    fallback.touch()
    missing = tmp_path / 'missing' / 'rpu_backend.so'
    try:
        loader.find_shared_object(package_dir=str(package_dir), executable='/opt/test-env/bin/python', environ={'RPU_BACKEND_SO': str(missing)})  # lint-exempt: hardcoded-path; synthetic executable search fixture
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError('missing RPU_BACKEND_SO did not fail')

def test_extension_execution_restores_python_package_metadata(tmp_path):
    loader = _load_source_module()
    extension_source = tmp_path / 'fake_extension.py'
    extension_source.write_text("import sys, types\nreplacement = types.ModuleType('rpu_backend')\nreplacement.replaced = True\nsys.modules['rpu_backend'] = replacement\nnative_token = 7\n")
    original = sys.modules.get('rpu_backend')
    parent = types.ModuleType('rpu_backend')
    parent.__file__ = 'python-package-init.py'
    parent.__path__ = ['python-package']
    sys.modules['rpu_backend'] = parent
    try:
        extension = loader._load_extension(str(extension_source), 'rpu_backend')
        assert extension.native_token == 7
        assert sys.modules['rpu_backend'] is parent
        assert parent.__file__ == 'python-package-init.py'
        assert parent.__path__ == ['python-package']
    finally:
        if original is None:
            sys.modules.pop('rpu_backend', None)
        else:
            sys.modules['rpu_backend'] = original
