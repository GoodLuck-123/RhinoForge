import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import pytest
ROOT = Path(__file__).resolve().parents[1]
LIB = importlib.import_module('examples.common.cold_libraries')

@pytest.fixture(autouse=True)
def cold_process(monkeypatch):
    for key in (*LIB.LIBRARY_KEYS.values(), LIB.AUTOLOAD, 'RPU_CHUNK_FORCE_UNSAFE'):
        monkeypatch.delenv(key, raising=False)
    for name in ('torch', 'rpu_backend', 'rpu_backend._cpp_ext'):
        monkeypatch.delitem(sys.modules, name, raising=False)

@pytest.fixture
def artifacts(tmp_path):
    root = tmp_path / 'checkout'
    source = root / 'src'
    source.mkdir(parents=True)
    manifest = root / 'release'
    manifest.mkdir()
    paths = {}
    records = {}
    for (role, filename) in (('main', 'rhinoOpLib_current.ref'), ('expansion', 'rhinoExpansionOpLib_current.ref')):
        target = source / filename
        target.write_bytes((role + '-current').encode())
        paths[role] = str(source / filename)
        records['rhino_ops_' + role] = {'path': filename, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
    (manifest / 'runtime-dependencies.json').write_text(json.dumps({'artifacts': records}))
    native = tmp_path / 'native.so'
    native.write_bytes(b'native')
    paths['native'] = str(native)
    return (root, paths)

def test_schema_accepts_placeholders_but_rejects_partial_or_unknown_roles():
    LIB.validate_libraries(None)
    LIB.validate_libraries({name: '/placeholder/' + name for name in LIB.LIBRARY_KEYS})
    for invalid in ({}, [], {'native': '/native.so'}, {'native': '/n', 'main': '/m', 'expansion': ''}, {'native': '/n', 'main': True, 'expansion': '/e'}, {'native': '/n', 'main': '/m', 'expansion': '/e', 'extra': 'x'}):
        with pytest.raises(ValueError, match='native, main, and expansion'):
            LIB.validate_libraries(invalid)

def test_explicit_libraries_resolve_real_files_and_identity(artifacts):
    (root, paths) = artifacts
    (env, records) = LIB.bind_libraries({'libraries': paths}, root)
    assert set(env) == set(LIB.LIBRARY_KEYS.values())
    for (role, key) in LIB.LIBRARY_KEYS.items():
        path = Path(paths[role]).resolve()
        assert env[key] == str(path)
        assert records[key] == {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    assert not any((key in os.environ for key in env))

def test_explicit_libraries_require_files_and_distinct_ref_targets(artifacts, tmp_path):
    (root, paths) = artifacts
    with pytest.raises(FileNotFoundError):
        LIB.bind_libraries({'libraries': paths | {'native': str(tmp_path / 'missing.so')}}, root)
    with pytest.raises(ValueError, match='must be a file'):
        LIB.bind_libraries({'libraries': paths | {'native': str(tmp_path)}}, root)
    alias = tmp_path / 'expansion_alias.ref'
    alias.symlink_to(paths['main'])
    with pytest.raises(ValueError, match='checksum mismatch'):
        LIB.bind_libraries({'libraries': paths | {'expansion': str(alias)}}, root)

@pytest.fixture(params=['g05', 'hyvla'])
def library_binder(request):
    if request.param == 'g05':
        return LIB.bind_libraries
    runner = importlib.import_module('examples.runners.hy_vla.optimized')
    return lambda config, root: runner.bind_libraries(SimpleNamespace(libraries=config.get('libraries')), root)

@pytest.mark.parametrize('selection', ['explicit', 'environment'])
@pytest.mark.parametrize('fault', ['main', 'expansion', 'swapped'])
def test_explicit_and_environment_libraries_reject_unpublished_role_bytes(monkeypatch, artifacts, tmp_path, library_binder, selection, fault):
    (root, paths) = artifacts
    selected = dict(paths)
    if fault == 'swapped':
        (selected['main'], selected['expansion']) = (selected['expansion'], selected['main'])
    else:
        old = tmp_path / 'round31.ref'
        old.write_bytes(b'preserved-round31')
        selected[fault] = str(old)
    config = {'libraries': selected}
    if selection == 'environment':
        config = {}
        for (role, key) in LIB.LIBRARY_KEYS.items():
            monkeypatch.setenv(key, selected[role])
    before = dict(os.environ)
    with pytest.raises(ValueError, match='checksum mismatch'):
        library_binder(config, root)
    assert dict(os.environ) == before

@pytest.mark.parametrize('selection', ['explicit', 'environment'])
def test_published_libraries_allow_installed_copies_without_checkout_payloads(monkeypatch, artifacts, tmp_path, library_binder, selection):
    (root, paths) = artifacts
    installed = dict(paths)
    for role in ('main', 'expansion'):
        original = Path(paths[role])
        copied = tmp_path / (role + '.ref')
        copied.write_bytes(original.read_bytes())
        original.unlink()
        installed[role] = str(copied)
    config = {'libraries': installed}
    if selection == 'environment':
        config = {}
        for (role, key) in LIB.LIBRARY_KEYS.items():
            monkeypatch.setenv(key, installed[role])
    monkeypatch.delattr(hashlib, 'file_digest', raising=False)
    (env, records) = library_binder(config, root)
    for (role, key) in LIB.LIBRARY_KEYS.items():
        assert env[key] == installed[role]
        assert records[key]['sha256'] == hashlib.sha256(Path(installed[role]).read_bytes()).hexdigest()

@pytest.mark.parametrize('selection', ['explicit', 'environment'])
def test_complete_overrides_cannot_bypass_missing_manifest(monkeypatch, artifacts, library_binder, selection):
    (root, paths) = artifacts
    (root / 'release/runtime-dependencies.json').unlink()
    config = {'libraries': paths}
    if selection == 'environment':
        config = {}
        for (role, key) in LIB.LIBRARY_KEYS.items():
            monkeypatch.setenv(key, paths[role])
    with pytest.raises(FileNotFoundError, match='runtime-dependencies'):
        library_binder(config, root)

def test_default_files_use_existing_manifest_checksum_validation(artifacts, monkeypatch):
    (root, paths) = artifacts
    monkeypatch.setenv('RPU_KERNEL_LIB_PATH', paths['main'])
    monkeypatch.setenv('RPU_KERNEL_LIB_PATH_EXPANSION', paths['expansion'])
    (env, records) = LIB.bind_libraries({}, root)
    assert 'RPU_BACKEND_SO' not in env
    assert set(records) == {'RPU_KERNEL_LIB_PATH', 'RPU_KERNEL_LIB_PATH_EXPANSION'}
    Path(paths['main']).write_bytes(b'replaced')
    with pytest.raises(ValueError, match='checksum mismatch'):
        LIB.bind_libraries({}, root)

def test_cold_binding_accepts_equivalent_symlink_paths_and_rejects_conflicts(monkeypatch, artifacts, tmp_path):
    (root, paths) = artifacts
    monkeypatch.setenv('RPU_KERNEL_LIB_PATH', paths['main'])
    monkeypatch.setenv(LIB.AUTOLOAD, '0')
    monkeypatch.setitem(sys.modules, 'torch', ModuleType('torch'))
    (env, _) = LIB.bind_libraries({'libraries': paths}, root)
    LIB.bind_cold_environment(env)
    assert os.environ['RPU_KERNEL_LIB_PATH'] == str(Path(paths['main']).resolve())
    other = tmp_path / 'other.so'
    other.write_bytes(b'other')
    before = dict(os.environ)
    with pytest.raises(ValueError, match='RPU_BACKEND_SO'):
        LIB.bind_cold_environment(env | {'RPU_BACKEND_SO': str(other), 'G05_TEST_NOT_SET': '1'})
    assert dict(os.environ) == before

@pytest.mark.parametrize('fault', ['backend', 'extension', 'autoload', 'torch_before_binding', 'unsafe', 'pending_unsafe', 'pending_autoload'])
def test_cold_binding_rejects_imported_runtime_autoload_and_unsafe(monkeypatch, fault):
    pending = {'G05_TEST_NOT_SET': '1'}
    if fault == 'backend':
        monkeypatch.setitem(sys.modules, 'rpu_backend', ModuleType('rpu_backend'))
    elif fault == 'extension':
        monkeypatch.setitem(sys.modules, 'rpu_backend._cpp_ext', ModuleType('rpu_backend._cpp_ext'))
    elif fault == 'autoload':
        monkeypatch.setenv(LIB.AUTOLOAD, '1')
    elif fault == 'torch_before_binding':
        monkeypatch.setitem(sys.modules, 'torch', ModuleType('torch'))
    elif fault == 'unsafe':
        monkeypatch.setenv('RPU_CHUNK_FORCE_UNSAFE', '1')
    elif fault == 'pending_unsafe':
        pending['RPU_CHUNK_FORCE_UNSAFE'] = '1'
    elif fault == 'pending_autoload':
        pending[LIB.AUTOLOAD] = '1'
    before = dict(os.environ)
    with pytest.raises(ValueError):
        LIB.bind_cold_environment(pending)
    assert dict(os.environ) == before

def _loaded(monkeypatch, paths, records):
    actual = {'extension': paths['native'], 'oplib_roles': {}, 'oplib_payloads': {}}
    for role in ('main', 'expansion'):
        record = records[LIB.LIBRARY_KEYS[role]]
        actual['oplib_roles'][role] = record['path']
        actual['oplib_payloads'][record['path']] = {'sha256': record['sha256']}
    backend = ModuleType('rpu_backend')
    backend._cpp_ext = SimpleNamespace(get_runtime_payload_paths=lambda : actual)
    monkeypatch.setitem(sys.modules, 'rpu_backend', backend)
    return actual

def test_loaded_snapshots_and_native_match_selected_identities(monkeypatch, artifacts):
    (root, paths) = artifacts
    (_, records) = LIB.bind_libraries({'libraries': paths}, root)
    actual = _loaded(monkeypatch, paths, records)
    assert LIB.runtime_payloads(records) is actual
    Path(paths['expansion']).write_bytes(b'changed-after-snapshot')
    assert LIB.runtime_payloads(records) is actual
    before = copy.deepcopy(records)
    actual['oplib_payloads'][actual['oplib_roles']['expansion']]['sha256'] = 'wrong'
    with pytest.raises(RuntimeError, match='loaded expansion'):
        LIB.runtime_payloads(records)
    assert records == before

@pytest.mark.parametrize('fault', ['main_path', 'main_digest', 'expansion_path', 'expansion_digest', 'native_path', 'native_digest'])
def test_loaded_role_and_native_mismatches_fail(monkeypatch, artifacts, tmp_path, fault):
    (root, paths) = artifacts
    (_, records) = LIB.bind_libraries({'libraries': paths}, root)
    actual = _loaded(monkeypatch, paths, records)
    (role, kind) = fault.split('_')
    if role == 'native':
        if kind == 'path':
            other = tmp_path / 'other.so'
            other.write_bytes(Path(paths['native']).read_bytes())
            actual['extension'] = str(other)
        else:
            Path(paths['native']).write_bytes(b'changed-native')
    elif kind == 'path':
        actual['oplib_roles'][role] = '/different/' + role + '.ref'
    else:
        actual['oplib_payloads'][actual['oplib_roles'][role]]['sha256'] = 'wrong'
    with pytest.raises(RuntimeError, match='loaded ' + role):
        LIB.runtime_payloads(records)

def test_default_native_identity_is_recorded_after_import(monkeypatch, artifacts):
    (root, paths) = artifacts
    monkeypatch.setenv('RPU_KERNEL_LIB_PATH', paths['main'])
    monkeypatch.setenv('RPU_KERNEL_LIB_PATH_EXPANSION', paths['expansion'])
    (_, records) = LIB.bind_libraries({}, root)
    actual = _loaded(monkeypatch, paths, records)
    assert LIB.runtime_payloads(records) is actual
    native = Path(paths['native'])
    assert records['RPU_BACKEND_SO'] == {'path': str(native), 'sha256': hashlib.sha256(native.read_bytes()).hexdigest()}
