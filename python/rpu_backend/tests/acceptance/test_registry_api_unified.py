import os
import subprocess
import sys
import textwrap
from pathlib import Path
from rpu_backend.tests.acceptance import COLD_IMPORT_TIMEOUT_S, timeout_message
_PYTHON_ROOT = Path(__file__).resolve().parents[4] / 'python'

def test_internal_registry_public_names_reachable():
    from rpu_backend.runtime.registry import list_adapters, register_adapter, get_adapter
    assert callable(list_adapters), f'list_adapters not callable: {type(list_adapters)}'
    assert callable(register_adapter), f'register_adapter not callable: {type(register_adapter)}'
    assert callable(get_adapter), f'get_adapter not callable: {type(get_adapter)}'

def test_qwen3_in_list_adapters():
    import rpu_backend
    from rpu_backend.runtime.registry import list_adapters
    archs = list_adapters()
    assert isinstance(archs, list), f'list_adapters returned non-list: {type(archs)}'
    assert 'Qwen3ForCausalLM' in archs, f'ROADMAP SC#3 failure: Qwen3ForCausalLM not in list_adapters() = {archs!r}. Built-in manifest registration may have failed.'

def test_pi05_not_in_list_adapters():
    import rpu_backend
    from rpu_backend.runtime.registry import list_adapters
    archs = list_adapters()
    assert 'Pi05ForCausalLM' not in archs, f'Pi0.5 wrongly registered: {archs!r}'
    assert 'Pi05VLA' not in archs, f'Pi0.5 wrongly registered: {archs!r}'

def test_canonical_registry_matches_internal():
    from rpu_backend.runtime.registry import list_adapters as _i_list, register_adapter as _i_reg, get_adapter as _i_get
    from rpu_backend.runtime.registry import list_adapters as _c_list, register_adapter as _c_reg, get_adapter as _c_get
    assert _i_list is _c_list
    assert _i_reg is _c_reg
    assert _i_get is _c_get

def test_unsupported_model_message_shape():
    import rpu_backend
    from rpu_backend.runtime.registry import get_adapter, list_adapters
    from rpu_backend.api.errors import UnsupportedModelError
    try:
        get_adapter('NoSuchArchForCausalLM')
    except UnsupportedModelError as e:
        msg = str(e)
        assert 'NoSuchArchForCausalLM' in msg, msg
        assert 'docs/api_reference.md#Migration' in msg, msg
        assert repr(sorted(list_adapters())) in msg or 'Registered archs:' in msg, msg
    else:
        raise AssertionError('get_adapter did not raise for unknown arch')

def test_fresh_process_import_cycle_clean():
    probe_script = textwrap.dedent(f"""\n        import sys\n        sys.meta_path[:] = [\n            finder for finder in sys.meta_path\n            if not type(finder).__module__.startswith("_rpu_backend_editable")\n        ]\n        sys.path.insert(0, {str(_PYTHON_ROOT)!r})\n        import rpu_backend\n        from rpu_backend.runtime.registry import list_adapters\n        archs = list_adapters()\n        assert 'Qwen3ForCausalLM' in archs, f'Qwen3 not discovered in fresh process: {{archs!r}}'\n        print('OK: fresh-process import cycle clean')\n    """)
    try:
        result = subprocess.run([sys.executable, '-W', 'error::ImportWarning', '-c', probe_script], capture_output=True, text=True, timeout=COLD_IMPORT_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(timeout_message('H1 fresh-process cycle', exc)) from exc
    assert result.returncode == 0, f'H1 failure: fresh-process `import rpu_backend` exited with code {result.returncode}.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}'
    assert 'OK: fresh-process import cycle clean' in result.stdout, f'H1 failure: success sentinel missing.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}'
    assert 'Traceback (most recent call last):' not in result.stderr, f'H1 failure: uncaught traceback on stderr — fresh-process import surfaced an ImportWarning or partial-module error.\nstderr:\n{result.stderr}'

def test_hf_cache_env_defaults_are_coherent(monkeypatch, tmp_path):
    from rpu_backend.model_registry import set_hf_home_env
    monkeypatch.setenv('RPU_MODEL_CACHE', str(tmp_path / 'models'))
    for name in ('HF_HOME', 'HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE'):
        monkeypatch.delenv(name, raising=False)
    set_hf_home_env()
    root = tmp_path / 'models' / '.hf'
    assert Path(os.environ['HF_HOME']) == root
    assert Path(os.environ['HF_HUB_CACHE']) == root / 'hub'
    assert Path(os.environ['HUGGINGFACE_HUB_CACHE']) == root / 'hub'

def test_hf_cache_env_preserves_explicit_values(monkeypatch, tmp_path):
    from rpu_backend.model_registry import set_hf_home_env
    explicit_home = tmp_path / 'explicit-home'
    explicit_hub = tmp_path / 'explicit-hub'
    explicit_legacy = tmp_path / 'explicit-legacy'
    monkeypatch.setenv('HF_HOME', str(explicit_home))
    monkeypatch.setenv('HF_HUB_CACHE', str(explicit_hub))
    monkeypatch.setenv('HUGGINGFACE_HUB_CACHE', str(explicit_legacy))
    set_hf_home_env()
    assert Path(os.environ['HF_HOME']) == explicit_home
    assert Path(os.environ['HF_HUB_CACHE']) == explicit_hub
    assert Path(os.environ['HUGGINGFACE_HUB_CACHE']) == explicit_legacy

def test_hf_cache_env_derives_missing_hubs_from_explicit_home(monkeypatch, tmp_path):
    from rpu_backend.model_registry import set_hf_home_env
    explicit_home = tmp_path / 'explicit-home'
    monkeypatch.setenv('HF_HOME', str(explicit_home))
    monkeypatch.delenv('HF_HUB_CACHE', raising=False)
    monkeypatch.delenv('HUGGINGFACE_HUB_CACHE', raising=False)
    set_hf_home_env()
    assert Path(os.environ['HF_HUB_CACHE']) == explicit_home / 'hub'
    assert Path(os.environ['HUGGINGFACE_HUB_CACHE']) == explicit_home / 'hub'

def main() -> int:
    test_internal_registry_public_names_reachable()
    test_qwen3_in_list_adapters()
    test_pi05_not_in_list_adapters()
    test_canonical_registry_matches_internal()
    test_unsupported_model_message_shape()
    test_fresh_process_import_cycle_clean()
    print('PASS: REG-02 + ROADMAP SC#3 + REG-07 + H1 fresh-process registry')
    return 0
if __name__ == '__main__':
    sys.exit(main())
