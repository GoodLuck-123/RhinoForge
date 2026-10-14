import sys
from types import SimpleNamespace
import pytest

def _make_entry_point(name: str, register_callable):
    ep = SimpleNamespace(name=name, group='rpu_backend.plugins')
    ep.load = lambda : register_callable
    return ep

class _FakeAdapter:

    @classmethod
    def preflight(cls, config):
        pass

class _AltFakeAdapter:

    @classmethod
    def preflight(cls, config):
        pass

def _plugin_register_novel(api):
    api.register_adapter('FakeForCausalLM', _FakeAdapter)

def _plugin_register_shadow(api):
    api.register_adapter('Qwen3ForCausalLM', _FakeAdapter)

def _plugin_register_same_class_twice(api):
    api.register_adapter('NovelArchForCausalLM', _FakeAdapter)
    api.register_adapter('NovelArchForCausalLM', _FakeAdapter)

def _plugin_register_different_class_twice(api):
    api.register_adapter('NovelArchForCausalLM', _FakeAdapter)
    api.register_adapter('NovelArchForCausalLM', _AltFakeAdapter)

def test_cold_bootstrap_defers_plugin_load_until_unknown_arch(monkeypatch):
    from rpu_backend.runtime import registry
    from rpu_backend.tests.fixtures._unsafe_test_hooks import _test_only_unregister
    _test_only_unregister('FakeForCausalLM')
    loaded = []
    ep = _make_entry_point('fake_lazy', _plugin_register_novel)
    original_load = ep.load

    def tracked_load():
        loaded.append(ep.name)
        return original_load()
    ep.load = tracked_load

    def _fake_entry_points(group=None):
        return [ep] if group == 'rpu_backend.plugins' else []
    monkeypatch.setattr('importlib.metadata.entry_points', _fake_entry_points)
    monkeypatch.setattr(registry, '_PLUGINS_DISCOVERED', False)
    registry.discover_all(())
    assert loaded == []
    assert 'FakeForCausalLM' not in registry.list_adapters()
    assert registry.get_adapter('FakeForCausalLM') is _FakeAdapter
    assert loaded == ['fake_lazy']

@pytest.fixture(autouse=True)
def _ensure_rpu_backend_imported():
    import rpu_backend
    yield
    from rpu_backend.tests.fixtures._unsafe_test_hooks import _test_only_unregister
    _test_only_unregister('FakeForCausalLM')
    _test_only_unregister('NovelArchForCausalLM')

def test_external_plugin_novel_arch_dispatches(monkeypatch):
    from rpu_backend.tests.fixtures._unsafe_test_hooks import _test_only_unregister
    _test_only_unregister('FakeForCausalLM')

    def _fake_entry_points(group=None):
        if group == 'rpu_backend.plugins':
            return [_make_entry_point('fake_novel', _plugin_register_novel)]
        return []
    monkeypatch.setattr('importlib.metadata.entry_points', _fake_entry_points)
    from rpu_backend.runtime.registry import discover_entry_points
    discover_entry_points()
    from rpu_backend.runtime.registry import get_adapter
    assert get_adapter('FakeForCausalLM') is _FakeAdapter

def test_external_plugin_shadow_raises(monkeypatch):
    from rpu_backend.runtime.registry import list_adapters
    assert 'Qwen3ForCausalLM' in list_adapters(), f'Precondition failed: Qwen3ForCausalLM not in built-ins: {list_adapters()!r}'

    def _fake_entry_points(group=None):
        if group == 'rpu_backend.plugins':
            return [_make_entry_point('fake_shadow', _plugin_register_shadow)]
        return []
    monkeypatch.setattr('importlib.metadata.entry_points', _fake_entry_points)
    from rpu_backend.runtime.registry import discover_entry_points
    with pytest.raises(RuntimeError) as excinfo:
        discover_entry_points()
    msg = str(excinfo.value)
    assert 'Qwen3ForCausalLM is a built-in' in msg, f'REG-03 text missing: {msg!r}'
    assert 'plugins cannot override' in msg, f'REG-03 text missing: {msg!r}'
    assert 'docs/api_reference.md#Migration' in msg, f'REG-03 Migration anchor missing: {msg!r}'
    from rpu_backend.runtime.registry import ADAPTERS
    assert ADAPTERS['Qwen3ForCausalLM'] is not _FakeAdapter, 'Shadow attempt mutated ADAPTERS — snapshot boundary failed'

def test_external_plugin_reinvocation_same_class_idempotent(monkeypatch):
    from rpu_backend.tests.fixtures._unsafe_test_hooks import _test_only_unregister
    _test_only_unregister('NovelArchForCausalLM')

    def _fake_entry_points(group=None):
        if group == 'rpu_backend.plugins':
            return [_make_entry_point('fake_same_twice', _plugin_register_same_class_twice)]
        return []
    monkeypatch.setattr('importlib.metadata.entry_points', _fake_entry_points)
    from rpu_backend.runtime.registry import discover_entry_points
    from rpu_backend.runtime.registry import get_adapter
    from rpu_backend.runtime.registry import ADAPTERS
    discover_entry_points()
    assert get_adapter('NovelArchForCausalLM') is _FakeAdapter
    assert ADAPTERS['NovelArchForCausalLM'] is _FakeAdapter, 'Idempotent re-register should not mutate the ADAPTERS entry identity'

def test_external_plugin_reinvocation_different_class_raises(monkeypatch):
    from rpu_backend.tests.fixtures._unsafe_test_hooks import _test_only_unregister
    _test_only_unregister('NovelArchForCausalLM')

    def _fake_entry_points(group=None):
        if group == 'rpu_backend.plugins':
            return [_make_entry_point('fake_diff_twice', _plugin_register_different_class_twice)]
        return []
    monkeypatch.setattr('importlib.metadata.entry_points', _fake_entry_points)
    from rpu_backend.runtime.registry import discover_entry_points
    with pytest.raises(RuntimeError) as excinfo:
        discover_entry_points()
    msg = str(excinfo.value)
    assert 'NovelArchForCausalLM' in msg, f'arch name missing from conflict msg: {msg!r}'
    assert 'Adapter already registered' in msg or 'already registered' in msg, f'duplicate-class raise message did not match preserved register_adapter semantics: {msg!r}'
    assert 'is a built-in' not in msg, f'Wrong message: got REG-03 built-in shadow message for a non-built-in plugin-vs-plugin collision: {msg!r}'
    from rpu_backend.runtime.registry import ADAPTERS
    assert ADAPTERS['NovelArchForCausalLM'] is _FakeAdapter, 'Second invocation mutated ADAPTERS despite the conflict-raise'

def main() -> int:
    print('RUN via pytest: pytest python/rpu_backend/tests/acceptance/test_external_plugin.py')
    return 0
if __name__ == '__main__':
    sys.exit(main())
