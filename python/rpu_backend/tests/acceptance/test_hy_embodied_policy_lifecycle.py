from __future__ import annotations
import gc
import os
from pathlib import Path
import sys
import types
import weakref
import pytest
import torch
from rpu_backend.api import HyEmbodiedPolicy
from rpu_backend.api.errors import RPUSingleHandleError

class _Runner:

    def __init__(self, *, close_error: Exception | None=None):
        self.close_calls = 0
        self.close_error = close_error

    def close(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error

def _ckpt(tmp_path, name: str):
    root = tmp_path / name
    root.mkdir()
    for filename in ('model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'):
        (root / filename).write_bytes(b'{}')
    return root

def _policy(tmp_path, name: str, **kwargs):
    return HyEmbodiedPolicy.from_checkpoint(_ckpt(tmp_path, name), **kwargs)

def _install_builder(monkeypatch, build):
    from rpu_backend.adapters.hy_vla import runtime
    monkeypatch.setattr(runtime, 'build_hy_vla', build)

def _poisoning_builder(runner, calls, *, extra_env='RPU_TEST_HY_DEFAULT'):

    def build(*args, _env_snapshot, _owner, **kwargs):
        from rpu_backend.api.causal_lm import _claim_live_instance, _poison_live_instance
        calls.append((args, kwargs))
        _claim_live_instance(_owner)
        _poison_live_instance(_owner, 'synthetic Hy-VLA materialization')
        _env_snapshot.setdefault(extra_env, os.environ.get(extra_env))
        os.environ.setdefault(extra_env, 'builder-default')
        return runner
    return build

@pytest.fixture
def checkpoint_metadata(monkeypatch):
    from rpu_backend.adapters.hy_vla import weights
    shapes = weights._expected_checkpoint_shapes(weights.HyVlaConfig())

    class MetadataHandle:

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def keys(self):
            return shapes.keys()

        def get_slice(self, name):
            return types.SimpleNamespace(get_dtype=lambda : 'BF16', get_shape=lambda : shapes[name])

        def get_tensor(self, name):
            raise AssertionError(f'lifecycle fixture must not materialize {name}')

    def safe_open(path, *, framework, device):
        assert Path(path).is_file()
        assert (framework, device) == ('pt', 'cpu')
        return MetadataHandle()
    monkeypatch.setattr(weights, 'safe_open', safe_open)

@pytest.fixture
def fake_threads(monkeypatch):
    state = {'value': 11}
    monkeypatch.setattr(torch, 'get_num_threads', lambda : state['value'])
    monkeypatch.setattr(torch, 'set_num_threads', lambda value: state.__setitem__('value', value))
    return state

@pytest.mark.parametrize('cameras', [2, 3])
def test_success_is_process_terminal_even_after_close(tmp_path, monkeypatch, fake_threads, checkpoint_metadata, cameras):
    monkeypatch.setenv('RPU_HY_VLA_W8A16', 'ambient-profile')
    monkeypatch.setenv('RPU_HY_VLA_W4A16', 'ambient-w4')
    monkeypatch.setenv('RPU_TEST_CALLER_ENV', 'ambient-caller')
    monkeypatch.setenv('RPU_TEST_UNRELATED_ENV', 'ambient-unrelated')
    monkeypatch.delenv('RPU_TEST_HY_DEFAULT', raising=False)
    runner = _Runner()
    calls = []
    selected = {}
    base_builder = _poisoning_builder(runner, calls)

    def builder(*args, **kwargs):
        selected['w8'] = os.environ['RPU_HY_VLA_W8A16']
        selected['w4'] = os.environ['RPU_HY_VLA_W4A16']
        result = base_builder(*args, **kwargs)
        os.environ['RPU_TEST_UNRELATED_ENV'] = 'external-change'
        return result
    _install_builder(monkeypatch, builder)
    first = _policy(tmp_path, 'first', dtype='w8a16-expert', n_cameras=cameras, runtime_env={'RPU_TEST_CALLER_ENV': 'policy-caller'}, threads=3)
    second = _policy(tmp_path, 'second')
    assert first.to('rpu') is first
    assert calls[0][1]['cfg'].num_cameras == cameras
    assert calls[0][1]['prefix_len'] == (192 if cameras == 2 else 240)
    assert selected == {'w8': 'expert', 'w4': '0'}
    assert fake_threads['value'] == 3
    assert os.environ['RPU_TEST_CALLER_ENV'] == 'policy-caller'
    assert os.environ['RPU_TEST_HY_DEFAULT'] == 'builder-default'
    with pytest.raises(RPUSingleHandleError, match='Restart'):
        second.to('rpu')
    assert len(calls) == 1
    assert fake_threads['value'] == 3
    first.close()
    assert runner.close_calls == 1
    assert fake_threads['value'] == 11
    assert os.environ['RPU_HY_VLA_W8A16'] == 'ambient-profile'
    assert os.environ['RPU_HY_VLA_W4A16'] == 'ambient-w4'
    assert os.environ['RPU_TEST_CALLER_ENV'] == 'ambient-caller'
    assert os.environ['RPU_TEST_UNRELATED_ENV'] == 'external-change'
    assert 'RPU_TEST_HY_DEFAULT' not in os.environ
    with pytest.raises(RPUSingleHandleError, match='Restart'):
        second.to('rpu')
    assert len(calls) == 1

def test_decoder_preflight_failure_does_not_claim_or_mutate_process(tmp_path, monkeypatch, fake_threads, checkpoint_metadata):
    calls = []
    runner = _Runner()
    _install_builder(monkeypatch, _poisoning_builder(runner, calls))

    class Decoder:

        @staticmethod
        def from_norm_stats(*args, **kwargs):
            raise ValueError('synthetic decoder failure')
    decoder_module = types.ModuleType('rpu_backend.api.hy_action_decode')
    decoder_module.HyVlaActionDecoder = Decoder
    monkeypatch.setitem(sys.modules, decoder_module.__name__, decoder_module)
    broken = _policy(tmp_path, 'broken-decoder', norm_stats_path=tmp_path / 'bad.pkl', threads=2)
    with pytest.raises(ValueError, match='decoder failure'):
        broken.to('rpu')
    assert calls == []
    assert broken._rpu_build_started is False
    assert fake_threads['value'] == 11
    other = _policy(tmp_path, 'after-decoder')
    assert other.to('rpu') is other
    other.close()

def test_failure_after_materialization_restores_owned_globals_but_keeps_terminal(tmp_path, monkeypatch, fake_threads, checkpoint_metadata):
    monkeypatch.setenv('RPU_HY_VLA_W8A16', 'ambient')
    monkeypatch.setenv('RPU_HY_VLA_W4A16', 'ambient-w4')
    monkeypatch.delenv('RPU_TEST_HY_DEFAULT', raising=False)

    def fail(*args, _env_snapshot, _owner, **kwargs):
        from rpu_backend.api.causal_lm import _claim_live_instance, _poison_live_instance
        _claim_live_instance(_owner)
        _poison_live_instance(_owner, 'synthetic failed materialization')
        _env_snapshot.setdefault('RPU_TEST_HY_DEFAULT', os.environ.get('RPU_TEST_HY_DEFAULT'))
        os.environ['RPU_TEST_HY_DEFAULT'] = 'changed'
        raise RuntimeError('synthetic build failure')
    _install_builder(monkeypatch, fail)
    broken = _policy(tmp_path, 'broken', threads=2)
    other = _policy(tmp_path, 'other')
    with pytest.raises(RuntimeError, match='synthetic build failure'):
        broken.to('rpu')
    assert fake_threads['value'] == 11
    assert os.environ['RPU_HY_VLA_W8A16'] == 'ambient'
    assert os.environ['RPU_HY_VLA_W4A16'] == 'ambient-w4'
    assert 'RPU_TEST_HY_DEFAULT' not in os.environ
    with pytest.raises(RPUSingleHandleError, match='Restart'):
        other.to('rpu')

def test_gc_finalizer_closes_runner_and_restores_globals_but_not_reusability(tmp_path, monkeypatch, fake_threads, checkpoint_metadata):
    monkeypatch.delenv('RPU_TEST_HY_DEFAULT', raising=False)
    runner = _Runner()
    _install_builder(monkeypatch, _poisoning_builder(runner, []))
    policy = _policy(tmp_path, 'gc-owner', threads=4)
    other = _policy(tmp_path, 'gc-other')
    policy.to('rpu')
    ref = weakref.ref(policy)
    del policy
    gc.collect()
    assert ref() is None
    assert runner.close_calls == 1
    assert fake_threads['value'] == 11
    assert 'RPU_TEST_HY_DEFAULT' not in os.environ
    with pytest.raises(RPUSingleHandleError, match='Restart'):
        other.to('rpu')

def test_finalizer_restores_env_and_threads_even_when_runner_close_fails(monkeypatch, fake_threads):
    from rpu_backend.api.hy_embodied import _finalize_runner
    monkeypatch.setenv('RPU_TEST_HY_FINALIZE', 'changed')
    fake_threads['value'] = 3
    ok = _finalize_runner(_Runner(close_error=RuntimeError('close failed')), {'RPU_TEST_HY_FINALIZE': 'ambient'}, 11)
    assert ok is False
    assert os.environ['RPU_TEST_HY_FINALIZE'] == 'ambient'
    assert fake_threads['value'] == 11

def test_low_level_builder_also_claims_and_poison_process(tmp_path, monkeypatch, checkpoint_metadata):
    from rpu_backend.adapters.hy_vla import runtime

    class FakeLowLevelRunner:

        def __init__(self, **kwargs):
            self.kwargs = kwargs
    monkeypatch.setattr(os, 'environ', dict(os.environ))
    monkeypatch.setenv('RPU_HY_VLA_CACHING_ALLOC', '0')
    monkeypatch.setattr(runtime, 'load_hy_vla_weights', lambda *args, **kwargs: object())
    monkeypatch.setattr(runtime, 'HyVlaRunner', FakeLowLevelRunner)
    ckpt = tmp_path / 'model.safetensors'
    ckpt.write_bytes(b'')
    runner = runtime.build_hy_vla(ckpt)
    assert runner._lifecycle_owner is not None
    with pytest.raises(RPUSingleHandleError, match='Restart'):
        runtime.build_hy_vla(ckpt)

def test_facade_owner_is_not_retained_by_the_built_runner(tmp_path, monkeypatch, checkpoint_metadata):
    from rpu_backend.adapters.hy_vla import runtime

    class Owner:
        pass

    class FakeLowLevelRunner:

        def __init__(self, **kwargs):
            self.kwargs = kwargs
    owner = Owner()
    monkeypatch.setattr(os, 'environ', dict(os.environ))
    monkeypatch.setenv('RPU_HY_VLA_CACHING_ALLOC', '0')
    monkeypatch.setattr(runtime, 'load_hy_vla_weights', lambda *args, **kwargs: object())
    monkeypatch.setattr(runtime, 'HyVlaRunner', FakeLowLevelRunner)
    ckpt = tmp_path / 'model.safetensors'
    ckpt.write_bytes(b'')
    runner = runtime.build_hy_vla(ckpt, _owner=owner)
    assert not hasattr(runner, '_lifecycle_owner'), 'runner -> policy would keep weakref.finalize(policy, ..., runner) alive forever'

def test_low_level_missing_checkpoint_fails_before_process_claim(tmp_path):
    from rpu_backend.adapters.hy_vla import runtime
    import rpu_backend.api.causal_lm as lifecycle
    with pytest.raises(FileNotFoundError, match='权重不存在'):
        runtime.build_hy_vla(tmp_path / 'missing.safetensors')
    assert lifecycle._LIVE_REF is None
    assert lifecycle._LIVE_TERMINAL_REASON is None

@pytest.mark.parametrize('failure', ['bind', 'graph_cache'])
def test_sub_open_failure_destroys_native_handle_and_resets_spm(monkeypatch, failure):
    from rpu_backend.adapters.hy_vla import runtime
    events = []

    def bind(handle):
        events.append(('bind', handle))
        if failure == 'bind':
            raise RuntimeError('bind failed')

    def graph_cache():
        events.append(('graph_cache',))
        if failure == 'graph_cache':
            raise RuntimeError('graph cache failed')
        return object()
    monkeypatch.setattr(runtime.rpu_backend.graph, 'GraphCache', graph_cache)
    monkeypatch.setattr(runtime.torch.ops.rpu, 'spm_alloc_reset_all', lambda : events.append(('reset',)), raising=False)
    sub = runtime._Sub('test', lambda : 73, lambda handle: events.append(('destroy', handle)), bind, sig=None)
    with pytest.raises(RuntimeError, match='failed'):
        sub.open()
    assert ('destroy', 73) in events
    assert ('reset',) in events
    assert sub.handle is None
    assert sub.gc is None

@pytest.mark.parametrize('failure', [None, 'graph_clear', 'destroy', 'reset'])
def test_sub_close_stops_at_first_failure_and_never_retries(monkeypatch, failure):
    from rpu_backend.adapters.hy_vla import runtime
    events = []

    class GraphCache:

        def clear(self):
            events.append('clear')
            if failure == 'graph_clear':
                raise RuntimeError('clear failed')

    def destroy(handle):
        events.append(('destroy', handle))
        if failure == 'destroy':
            raise RuntimeError('destroy failed')

    def reset():
        events.append('reset')
        if failure == 'reset':
            raise RuntimeError('reset failed')
    monkeypatch.setattr(runtime.torch.ops.rpu, 'spm_alloc_reset_all', reset, raising=False)
    sub = runtime._Sub('test', lambda : 73, destroy, lambda handle: None, sig=None)
    sub.handle = 73
    cache = sub.gc = GraphCache()
    if failure is None:
        sub.close()
        sub.close()
        assert events == ['clear', ('destroy', 73), 'reset']
        assert sub.handle is None and sub.gc is None
    else:
        with pytest.raises(RuntimeError, match='failed'):
            sub.close()
        expected = {'graph_clear': ['clear'], 'destroy': ['clear', ('destroy', 73)], 'reset': ['clear', ('destroy', 73), 'reset']}[failure]
        assert events == expected
        assert sub.handle == (None if failure == 'reset' else 73) and sub.gc is cache
        for operation in (sub.close, sub.open, sub.__enter__):
            with pytest.raises(RuntimeError, match='restart'):
                operation()
        assert events == expected

def _native_owner(*, persistent=()):
    from rpu_backend.adapters.hy_vla import runtime
    from rpu_backend.api._execution import bind_execution_session
    owner = runtime.HyVlaRunner.__new__(runtime.HyVlaRunner)
    (owner._persist, owner._psubs) = (frozenset(persistent), {})
    owner._initialize_native_execution = lambda *_: None
    session = bind_execution_session(owner, {}, entry_point='HY strict retirement', graph_mode='COMPOSITE_CHILD')
    return (owner, session)

@pytest.mark.parametrize('failure', [None, 'first_clear', 'first_destroy', 'second_clear', 'second_destroy', 'reset'])
def test_bulk_retirement_preserves_unreleased_siblings_and_resets_only_after_all(monkeypatch, failure):
    from rpu_backend.adapters.hy_vla import runtime
    events = []
    (owner, session) = _native_owner(persistent=('vit', 'vlm', 'expert'))
    error = RuntimeError(f'{failure} failed')

    def event(name):
        events.append(name)
        if name == failure:
            raise error
    for (index, name) in enumerate(('vit', 'vlm', 'expert')):
        tag = ('first', 'second', 'third')[index]
        sub = owner._sub(name, None, lambda _handle, tag=tag: event(tag + '_destroy'), None, None)
        sub.handle = index + 10
        sub.gc = types.SimpleNamespace(clear=lambda tag=tag: event(tag + '_clear'))
    children = dict(owner._psubs)
    monkeypatch.setattr(runtime.torch.ops.rpu, 'spm_alloc_reset_all', lambda : event('reset'), raising=False)
    sequence = ['first_clear', 'second_clear', 'third_clear', 'first_destroy', 'second_destroy', 'third_destroy', 'reset']
    if failure is None:
        owner.close()
        owner.close()
        assert events == sequence and owner._psubs == {}
        assert session.stats()['state'] == 'CLOSED'
    else:
        with pytest.raises(RuntimeError) as caught:
            owner.close()
        assert caught.value is error and events == sequence[:sequence.index(failure) + 1]
        assert owner._psubs == children and session.stats()['state'] == 'POISONED'
        assert not owner._closed and owner._retirement_failed
        if failure != 'reset':
            assert children['expert'].handle == 12 and children['expert'].gc is not None
        before = list(events)
        with pytest.raises(RuntimeError, match='shutdown failed'):
            owner.close()
        owner.__del__()
        with pytest.raises(RuntimeError, match='restart'):
            owner._drop_psubs()
        assert events == before

@pytest.mark.parametrize('failure', ['bind', 'destroy', 'reset'])
def test_transient_open_failure_keeps_actual_child_and_original_error(monkeypatch, failure):
    from rpu_backend.adapters.hy_vla import runtime
    (owner, session) = _native_owner()
    events = []
    original = RuntimeError('original bind failed')

    def bind(_handle):
        raise original

    def destroy(handle):
        events.append(('destroy', handle))
        if failure == 'destroy':
            raise RuntimeError('cleanup destroy failed')

    def reset():
        events.append('reset')
        if failure == 'reset':
            raise RuntimeError('cleanup reset failed')
    monkeypatch.setattr(runtime.torch.ops.rpu, 'spm_alloc_reset_all', reset, raising=False)
    sub = owner._sub('vit', lambda : 73, destroy, bind, None)
    assert not owner._psubs
    with pytest.raises(RuntimeError) as caught:
        with sub:
            pytest.fail('failed bind entered its body')
    assert caught.value is original and owner._psubs['vit'] is sub
    assert sub.handle == (73 if failure == 'destroy' else None)
    assert events == ([('destroy', 73)] if failure == 'destroy' else [('destroy', 73), 'reset'])
    assert session.stats()['state'] == 'POISONED'
    before = list(events)
    with pytest.raises(RuntimeError, match='restart'):
        owner.close()
    owner.__del__()
    assert events == before

@pytest.mark.parametrize('persistent', [False, True])
@pytest.mark.parametrize('poison_failure', [False, True])
def test_forward_exception_preserves_graph_handle_and_never_cleans_uncertain_capture(monkeypatch, persistent, poison_failure):
    from rpu_backend.adapters.hy_vla import runtime
    (owner, session) = _native_owner(persistent=('vit',) if persistent else ())
    events = []
    cache = types.SimpleNamespace(clear=lambda : events.append('clear'))
    monkeypatch.setattr(runtime.rpu_backend.graph, 'GraphCache', lambda : cache)
    monkeypatch.setattr(runtime.torch.ops.rpu, 'spm_alloc_reset_all', lambda : events.append('reset'), raising=False)
    if poison_failure:

        def fail_poison():
            raise RuntimeError('poison callback failed')
        monkeypatch.setattr(session, 'poison', fail_poison)
    sub = owner._sub('vit', lambda : 73, lambda _handle: events.append('destroy'), lambda _handle: None, None)
    original = KeyboardInterrupt('capture interrupted')
    with pytest.raises(KeyboardInterrupt) as caught:
        with session.execute():
            with sub:
                raise original
    assert caught.value is original and (not events)
    assert owner._psubs['vit'] is sub and sub.handle == 73 and (sub.gc is cache)
    assert owner._retirement_failed
    if not poison_failure:
        assert session.stats()['state'] == 'POISONED'
        with pytest.raises(RuntimeError, match='poisoned'):
            with session.execute():
                pytest.fail('poisoned session executed')
    with pytest.raises(RuntimeError, match='restart'):
        owner._sub('vlm', None, None, None, None)
    owner.__del__()
    assert not events

@pytest.mark.parametrize('failure', [None, 'vit_clear', 'vlm_clear', 'vit_destroy', 'vlm_destroy', 'reset'])
def test_partial_persist_retires_all_graphs_before_any_handle_and_keeps_ddr(monkeypatch, failure):
    from rpu_backend.adapters.hy_vla import runtime
    (owner, session) = _native_owner(persistent=('vit',))
    events = []
    original = RuntimeError('partial retirement failed')

    def event(name):
        events.append(name)
        if failure == name:
            raise original
    ddr = owner._cache = types.SimpleNamespace(k_caches=[object()], v_caches=[object()])
    weights = owner.w = object()
    vit = owner._sub('vit', None, lambda _handle: event('vit_destroy'), None, None)
    (vit.handle, vit.gc) = (72, types.SimpleNamespace(clear=lambda : event('vit_clear')))
    vlm = owner._sub('vlm', None, lambda _handle: event('vlm_destroy'), None, None)
    (vlm.handle, vlm.gc) = (73, types.SimpleNamespace(clear=lambda : event('vlm_clear')))
    monkeypatch.setattr(runtime.torch.ops.rpu, 'spm_alloc_reset_all', lambda : event('reset'), raising=False)
    sequence = ['vit_clear', 'vlm_clear', 'vit_destroy', 'vlm_destroy', 'reset']
    if failure:
        with pytest.raises(RuntimeError) as caught:
            vlm.__exit__(None, None, None)
        assert caught.value is original and events == sequence[:sequence.index(failure) + 1]
        assert owner._psubs == {'vit': vit, 'vlm': vlm}
        assert session.stats()['state'] == 'POISONED'
        if failure.endswith('clear'):
            assert (vit.handle, vlm.handle) == (72, 73)
        elif failure == 'vit_destroy':
            assert (vit.handle, vlm.handle) == (72, 73)
        elif failure == 'vlm_destroy':
            assert (vit.handle, vlm.handle) == (None, 73)
    else:
        vlm.__exit__(None, None, None)
        assert events == sequence and owner._psubs == {}
        assert vit.handle is None and vlm.handle is None
        assert session.stats()['state'] == 'QUIESCENT'
        replacement = owner._sub('vit', None, None, None, None)
        assert replacement is not vit and replacement.persist and (replacement.handle is None)
    assert owner._cache is ddr and owner.w is weights
    before = list(events)
    owner.__del__()
    assert events == before

def test_bind_failure_with_live_sibling_preserves_original_when_bulk_cleanup_fails(monkeypatch):
    from rpu_backend.adapters.hy_vla import runtime
    (owner, session) = _native_owner(persistent=('vit',))
    vit = owner._sub('vit', None, None, None, None)
    events = []

    def clear():
        events.append('clear')
        raise RuntimeError('sibling clear failed')
    (vit.handle, vit.gc) = (72, types.SimpleNamespace(clear=clear))
    original = RuntimeError('bind interrupted with another child alive')

    def bind(_handle):
        raise original
    vlm = owner._sub('vlm', lambda : 73, lambda _handle: events.append('destroy'), bind, None)
    monkeypatch.setattr(runtime.torch.ops.rpu, 'spm_alloc_reset_all', lambda : events.append('reset'), raising=False)
    with pytest.raises(RuntimeError) as caught:
        vlm.open()
    assert caught.value is original and events == ['clear']
    assert owner._psubs == {'vit': vit, 'vlm': vlm} and (vit.handle, vlm.handle) == (72, 73)
    assert session.stats()['state'] == 'POISONED'
    owner.__del__()
    assert events == ['clear']

@pytest.mark.parametrize('failure', [False, True])
def test_constant_replacement_waits_for_actual_child_retirement(monkeypatch, failure):
    from rpu_backend.adapters.hy_vla import runtime
    (owner, session) = _native_owner(persistent=('vit',))
    events = []
    (old_key, old_value) = ([torch.zeros(1)], object())
    (owner._const_key, owner._const_val) = ({'mask': old_key}, {'mask': old_value})
    child = owner._sub('vit', None, lambda _handle: events.append('destroy'), None, None)
    child.handle = 72

    def clear():
        events.append('clear')
        if failure:
            raise RuntimeError('clear failed before replacing constants')
    child.gc = types.SimpleNamespace(clear=clear)
    monkeypatch.setattr(runtime.torch.ops.rpu, 'spm_alloc_reset_all', lambda : events.append('reset'), raising=False)
    value = object()

    def build():
        events.append('build')
        return value
    if failure:
        with pytest.raises(RuntimeError, match='clear failed'):
            with session.execute():
                owner._cached('mask', [torch.ones(1)], build)
        assert events == ['clear'] and owner._psubs == {'vit': child}
        assert owner._const_key['mask'] is old_key and owner._const_val['mask'] is old_value
        assert session.stats()['state'] == 'POISONED'
    else:
        with session.execute():
            assert owner._cached('mask', [torch.ones(1)], build) is value
        assert events == ['clear', 'destroy', 'reset', 'build'] and (not owner._psubs)
        assert torch.equal(owner._const_key['mask'][0], torch.ones(1))
        assert owner._const_val['mask'] is value
        assert session.stats()['state'] == 'QUIESCENT'

@pytest.mark.parametrize('failure', [False, True])
def test_expert_variant_replacement_retires_all_siblings_before_rebuilding(monkeypatch, failure):
    from rpu_backend.adapters.hy_vla import runtime
    (owner, session) = _native_owner(persistent=('vit', 'vlm', 'expert'))
    events = []
    for (name, handle) in (('vit', 72), ('vlm', 73), ('expert', 74)):
        child = owner._sub(name, None, lambda handle: events.append(('destroy', handle)), None, None)

        def clear(handle=handle):
            events.append(('clear', handle))
            if failure and handle == 73:
                raise RuntimeError('sibling clear failed')
        (child.handle, child.gc) = (handle, types.SimpleNamespace(clear=clear))
    children = dict(owner._psubs)
    monkeypatch.setattr(runtime.torch.ops.rpu, 'spm_alloc_reset_all', lambda : events.append('reset'), raising=False)
    if failure:
        with pytest.raises(RuntimeError, match='sibling clear failed'):
            owner._sub('expert', lambda : events.append('create'), None, None, None, variant=True)
        assert events == [('clear', 72), ('clear', 73)]
        assert owner._psubs == children and children['expert'].handle == 74
        assert session.stats()['state'] == 'POISONED'
    else:
        new = owner._sub('expert', lambda : events.append('create'), None, None, None, variant=True)
        assert events == [('clear', 72), ('clear', 73), ('clear', 74), ('destroy', 72), ('destroy', 73), ('destroy', 74), 'reset']
        assert owner._psubs == {'expert': new} and new is not children['expert']
        assert new.handle is None and new.variant is True
        assert session.stats()['state'] == 'QUIESCENT'

@pytest.mark.parametrize('persistent', [(), ('vit',), ('vlm',), ('expert',), ('vit', 'vlm'), ('vit', 'expert'), ('vlm', 'expert'), ('vit', 'vlm', 'expert')])
def test_all_original_persistence_subsets_keep_running_without_resetting_live_graphs(monkeypatch, persistent):
    from rpu_backend.adapters.hy_vla import runtime
    (owner, session) = _native_owner(persistent=persistent)
    (live, created, resets) = ({}, [], [])
    ddr = owner._cache = object()
    weights = owner.w = object()

    def create():
        handle = len(created) + 1
        created.append(handle)
        live[handle] = True
        return handle

    def graph_cache():
        handle = created[-1]
        return types.SimpleNamespace(clear=lambda : live.__setitem__(handle, False))

    def destroy(handle):
        assert not any(live.values()), "destroy preceded a sibling Graph's retirement"
        del live[handle]

    def reset():
        assert not live, 'reset changed an address still owned by another handle'
        resets.append(len(created))
    monkeypatch.setattr(runtime.rpu_backend.graph, 'GraphCache', graph_cache)
    monkeypatch.setattr(runtime.torch.ops.rpu, 'spm_alloc_reset_all', reset, raising=False)
    for _frame in range(3):
        with session.execute():
            for name in ('vit', 'vlm', 'expert'):
                with owner._sub(name, create, destroy, lambda _handle: None, None) as sub:
                    assert sub.handle in live and live[sub.handle]
        assert owner._cache is ddr and owner.w is weights
    owner.close()
    assert not live and resets and (session.stats()['state'] == 'CLOSED')
    if len(persistent) == 3:
        assert len(created) == 3 and resets == [3]
    elif not persistent:
        assert len(created) == 9 and len(resets) == 9

def test_closed_runner_cannot_recreate_handles_or_infer():
    from rpu_backend.adapters.hy_vla import runtime
    runner = runtime.HyVlaRunner.__new__(runtime.HyVlaRunner)
    runner._closed = True
    with pytest.raises(RuntimeError, match='已 close'):
        runner._sub('vit', None, None, None, None)
    with pytest.raises(RuntimeError, match='已 close'):
        runner.get_action([], None, None, None, None, None, None, None, state=None)
