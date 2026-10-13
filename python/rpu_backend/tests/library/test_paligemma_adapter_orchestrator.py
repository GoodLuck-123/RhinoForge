import ast
from pathlib import Path
from types import MethodType, SimpleNamespace
import pytest
import torch
from rpu_backend.api.errors import RPUBackendError, RPUUnsupportedDtypeError
_REPO_ROOT = Path(__file__).resolve().parents[4]
_GEMMA2_PATH = _REPO_ROOT / 'python/rpu_backend/adapters/paligemma/gemma2.py'
_GEMMA2_CPP_PATH = _REPO_ROOT / 'src/fused/rpu_gemma2_model.cpp'

def _module_source() -> str:
    return _GEMMA2_PATH.read_text()

def _module_tree() -> ast.Module:
    return ast.parse(_module_source(), filename=str(_GEMMA2_PATH))

def _first_rpu_op_lineno(fn: ast.FunctionDef, op_name: str) -> int | None:
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != op_name:
            continue
        rpu = func.value
        if not isinstance(rpu, ast.Attribute) or rpu.attr != 'rpu':
            continue
        ops = rpu.value
        if not isinstance(ops, ast.Attribute) or ops.attr != 'ops':
            continue
        torch_name = ops.value
        if isinstance(torch_name, ast.Name) and torch_name.id == 'torch':
            return node.lineno
    return None

def _top_level_fn(name: str) -> ast.FunctionDef:
    fn = next((node for node in ast.iter_child_nodes(_module_tree()) if isinstance(node, ast.FunctionDef) and node.name == name), None)
    assert fn is not None
    return fn

def test_gemma2_patch_function_exists():
    assert _top_level_fn('patch_gemma2_model_for_rpu_all_layers_once')

def test_layer_type_ids_maps_profile_pattern():
    namespace: dict[str, object] = {}
    fn = _top_level_fn('_layer_type_ids')
    module = ast.Module(body=[fn], type_ignores=[])
    exec(compile(module, str(_GEMMA2_PATH), 'exec'), namespace)
    layer_type_ids = namespace['_layer_type_ids']
    assert layer_type_ids(['sliding_attention', 'full_attention']) == [0, 1]
    with pytest.raises(ValueError, match='unknown Gemma2 layer type'):
        layer_type_ids(['sliding_attention', 'local_attention'])

def test_gemma2_patch_body_calls_fused_forward_and_spm_reset():
    patch_fn = _top_level_fn('patch_gemma2_model_for_rpu_all_layers_once')
    source = ast.unparse(patch_fn)
    assert 'gemma2_forward' in source
    assert 'spm_alloc_reset_temporary' in source
    nested = [node for node in ast.walk(patch_fn) if isinstance(node, ast.FunctionDef) and node is not patch_fn and node.name.startswith('rpu_') and node.name.endswith('forward')]
    assert nested
    for inner in nested:
        fused_lineno = _first_rpu_op_lineno(inner, 'gemma2_forward')
        reset_lineno = _first_rpu_op_lineno(inner, 'spm_alloc_reset_temporary')
        assert fused_lineno is not None
        assert reset_lineno is not None
        assert reset_lineno > fused_lineno

def test_gemma2_forward_uses_explicit_cache_layout_and_updates_position():
    patch_source = ast.unparse(_top_level_fn('patch_gemma2_model_for_rpu_all_layers_once'))
    assert 'RPUCache.from_model' not in patch_source
    assert 'RPUCache(' in patch_source
    assert '_rpu_attn_tp' in patch_source
    assert 'attn_tp=attn_tp' in patch_source
    assert '_rpu_max_seq_len' in patch_source
    assert 'past_key_values.position' in patch_source
    assert 'update_position(seq_len)' in patch_source

def test_gemma2_fresh_call_resets_existing_cache_unconditionally():
    patch_source = ast.unparse(_top_level_fn('patch_gemma2_model_for_rpu_all_layers_once'))
    assert 'self._rpu_cache.reset_to_position(0)' in patch_source
    assert 'if use_cache:\n                    self._rpu_cache.reset_to_position(0)' not in patch_source

def test_gemma2_validates_external_cache_layout():
    helper_source = ast.unparse(_top_level_fn('_validate_gemma2_cache'))
    patch_source = ast.unparse(_top_level_fn('patch_gemma2_model_for_rpu_all_layers_once'))
    assert 'NUM_CORES' in _module_source()
    assert 'attn_tp' in helper_source
    assert 'nKVHeadChunk' in helper_source
    assert 'nKVHeadVx' in helper_source
    assert 'Gemma2 device cache layout' in helper_source
    assert '_validate_gemma2_cache(' in patch_source

def test_gemma2_attention_mask_helper_rejects_unselected_masks():
    helper_source = ast.unparse(_top_level_fn('_select_gemma2_attention_mask'))
    patch_source = ast.unparse(_top_level_fn('patch_gemma2_model_for_rpu_all_layers_once'))
    assert 'full_attention' in helper_source
    assert 'sliding_attention' in helper_source
    assert 'Gemma2 device requires additive attention mask' in helper_source
    assert 'selected_attention_mask = _select_gemma2_attention_mask' in patch_source
    assert 'selected_attention_mask' in patch_source

def test_gemma2_attention_mask_helper_rejects_positive_padding_masks():
    helper_source = ast.unparse(_top_level_fn('_select_gemma2_attention_mask'))
    assert 'selected > 0' in helper_source
    assert 'Batch padding masks' in helper_source
    assert 'must be converted' in helper_source

def test_gemma2_cpp_keeps_zero_index_cache_and_one_index_rope_contract():
    source = _GEMMA2_CPP_PATH.read_text()
    assert 'cache_start is the 0-indexed cache slot' in source
    assert "rope_start is PaliGemma's 1-indexed RoPE position" in source
    assert 'int64_t rope_start = cache_start + 1' in source

def test_gemma2_handle_is_published_after_set_weights():
    configure_fn = _top_level_fn('_configure_gemma2_handle')
    patch_fn = _top_level_fn('patch_gemma2_model_for_rpu_all_layers_once')
    set_weights_lineno = _first_rpu_op_lineno(configure_fn, 'gemma2_set_weights')
    create_lineno = _first_rpu_op_lineno(configure_fn, 'gemma2_create')
    configure_lineno = None
    publish_lineno = None
    finalize_lineno = None
    for node in ast.walk(patch_fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and (node.func.id == '_configure_gemma2_handle'):
            configure_lineno = node.lineno
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == '_rpu_gemma2_handle' and isinstance(target.value, ast.Name) and (target.value.id == 'model'):
                    publish_lineno = node.lineno
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and (node.func.attr == 'finalize'):
            finalize_lineno = node.lineno
    assert create_lineno is not None
    assert set_weights_lineno is not None
    assert create_lineno < set_weights_lineno
    assert configure_lineno is not None
    assert publish_lineno is not None
    assert publish_lineno > configure_lineno
    assert finalize_lineno is not None
    assert configure_lineno < finalize_lineno < publish_lineno

class _FakeModule:

    def __init__(self, *, dtype: torch.dtype=torch.float16) -> None:
        self._rpu_swizzled = False
        self.config = SimpleNamespace(num_key_value_heads=4, num_attention_heads=8, head_dim=256, hidden_size=2304, intermediate_size=9216, rms_norm_eps=1e-06, query_pre_attn_scalar=256, attn_logit_softcapping=50.0, layer_types=['sliding_attention', 'full_attention'] * 13)
        self._param = torch.empty((), dtype=dtype)
        self._buffer = torch.empty((), dtype=torch.float16)
        self.last_device = None

    def named_parameters(self):
        return [('weight', self._param)]

    def named_buffers(self):
        return [('buffer', self._buffer)]

    def to(self, device):
        self.last_device = device
        return self

def _fake_hf_model(*, dtype: torch.dtype=torch.float16):
    vision = _FakeModule(dtype=dtype)
    vision.vision_model = _FakeModule(dtype=dtype)
    vision.vision_model.encoder = _FakeModule(dtype=dtype)
    vision.vision_model.post_layernorm = _FakeModule(dtype=dtype)
    projector = _FakeModule(dtype=dtype)
    decoder = _FakeModule(dtype=dtype)
    decoder.embed_tokens = _FakeModule(dtype=dtype)
    decoder.layers = []
    language_model = SimpleNamespace(model=decoder, lm_head=_FakeModule(dtype=dtype))
    root = SimpleNamespace(_rpu_swizzled=False, vision_tower=vision, multi_modal_projector=projector, language_model=language_model)
    root._param = torch.empty((), dtype=dtype)
    root._buffer = torch.empty((), dtype=torch.float16)
    root.named_parameters = lambda : [('root.weight', root._param)]
    root.named_buffers = lambda : [('root.buffer', root._buffer)]
    return root

class _Finalizer:

    def __init__(self, events=None, label='handle', *, alive=True):
        self._events = events
        self._label = label
        self.alive = alive

    def __call__(self):
        if self.alive:
            if self._events is not None:
                self._events.append(('finalize', self._label))
            self.alive = False
        return True

class _GraphCache:

    def __init__(self, events=None):
        self._events = events

    def clear(self):
        if self._events is not None:
            self._events.append(('clear', 'siglip_graph'))

def _publish_siglip_runtime(vision, *, handle=1, events=None):

    def rpu_siglip_forward(self):
        return self
    vision._rpu_cache = object()
    vision._rpu_siglip_graph_cache = _GraphCache(events)
    vision._rpu_lazy_init_checked = True
    vision._rpu_required_attrs = ('_rpu_cache', '_rpu_vision_handle', '_rpu_siglip_graph_cache')
    vision._rpu_vision_handle = handle
    vision._rpu_vision_handle_finalizer = _Finalizer(events, 'siglip', alive=True)
    vision.forward = MethodType(rpu_siglip_forward, vision)
    return handle

def _publish_gemma2_runtime(decoder, *, handle=2, events=None):

    def rpu_gemma2_model_forward(self):
        return self
    decoder._rpu_effective_num_kv_heads = 4
    decoder._rpu_gemma2_handle = handle
    decoder._rpu_gemma2_handle_finalizer = _Finalizer(events, 'gemma2', alive=True)
    decoder._rpu_cache = object()
    decoder._rpu_lazy_init_checked = True
    decoder._rpu_required_attrs = ('_rpu_cache', '_rpu_gemma2_handle', '_rpu_gemma2_handle_finalizer')
    decoder._rpu_gemma2_had_instance_forward = False
    decoder._rpu_gemma2_original_instance_forward = None
    decoder._rpu_gemma2_install_started = True
    decoder._rpu_gemma2_install_ready = True
    decoder._rpu_attn_tp = 4
    decoder._rpu_max_seq_len = 4096
    decoder.forward = MethodType(rpu_gemma2_model_forward, decoder)
    return handle

def _publish_complete_submodel_runtime(model, *, events=None):
    vision = model.vision_tower.vision_model
    decoder = model.language_model.model
    _publish_siglip_runtime(vision, events=events)
    _publish_gemma2_runtime(decoder, events=events)

def test_paligemma_components_resolves_existing_root_fake_layout():
    from rpu_backend.adapters.paligemma import _paligemma_components
    model = _fake_hf_model()
    (base_model, vision, projector, decoder, lm_head) = _paligemma_components(model)
    assert base_model is model
    assert vision is model.vision_tower
    assert projector is model.multi_modal_projector
    assert decoder is model.language_model.model
    assert lm_head is model.language_model.lm_head

def test_paligemma_components_resolves_real_hf_nested_layout():
    from rpu_backend.adapters.paligemma import _paligemma_components
    base_model = SimpleNamespace(vision_tower=object(), multi_modal_projector=object(), language_model=object())
    hf_model = SimpleNamespace(model=base_model, lm_head=object())
    (resolved_base, vision, projector, decoder, lm_head) = _paligemma_components(hf_model)
    assert resolved_base is base_model
    assert vision is base_model.vision_tower
    assert projector is base_model.multi_modal_projector
    assert decoder is base_model.language_model
    assert lm_head is hf_model.lm_head

def test_adapter_repeated_to_rpu_is_idempotent(monkeypatch):
    import rpu_backend.adapters.paligemma as paligemma
    calls = []
    monkeypatch.setattr(paligemma, '_claim_live_instance', lambda model: calls.append('claim') or id(model))
    monkeypatch.setattr(paligemma, 'validate_preinstall', lambda model: calls.append('pre'))
    monkeypatch.setattr(paligemma, 'validate_postinstall', lambda model: calls.append('post'))
    monkeypatch.setattr(paligemma, 'install_hw_attr_validator', lambda model: calls.append('validator'))
    monkeypatch.setattr(paligemma, 'patch_siglip_model_for_rpu_all_layers_once', lambda vision, projector: calls.append('siglip') or _publish_siglip_runtime(vision))
    monkeypatch.setattr(paligemma, 'patch_gemma2_model_for_rpu_all_layers_once', lambda model: calls.append('gemma2') or _publish_gemma2_runtime(model))
    model = _fake_hf_model()
    adapter = paligemma.PaliGemmaAdapter(model)
    assert adapter.to_rpu() is model
    assert adapter.to_rpu() is model
    assert calls.count('claim') == 1
    assert calls.count('siglip') == 1
    assert calls.count('gemma2') == 1
    assert model._rpu_swizzled is True
    assert adapter._rpu_is_ready is True
    assert model.vision_tower.vision_model.encoder.last_device == 'rpu'
    assert model.vision_tower.vision_model.post_layernorm.last_device == 'rpu'
    assert model.multi_modal_projector.last_device == 'rpu'
    assert model.language_model.model.last_device == 'rpu'
    assert model.language_model.model.embed_tokens.last_device == 'cpu'
    assert model.language_model.model._rpu_attn_tp == 4
    assert model.language_model.model._rpu_max_seq_len == 4096
    monkeypatch.setattr(paligemma, '_run_spm_contract_check', lambda : {'returncode': 0, 'stdout': 'cpu fixture', 'stderr': ''})
    probe = adapter.run_runtime_probe({})
    assert probe['gemma2_sentinel_set'] is True
    assert probe['second_to_rpu_no_double_swizzle'] is True
    assert calls.count('gemma2') == 1

def test_second_adapter_rehydrates_only_complete_live_runtime(monkeypatch):
    import rpu_backend.adapters.paligemma as paligemma
    model = _fake_hf_model()
    model._rpu_swizzle_started = True
    _publish_complete_submodel_runtime(model)
    model._rpu_swizzled = True
    calls = []
    monkeypatch.setattr(paligemma, 'validate_preinstall', lambda owner: calls.append('pre'))
    monkeypatch.setattr(paligemma, '_claim_live_instance', lambda owner: calls.append('claim'))
    adapter = paligemma.PaliGemmaAdapter(model)
    assert adapter._rpu_is_ready is True
    assert adapter._siglip_handle == 1
    assert adapter._gemma2_handle == 2
    assert adapter.to_rpu() is model
    assert calls == []

@pytest.mark.parametrize('break_runtime', ['marker_only', 'dead_siglip', 'missing_decoder_cache', 'wrong_forward', 'missing_tp', 'boolean_tp', 'zero_tp'])
def test_adapter_rejects_incomplete_ready_marker_before_preflight(monkeypatch, break_runtime):
    import rpu_backend.adapters.paligemma as paligemma
    model = _fake_hf_model()
    model._rpu_swizzle_started = True
    if break_runtime != 'marker_only':
        _publish_complete_submodel_runtime(model)
        if break_runtime == 'dead_siglip':
            model.vision_tower.vision_model._rpu_vision_handle_finalizer.alive = False
        elif break_runtime == 'missing_decoder_cache':
            del model.language_model.model._rpu_cache
        elif break_runtime == 'missing_tp':
            del model.language_model.model._rpu_attn_tp
        elif break_runtime in {'boolean_tp', 'zero_tp'}:
            model.language_model.model._rpu_attn_tp = True if break_runtime == 'boolean_tp' else 0
        else:
            model.language_model.model.forward = lambda : None
    model._rpu_swizzled = True
    calls = []
    monkeypatch.setattr(paligemma, 'validate_preinstall', lambda owner: calls.append('pre'))
    monkeypatch.setattr(paligemma, '_claim_live_instance', lambda owner: calls.append('claim'))
    adapter = paligemma.PaliGemmaAdapter(model)
    assert adapter._rpu_is_ready is False
    with pytest.raises(RPUBackendError, match='runtime is incomplete'):
        adapter.to_rpu()
    assert calls == []
    assert adapter._siglip_handle is None
    assert adapter._gemma2_handle is None

def test_adapter_rejects_retry_after_partial_failure():
    import rpu_backend.adapters.paligemma as paligemma
    model = _fake_hf_model()
    model._rpu_swizzle_started = True
    model._rpu_swizzled = False
    adapter = paligemma.PaliGemmaAdapter(model)
    with pytest.raises(RPUBackendError, match='prior swizzle attempt failed'):
        adapter.to_rpu()

def test_adapter_rejects_non_fp16_floating_params(monkeypatch):
    import rpu_backend.adapters.paligemma as paligemma
    calls = []
    model = _fake_hf_model(dtype=torch.float32)
    adapter = paligemma.PaliGemmaAdapter(model)
    monkeypatch.setattr(paligemma, '_claim_live_instance', lambda model: calls.append('claim') or id(model))
    monkeypatch.setattr(paligemma, 'validate_preinstall', lambda model: None)
    with pytest.raises(RPUUnsupportedDtypeError, match='torch.float16'):
        adapter.to_rpu()
    assert calls == []
    assert getattr(model, '_rpu_swizzle_started', False) is False
    assert model._rpu_swizzled is False

def test_adapter_siglip_patch_failure_cleans_published_handle(monkeypatch):
    import rpu_backend.adapters.paligemma as paligemma
    calls = []
    model = _fake_hf_model()
    adapter = paligemma.PaliGemmaAdapter(model)
    vision_inner = model.vision_tower.vision_model

    def fail_siglip(vision, projector):
        vision._rpu_vision_handle = 11
        raise RuntimeError('siglip failed')
    monkeypatch.setattr(paligemma, '_claim_live_instance', lambda model: calls.append('claim'))
    monkeypatch.setattr(paligemma, 'validate_preinstall', lambda model: None)
    monkeypatch.setattr(paligemma.torch.ops.rpu, 'siglip_destroy', lambda handle: calls.append(('siglip_destroy', handle)), raising=False)
    monkeypatch.setattr(paligemma, 'patch_siglip_model_for_rpu_all_layers_once', fail_siglip)
    with pytest.raises(RuntimeError, match='siglip failed'):
        adapter.to_rpu()
    assert calls == ['claim', ('siglip_destroy', 11)]
    assert not hasattr(vision_inner, '_rpu_vision_handle')
    assert adapter._siglip_handle is None
    assert adapter._gemma2_handle is None
    assert model._rpu_swizzle_started is True
    assert model._rpu_swizzled is False

def test_adapter_base_exception_still_cleans_and_poison_model(monkeypatch):
    import rpu_backend.adapters.paligemma as paligemma

    class InstallInterrupted(BaseException):
        pass
    calls = []
    model = _fake_hf_model()
    adapter = paligemma.PaliGemmaAdapter(model)
    vision_inner = model.vision_tower.vision_model

    def fail_siglip(vision, projector):
        vision._rpu_vision_handle = 12
        raise InstallInterrupted('synthetic interruption')
    monkeypatch.setattr(paligemma, '_claim_live_instance', lambda owner: calls.append('claim'))
    monkeypatch.setattr(paligemma, 'validate_preinstall', lambda owner: None)
    monkeypatch.setattr(paligemma.torch.ops.rpu, 'siglip_destroy', lambda handle: calls.append(('siglip_destroy', handle)), raising=False)
    monkeypatch.setattr(paligemma, 'patch_siglip_model_for_rpu_all_layers_once', fail_siglip)
    with pytest.raises(InstallInterrupted, match='synthetic interruption'):
        adapter.to_rpu()
    assert calls == ['claim', ('siglip_destroy', 12)]
    assert not hasattr(vision_inner, '_rpu_vision_handle')
    assert model._rpu_swizzle_started is True
    assert model._rpu_swizzled is False

def test_adapter_gemma2_patch_failure_cleans_siglip_handle(monkeypatch):
    import rpu_backend.adapters.paligemma as paligemma
    calls = []
    model = _fake_hf_model()
    adapter = paligemma.PaliGemmaAdapter(model)
    vision_inner = model.vision_tower.vision_model

    def patch_siglip(vision, projector):
        vision._rpu_vision_handle = 21
        return 21

    def fail_gemma2(decoder):
        decoder._rpu_gemma2_handle = 22
        raise RuntimeError('gemma2 failed')
    monkeypatch.setattr(paligemma, '_claim_live_instance', lambda model: calls.append('claim'))
    monkeypatch.setattr(paligemma, 'validate_preinstall', lambda model: None)
    monkeypatch.setattr(paligemma.torch.ops.rpu, 'siglip_destroy', lambda handle: calls.append(('siglip_destroy', handle)), raising=False)
    monkeypatch.setattr(paligemma.torch.ops.rpu, 'gemma2_destroy', lambda handle: calls.append(('gemma2_destroy', handle)), raising=False)
    monkeypatch.setattr(paligemma, 'patch_siglip_model_for_rpu_all_layers_once', patch_siglip)
    monkeypatch.setattr(paligemma, 'patch_gemma2_model_for_rpu_all_layers_once', fail_gemma2)
    with pytest.raises(RuntimeError, match='gemma2 failed'):
        adapter.to_rpu()
    assert calls == ['claim', ('siglip_destroy', 21), ('gemma2_destroy', 22)]
    assert not hasattr(vision_inner, '_rpu_vision_handle')
    assert not hasattr(model.language_model.model, '_rpu_gemma2_handle')
    assert adapter._siglip_handle is None
    assert adapter._gemma2_handle is None
    assert model._rpu_swizzle_started is True
    assert model._rpu_swizzled is False
    with pytest.raises(RPUBackendError, match='prior swizzle attempt failed'):
        adapter.to_rpu()

@pytest.mark.parametrize('failure_site', ['postinstall', 'validator'])
def test_adapter_post_patch_failure_cleans_both_handles(monkeypatch, failure_site):
    import rpu_backend.adapters.paligemma as paligemma
    calls = []
    model = _fake_hf_model()
    adapter = paligemma.PaliGemmaAdapter(model)
    vision_inner = model.vision_tower.vision_model
    decoder = model.language_model.model

    def patch_siglip(vision, projector):
        vision._rpu_vision_handle = 31
        return 31

    def patch_gemma2(model):
        model._rpu_gemma2_handle = 32
        return 32

    def maybe_fail_postinstall(model):
        if failure_site == 'postinstall':
            raise RuntimeError('postinstall failed')

    def maybe_fail_validator(model):
        if failure_site == 'validator':
            raise RuntimeError('validator failed')
    monkeypatch.setattr(paligemma, '_claim_live_instance', lambda model: calls.append('claim'))
    monkeypatch.setattr(paligemma, 'validate_preinstall', lambda model: None)
    monkeypatch.setattr(paligemma, 'validate_postinstall', maybe_fail_postinstall)
    monkeypatch.setattr(paligemma, 'install_hw_attr_validator', maybe_fail_validator)
    monkeypatch.setattr(paligemma.torch.ops.rpu, 'siglip_destroy', lambda handle: calls.append(('siglip_destroy', handle)), raising=False)
    monkeypatch.setattr(paligemma.torch.ops.rpu, 'gemma2_destroy', lambda handle: calls.append(('gemma2_destroy', handle)), raising=False)
    monkeypatch.setattr(paligemma, 'patch_siglip_model_for_rpu_all_layers_once', patch_siglip)
    monkeypatch.setattr(paligemma, 'patch_gemma2_model_for_rpu_all_layers_once', patch_gemma2)
    with pytest.raises(RuntimeError, match=f'{failure_site} failed'):
        adapter.to_rpu()
    assert calls == ['claim', ('siglip_destroy', 31), ('gemma2_destroy', 32)]
    assert not hasattr(vision_inner, '_rpu_vision_handle')
    assert not hasattr(decoder, '_rpu_gemma2_handle')
    assert not hasattr(decoder, '_rpu_attn_tp')
    assert adapter._siglip_handle is None
    assert adapter._gemma2_handle is None
    assert model._rpu_swizzle_started is True
    assert model._rpu_swizzled is False

def test_paligemma_failure_path_does_not_rollback_swizzled_after_validator():
    source = (_REPO_ROOT / 'python/rpu_backend/adapters/paligemma/__init__.py').read_text()
    assert 'self._hf_model._rpu_swizzled = False' not in source

def test_paligemma_success_marks_swizzled_after_validator():
    source = (_REPO_ROOT / 'python/rpu_backend/adapters/paligemma/__init__.py').read_text()
    validator_idx = source.index('install_hw_attr_validator(self._hf_model)')
    sync_idx = source.index('_sync_adapter_runtime_state(self, component_state)', validator_idx)
    swizzled_idx = source.index('self._hf_model._rpu_swizzled = True')
    assert validator_idx < sync_idx < swizzled_idx

def test_paligemma_downgrade_removes_unreachable_kv_replication_experiment():
    source = _module_source()
    assert '_kv_replication_factor' not in source
    assert '_replicate_kv_proj_weights' not in source
    assert 'kv_replication' not in source
    assert 'effective_num_kv_heads = cfg_num_kv_heads' in source

def test_ready_marker_publish_base_exception_cleans_components(monkeypatch):
    import rpu_backend.adapters.paligemma as paligemma

    class InstallInterrupted(BaseException):
        pass

    class FailingReadyModel(SimpleNamespace):

        def __setattr__(self, name, value):
            super().__setattr__(name, value)
            if name == '_rpu_swizzled' and value is True and getattr(self, '_fail_ready_publish', False):
                raise InstallInterrupted('ready publication interrupted')
    base = _fake_hf_model()
    model = FailingReadyModel(**vars(base))
    events = []
    adapter = paligemma.PaliGemmaAdapter(model)
    monkeypatch.setattr(paligemma, '_claim_live_instance', lambda owner: None)
    monkeypatch.setattr(paligemma, 'validate_preinstall', lambda owner: None)
    monkeypatch.setattr(paligemma, 'validate_postinstall', lambda owner: None)
    monkeypatch.setattr(paligemma, 'install_hw_attr_validator', lambda owner: None)
    monkeypatch.setattr(paligemma, 'patch_siglip_model_for_rpu_all_layers_once', lambda vision, projector: _publish_siglip_runtime(vision, events=events))
    monkeypatch.setattr(paligemma, 'patch_gemma2_model_for_rpu_all_layers_once', lambda decoder: _publish_gemma2_runtime(decoder, events=events))
    model._fail_ready_publish = True
    with pytest.raises(InstallInterrupted, match='ready publication interrupted'):
        adapter.to_rpu()
    vision = model.vision_tower.vision_model
    decoder = model.language_model.model
    assert events == [('clear', 'siglip_graph'), ('finalize', 'siglip'), ('finalize', 'gemma2')]
    assert 'forward' not in vars(vision)
    assert 'forward' not in vars(decoder)
    assert not hasattr(vision, '_rpu_vision_handle')
    assert not hasattr(vision, '_rpu_siglip_graph_cache')
    assert not hasattr(decoder, '_rpu_gemma2_handle')
    assert adapter._rpu_is_ready is False
    assert adapter._siglip_handle is None
    assert adapter._gemma2_handle is None
    assert model._rpu_swizzle_started is True
    assert model._rpu_swizzled is True
    with pytest.raises(RPUBackendError, match='runtime is incomplete'):
        adapter.to_rpu()
