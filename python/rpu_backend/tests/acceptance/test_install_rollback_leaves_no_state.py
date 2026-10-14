from __future__ import annotations
from types import SimpleNamespace
import pytest
import torch
import rpu_backend
from rpu_backend.adapters import pi05 as pi05_adapter
from rpu_backend.adapters.gr00t import runtime as gr00t_runtime
from rpu_backend.adapters.pi05 import adarms, gemma, w8a16
from rpu_backend.adapters.qwen3_vl import vision as qwen3vl_vision

class _FakeTensor:
    dtype = torch.float16

    def __init__(self, *shape):
        self.shape = shape

    @property
    def data(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def dim(self):
        return len(self.shape)

    def unsqueeze(self, dim):
        return _FakeTensor(1, *self.shape)

class _Movable:

    def to(self, *args, **kwargs):
        return self

class _Rotary:
    max_position_embeddings = 32

    def __call__(self, hidden, positions):
        return (_FakeTensor(32, 8), _FakeTensor(32, 8))

class _GraphCache:

    def clear(self):
        pass

    def cache_invariant_ok(self):
        return True

class _Module:

    def forward(self, *args, **kwargs):
        return 'original'

    def to(self, *args, **kwargs):
        return self

def _linear(out_features, in_features):
    return SimpleNamespace(weight=_FakeTensor(out_features, in_features), out_features=out_features)

def _decoder_module():
    (hidden, intermediate) = (128, 256)
    layer = SimpleNamespace(self_attn=SimpleNamespace(q_proj=_linear(hidden, hidden), k_proj=_linear(64, hidden), v_proj=_linear(64, hidden), o_proj=_linear(hidden, hidden)), mlp=SimpleNamespace(gate_proj=_linear(intermediate, hidden), up_proj=_linear(intermediate, hidden), down_proj=_linear(hidden, intermediate)), input_layernorm=SimpleNamespace(weight=_FakeTensor(hidden)), post_attention_layernorm=SimpleNamespace(weight=_FakeTensor(hidden)))
    module = _Module()
    module.layers = [layer]
    module.config = SimpleNamespace(head_dim=16, num_attention_heads=8, num_key_value_heads=4, hidden_size=hidden, intermediate_size=intermediate, rms_norm_eps=1e-06, num_hidden_layers=1)
    module.rotary_emb = _Rotary()
    module.norm = SimpleNamespace(weight=_FakeTensor(hidden))
    return module

def _fake_pi05_natives(monkeypatch):
    alive = set()
    counter = [0]

    def make(kind):

        def create(linear_acc32):
            assert isinstance(linear_acc32, bool)
            counter[0] += 1
            alive.add(counter[0])
            return counter[0]

        def set_weights(handle, *args):
            return None

        def destroy(handle):
            alive.discard(handle)
        return (create, set_weights, destroy)
    for (kind, module) in (('gemma', gemma), ('adarms', adarms)):
        (create, set_weights, destroy) = make(kind)
        monkeypatch.setattr(module.torch.ops.rpu, f'{kind}_create', create, raising=False)
        monkeypatch.setattr(module.torch.ops.rpu, f'{kind}_set_weights', set_weights, raising=False)
        if kind == 'gemma':
            monkeypatch.setattr(module.torch.ops.rpu, 'gemma_set_prefill_pair_rows', lambda handle, rows: None, raising=False)
        monkeypatch.setattr(module.torch.ops.rpu, f'{kind}_set_chunk_envelope', lambda handle, max_kv_len, chunk: None, raising=False)
        monkeypatch.setattr(module.torch.ops.rpu, f'{kind}_destroy', destroy, raising=False)

    class _RPUCache:

        def __init__(self, **kwargs):
            self.kwargs = kwargs
    monkeypatch.setattr(gemma, 'RPUCache', _RPUCache)
    monkeypatch.setattr(adarms, 'RPUCache', _RPUCache)
    monkeypatch.setattr(gemma.torch, 'zeros', lambda *a, **k: _FakeTensor(1, 1, 128))
    monkeypatch.setattr(gemma.torch, 'arange', lambda *a, **k: _FakeTensor(32))
    monkeypatch.setattr(adarms, '_prepare_adarms_dense_weights', lambda model: ([_FakeTensor(384, 128)], [_FakeTensor(384)], [_FakeTensor(384, 128)], [_FakeTensor(384)]))
    empty_scales = lambda *a, **k: ([], [], [], [], [], [], [])
    monkeypatch.setattr(w8a16, 'gemma_decoder_scale_lists', empty_scales)
    monkeypatch.setattr(w8a16, 'pi05_expert_scale_lists', empty_scales)
    return alive

def _pi05_case(kind, monkeypatch):
    _fake_pi05_natives(monkeypatch)
    module = _decoder_module()
    if kind == 'gemma':
        gemma.patch_gemma_model_for_rpu_all_layers_once(module)
        handle_attr = '_rpu_vlm_decoder_handle'
        finalizer_attr = '_rpu_vlm_decoder_handle_finalizer'
        cache_attr = '_rpu_gemma_graph_cache'
        install_attrs = gemma._GEMMA_RUNTIME_INSTALL_ATTRS
        destroy = torch.ops.rpu.gemma_destroy
    else:
        adarms.patch_adarms_model_for_rpu_all_layers_once(module)
        handle_attr = '_rpu_action_handle'
        finalizer_attr = '_rpu_action_handle_finalizer'
        cache_attr = '_rpu_adarms_graph_cache'
        install_attrs = adarms._ADARMS_RUNTIME_INSTALL_ATTRS
        destroy = torch.ops.rpu.adarms_destroy

    def rollback():
        pi05_adapter._clear_component_graph_cache(module, cache_attr)
        pi05_adapter._cleanup_component_handle(module, handle_attr=handle_attr, finalizer_attr=finalizer_attr, destroy=destroy, install_attrs=install_attrs)
    return (module, rollback)

def _vision_module():
    tensor = _FakeTensor(8, 8)
    block = SimpleNamespace(_rpu_q_w=tensor, _rpu_k_w=tensor, _rpu_v_w=tensor, _rpu_q_b=tensor, _rpu_k_b=tensor, _rpu_v_b=tensor, _rpu_o_b=tensor, _rpu_fc1_b=tensor, _rpu_fc2_b=tensor, attn=SimpleNamespace(proj=SimpleNamespace(weight=tensor)), mlp=SimpleNamespace(linear_fc1=SimpleNamespace(weight=tensor), linear_fc2=SimpleNamespace(weight=tensor)), norm1=SimpleNamespace(weight=tensor, bias=tensor), norm2=SimpleNamespace(weight=tensor, bias=tensor))
    module = _Module()
    module.blocks = [block]
    module.config = SimpleNamespace(num_heads=8, hidden_size=128, intermediate_size=256, hidden_act='gelu_pytorch_tanh', spatial_merge_size=2, deepstack_visual_indexes=[])
    module.patch_embed = SimpleNamespace(proj=SimpleNamespace(weight=tensor, bias=None))
    module.pos_embed = _Movable()
    module.merger = _Movable()
    module.deepstack_merger_list = []
    return module

def _gr00t_case(monkeypatch):
    alive = set()
    counter = [0]

    def create():
        counter[0] += 1
        alive.add(counter[0])
        return counter[0]

    def destroy(handle):
        alive.discard(handle)
    monkeypatch.setattr(qwen3vl_vision, '_convert_vision_block_weights_for_rpu', lambda *a, **k: None)
    monkeypatch.setattr(qwen3vl_vision, 'build_vision_rope_tables', lambda *a, **k: (_FakeTensor(), _FakeTensor()))
    monkeypatch.setattr(qwen3vl_vision, '_fold_conv3d_to_linear_weight', lambda weight: _FakeTensor())
    monkeypatch.setattr(qwen3vl_vision, '_make_dummy_vision_kv_caches', lambda layers, capacity, *args: SimpleNamespace(max_seq_len=capacity))
    monkeypatch.setattr(rpu_backend.graph, 'GraphCache', _GraphCache)
    ops = qwen3vl_vision.torch.ops.rpu
    monkeypatch.setattr(ops, 'qwen3vl_vision_create', create, raising=False)
    monkeypatch.setattr(ops, 'qwen3vl_vision_set_weights', lambda *a, **k: None, raising=False)
    monkeypatch.setattr(ops, 'qwen3vl_vision_set_chunk_envelope', lambda *a: None, raising=False)
    monkeypatch.setattr(ops, 'qwen3vl_vision_set_rope', lambda *a, **k: None, raising=False)
    monkeypatch.setattr(ops, 'qwen3vl_vision_set_rope_route', lambda *a, **k: None, raising=False)
    monkeypatch.setattr(ops, 'qwen3vl_vision_set_chunk_size', lambda *a, **k: None, raising=False)
    monkeypatch.setattr(ops, 'qwen3vl_vision_position_idx_keepalive', lambda handle: _FakeTensor(), raising=False)
    monkeypatch.setattr(ops, 'qwen3vl_vision_destroy', destroy, raising=False)
    backbone = _Module()
    module = _vision_module()
    snapshots = tuple(((owner, 'forward' in vars(owner), vars(owner).get('forward')) for owner in (backbone, module)))
    qwen3vl_vision.install_qwen3_vl_vision_for_rpu(module)

    def rollback():
        gr00t_runtime._cleanup_gr00t_partial_build(a_handle=None, d_handle=None, model=backbone, text_model=None, vision_model=module, forward_snapshots=snapshots)
    return (module, rollback)
_CASES = {'pi05_gemma': lambda mp: _pi05_case('gemma', mp), 'pi05_adarms': lambda mp: _pi05_case('adarms', mp), 'gr00t_vision': _gr00t_case}

@pytest.mark.parametrize('case', sorted(_CASES))
def test_rollback_removes_everything_the_real_install_published(case, monkeypatch):
    from rpu_backend.graph import lazy_init_guard
    monkeypatch.setattr(lazy_init_guard, '_verify_lazy_init', lambda model: None)
    (module, rollback) = _CASES[case](monkeypatch)
    published = sorted((n for n in vars(module) if n.startswith('_rpu_')))
    assert published, f'{case}: installer published no _rpu_* state to roll back'
    assert 'forward' in vars(module), f'{case}: installer published no instance forward to restore'
    rollback()
    leftover = sorted((n for n in vars(module) if n.startswith('_rpu_')))
    assert leftover == [], f'{case}: rollback left {leftover} on the module. These are attributes the real installer wrote and the rollback does not know about — the module still looks installed while its native handle is destroyed. Do not fix this by appending to a name tuple unless that tuple is what the rollback actually iterates.'
    assert 'forward' not in vars(module), f'{case}: rollback left the patched instance forward bound; it calls into a handle that has just been destroyed'
