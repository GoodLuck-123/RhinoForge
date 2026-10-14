from __future__ import annotations
import types
import pytest
import torch

def test_build_causal_prefix_mask() -> None:
    from rpu_backend.adapters.halo import build_causal_prefix_mask
    m0 = build_causal_prefix_mask(0, 8)
    assert tuple(m0.shape) == (8, 8)
    assert [(m0[i] == 0).sum().item() for i in range(8)] == [1, 2, 3, 4, 5, 6, 7, 8]
    mc = build_causal_prefix_mask(4698, 8)
    assert tuple(mc.shape) == (8, 4706)
    assert bool((mc[:, :4698] == 0).all()), '全部前缀须可见'
    assert [(mc[i] == 0).sum().item() for i in range(8)] == [4699, 4700, 4701, 4702, 4703, 4704, 4705, 4706]
    assert bool(torch.isinf(mc[0, 4699:]).all()), 'row0 新 token 因果遮蔽'

def test_rope_base_decouple_contract() -> None:
    from rpu_backend.adapters.halo.text_cache import HaloTextCacheUpdate
    s = types.SimpleNamespace(max_seq_len=4736)
    assert HaloTextCacheUpdate._rope_base(s, torch.arange(3, 3 + 8), 8) == 3
    assert HaloTextCacheUpdate._rope_base(s, torch.arange(0, 8), 8) == 0
    with pytest.raises(ValueError):
        HaloTextCacheUpdate._rope_base(s, torch.tensor([3, 4, 5, 7, 8, 9, 10, 11]), 8)
    with pytest.raises(ValueError):
        HaloTextCacheUpdate._rope_base(s, torch.arange(0, 7), 8)
    s_small = types.SimpleNamespace(max_seq_len=8)
    with pytest.raises(ValueError):
        HaloTextCacheUpdate._rope_base(s_small, torch.arange(4, 12), 8)

def _inv_freq(cfg) -> torch.Tensor:
    return 1.0 / cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float64) / cfg.head_dim)

def test_build_seq_rope_tables_values(monkeypatch) -> None:
    from rpu_backend.adapters.halo import HaloConfig
    from rpu_backend.adapters.halo import text_cache as tc
    monkeypatch.setattr(tc, '_to_rpu_half', lambda t: t.contiguous())
    cfg = HaloConfig()
    half = cfg.head_dim // 2
    (cos, sin) = tc.build_seq_rope_tables(cfg, 64)
    assert tuple(cos.shape) == (64, half) and tuple(sin.shape) == (64, half)
    assert cos.dtype == torch.float16 and sin.dtype == torch.float16
    inv = _inv_freq(cfg)
    for p in (0, 1, 17, 63):
        ang = p * inv
        assert torch.allclose(cos[p].double(), ang.cos().to(torch.float16).double(), atol=0.002)
        assert torch.allclose(sin[p].double(), ang.sin().to(torch.float16).double(), atol=0.002)
    assert torch.allclose(cos[0].double(), torch.ones(half, dtype=torch.float64), atol=0.002)
    assert torch.allclose(sin[0].double(), torch.zeros(half, dtype=torch.float64), atol=0.002)
    with pytest.raises(ValueError):
        tc.build_seq_rope_tables(cfg, 60)
    with pytest.raises(ValueError, match='positive multiple'):
        tc.build_seq_rope_tables(cfg, 0)
    with pytest.raises(ValueError, match='head_dim'):
        tc.build_seq_rope_tables(HaloConfig(head_dim=127), 64)

class _FakeKV:

    def __init__(self, shape, device_type: str) -> None:
        self._shape = torch.Size(shape)
        self._device = types.SimpleNamespace(type=device_type)
        self.copied_from = None
        self.dtype = torch.float16

    def data_ptr(self):
        return id(self)

    def is_contiguous(self):
        return True

    @property
    def device(self):
        return self._device

    @property
    def shape(self):
        return self._shape

    def copy_(self, other):
        self.copied_from = other
        return self

    def to(self, *args, **kwargs):
        return self

class _FakeCache:

    def __init__(self, k_caches, v_caches) -> None:
        self.k_caches = k_caches
        self.v_caches = v_caches
        self.reset_pos = None

    def reset_to_position(self, pos: int) -> None:
        self.reset_pos = pos
_SWZ_SHAPE = (1, 4, 1, 8, 8, 16, 16)

def _inject_self(monkeypatch, cache_device_type, cache_shape=_SWZ_SHAPE, num_layers=2):
    from rpu_backend.adapters.halo import HaloConfig
    from rpu_backend.adapters.halo import text_cache as tc
    monkeypatch.setattr(tc, '_swizzle_prefix_7d', lambda k, m, value_layout=False: _FakeKV(_SWZ_SHAPE, 'cpu'))
    cfg = HaloConfig(num_layers=num_layers)
    k_caches = [_FakeKV(cache_shape, cache_device_type) for _ in range(num_layers)]
    v_caches = [_FakeKV(cache_shape, cache_device_type) for _ in range(num_layers)]
    cache = _FakeCache(k_caches, v_caches)
    graph = types.SimpleNamespace(clear=lambda : None, cache_invariant_ok=lambda : True)
    s = types.SimpleNamespace(cfg=cfg, cache=cache, max_seq_len=64, _prefix_epoch=0, _graph_cache=graph)
    prefix_len = 16
    pk = [torch.zeros(prefix_len, cfg.num_kv_heads, cfg.head_dim) for _ in range(num_layers)]
    pv = [torch.zeros(prefix_len, cfg.num_kv_heads, cfg.head_dim) for _ in range(num_layers)]
    return (tc, s, cache, pk, pv, prefix_len)

def test_inject_prefix_inplace_branch(monkeypatch) -> None:
    from rpu_backend.adapters.halo.text_cache import HaloTextCacheUpdate
    (_tc, s, cache, pk, pv, P) = _inject_self(monkeypatch, 'rpu')
    orig_k0 = cache.k_caches[0]
    HaloTextCacheUpdate._inject_prefix(s, pk, pv, P)
    assert s._prefix_epoch == 0
    assert cache.k_caches[0] is orig_k0
    assert orig_k0.copied_from is not None
    assert cache.reset_pos == P

def test_inject_prefix_reassign_on_cpu_device(monkeypatch) -> None:
    from rpu_backend.adapters.halo.text_cache import HaloTextCacheUpdate
    (_tc, s, cache, pk, pv, P) = _inject_self(monkeypatch, 'cpu')
    orig_k0 = cache.k_caches[0]
    HaloTextCacheUpdate._inject_prefix(s, pk, pv, P)
    assert s._prefix_epoch == 1
    assert cache.k_caches[0] is not orig_k0
    assert cache.reset_pos == P

def test_inject_prefix_reassign_on_shape_mismatch(monkeypatch) -> None:
    from rpu_backend.adapters.halo.text_cache import HaloTextCacheUpdate
    (_tc, s, _cache, pk, pv, P) = _inject_self(monkeypatch, 'rpu', cache_shape=(9, 9, 9))
    HaloTextCacheUpdate._inject_prefix(s, pk, pv, P)
    assert s._prefix_epoch == 1

def test_inject_prefix_rejects_wrong_prefix_shape(monkeypatch) -> None:
    from rpu_backend.adapters.halo.text_cache import HaloTextCacheUpdate
    (_tc, s, _cache, pk, pv, P) = _inject_self(monkeypatch, 'rpu')
    with pytest.raises(ValueError):
        HaloTextCacheUpdate._inject_prefix(s, pk[:1], pv, P)
    with pytest.raises(ValueError):
        HaloTextCacheUpdate._inject_prefix(s, [torch.zeros(P, 3, 3) for _ in pk], pv, P)

@pytest.mark.parametrize('prefix_len,seq,max_seq_len,num_layers', [(0, 8, 64, 2), (16, 8, 64, 2), (10, 8, 64, 2), (34, 8, 64, 2), (4698, 8, 4736, 1)])
def test_read_tail_kv_vs_full_dequant(prefix_len, seq, max_seq_len, num_layers) -> None:
    from rpu_backend.adapters.halo import HaloConfig
    from rpu_backend.adapters.halo.text_cache import HaloTextCacheUpdate, _dynamic_layer_kv
    from rpu_backend.api.cache import RPUCache
    cfg = HaloConfig(num_layers=num_layers)
    cache = RPUCache(num_layers=num_layers, batch_size=1, max_seq_len=max_seq_len, num_kv_heads=cfg.num_kv_heads, head_dim=cfg.head_dim, device='cpu', attn_tp=cfg.attn_tp)
    torch.manual_seed(0)
    for i in range(num_layers):
        cache.k_caches[i].copy_(torch.randn_like(cache.k_caches[i]))
        cache.v_caches[i].copy_(torch.randn_like(cache.v_caches[i]))
    cache.reset_to_position(prefix_len + seq)
    dc = cache.to_dynamic_cache(device='cpu')
    s = types.SimpleNamespace(cache=cache, cfg=cfg)
    out = HaloTextCacheUpdate._read_tail_kv(s, prefix_len, seq)
    assert len(out) == num_layers
    for layer in range(num_layers):
        (kf, vf) = _dynamic_layer_kv(dc, layer)
        ref_k = kf[0, :, prefix_len:prefix_len + seq, :].permute(1, 0, 2).contiguous()
        ref_v = vf[0, :, prefix_len:prefix_len + seq, :].permute(1, 0, 2).contiguous()
        (ok, ov) = out[layer]
        assert tuple(ok.shape) == (seq, cfg.num_kv_heads, cfg.head_dim)
        assert torch.equal(ok, ref_k), f'layer {layer} K tail mismatch'
        assert torch.equal(ov, ref_v), f'layer {layer} V tail mismatch'
