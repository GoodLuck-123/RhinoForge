"""Official 27B cold metadata and raw-storage guards; no device or checkpoint."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.qwen3_5 import cores, quant_scope as qs, weights
from test_qwen35_core_profiles import _config as dense_config
from test_qwen35_legacy_finalized_upload import (
    _fixture, _install, _StopAtNativeWeights, host_device,
)


def config27(method=qs.W8A16, version=1):
    config = dense_config(9)
    tc = config.text_config
    for name, value in dict(hidden_size=5120, intermediate_size=17408,
            num_hidden_layers=64, num_attention_heads=24, linear_num_value_heads=48).items():
        setattr(tc, name, value)
    tc.layer_types = ([qs.LINEAR_ATTENTION] * 3 + [qs.FULL_ATTENTION]) * 16
    config.vision_config.out_hidden_size = 5120
    spec = {"method": method}
    if method == qs.W4A16_PGRP:
        spec["group_size"] = 32
    config.quant_config = ({"format_version": 1, **spec} if version == 1 else
                           {"format_version": 2, "default": spec,
                            "projection_overrides": {"linear_attn.in_proj_b": spec}})
    return config


@pytest.mark.parametrize("method", [qs.W8A16, qs.W4A16_PGRP])
@pytest.mark.parametrize("version", [1, 2])
def test_modern_profile_accepts_equivalent_raw_and_hf_metadata(method, version):
    config = config27(method, version)
    for candidate in (config, config.to_dict()):
        assert qs.validate_modern_27b_profile(candidate)
        assert not qs.validate_legacy_text_profile(candidate)
    for count in (4, 6, 8):
        if method == qs.W4A16_PGRP and count != 8:
            with pytest.raises(ValueError, match="requires num_cores=8"):
                cores.validate_core_profile(config, {"model": {"num_cores": count}})
        else:
            assert cores.validate_core_profile(config, {"model": {"num_cores": count}}).num_cores == count


@pytest.mark.parametrize("field,value", [("head_dim", 128), ("num_attention_heads", 16),
    ("num_hidden_layers", True), ("hidden_act", "gelu"), ("linear_num_value_heads", 32),
    ("attention_bias", True), ("attn_output_gate", False)])
def test_modern_metadata_near_neighbours_reject_before_allocation(field, value):
    config = config27()
    setattr(config.text_config, field, value)
    with pytest.raises(ValueError, match="exact dense semantics"):
        qs.validate_modern_27b_profile(config)


def test_fp16_source_recognition_does_not_grant_public_admission():
    config = config27()
    del config.quant_config
    assert qs.validate_modern_27b_profile(config, allow_fp16=True)
    with pytest.raises(ValueError, match="capacity validation"):
        cores.validate_core_profile(config)


@pytest.mark.parametrize("fault", ["mixed", "group64", "tied", "vision", "nested"])
def test_modern_profile_rejects_semantic_and_precision_conflicts(fault):
    config = config27(qs.W4A16_PGRP, 2)
    if fault == "mixed":
        config.quant_config["projection_overrides"]["linear_attn.in_proj_b"] = {"method": qs.FP16}
    elif fault == "group64":
        config.quant_config = {"method": qs.W4A16_PGRP, "group_size": 64}
    elif fault == "tied":
        config.tie_word_embeddings = True
    elif fault == "vision":
        config.vision_config.out_hidden_size = 4096
    else:
        config.text_config.quant_config = deepcopy(config.quant_config)
    with pytest.raises(ValueError):
        qs.validate_modern_27b_profile(config)


def _raw_uniform(method):
    inner, config = _fixture()
    config.quant_config = {"method": method}
    if method == qs.W4A16_PGRP:
        config.quant_config["group_size"] = 32
    paths = qs.quant_relpaths(qs.is_full_from_layer_types(config.layer_types))
    for name in paths:
        module = inner.get_submodule(name)
        module.weight = torch.nn.Parameter(torch.zeros_like(module.weight, dtype=torch.int8), requires_grad=False)
        shape = ((module.out_features,) if method == qs.W8A16 else
                 (module.in_features // 32, module.out_features))
        scale = torch.ones(shape, dtype=torch.float16)
        if "weight_scale" in module._buffers:
            module.weight_scale = scale
        else:
            module.register_buffer("weight_scale", scale)
    return inner, config


@pytest.mark.parametrize("method", [qs.W8A16, qs.W4A16_PGRP])
@pytest.mark.parametrize("fault", [None, "shape", "zero", "nan", "floating_weight"])
def test_all_raw_projection_scales_are_checked_including_last_layer(method, fault):
    inner, config = _raw_uniform(method)
    last = inner.layers[-1].self_attn.o_proj
    if fault == "shape":
        last.weight_scale = torch.ones(1, dtype=torch.float16)
    elif fault == "zero":
        last.weight_scale.reshape(-1)[-1] = 0
    elif fault == "nan":
        last.weight_scale.reshape(-1)[-1] = float("nan")
    elif fault == "floating_weight":
        last.weight = torch.nn.Parameter(last.weight.half(), requires_grad=False)
    if fault:
        with pytest.raises(ValueError):
            weights.validate_modern_27b_cold_weights(inner, config, cold_cpu=True)
    else:
        weights.validate_modern_27b_cold_weights(inner, config, cold_cpu=True)


def test_existing_9b_vl_does_not_inherit_modern_w4_admission():
    config = dense_config(9)
    config.quant_config = {"method": qs.W4A16_PGRP, "group_size": 32}
    with pytest.raises(ValueError, match="9B VL"):
        cores.validate_dense_large_vl_profile(config)


def test_w4_raw_out_of_range_values_are_rejected_before_packing():
    inner, config = _raw_uniform(qs.W4A16_PGRP)
    inner.layers[-1].self_attn.o_proj.weight[-1, -1] = 8
    with pytest.raises(ValueError, match="signed INT4"):
        weights.validate_modern_27b_cold_weights(inner, config, cold_cpu=True)


def test_modern_uniform_upload_moves_only_final_projections(monkeypatch, host_device):
    """Exercise the real text install/transform loop with small tensor storage."""
    from rpu_backend.adapters.qwen3_5 import text

    inner, config = _raw_uniform(qs.W8A16)
    monkeypatch.setattr(text, "validate_modern_27b_profile", lambda _: True)
    originals = [m.weight.detach() for m in inner.modules()
                 if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d))]
    pointers = {tensor.data_ptr() for tensor in originals}
    original_to = torch.Tensor.to

    def finalized_to(tensor, *args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        if isinstance(device, (str, torch.device)) and torch.device(device).type == "rpu":
            assert tensor.data_ptr() not in pointers
        return original_to(tensor, *args, **kwargs)

    payloads = []

    def native_boundary(*args):
        payloads.append(args)
        raise _StopAtNativeWeights

    monkeypatch.setattr(torch.Tensor, "to", finalized_to)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_weights", native_boundary, raising=False)
    with pytest.raises(_StopAtNativeWeights):
        _install(inner, config)
    args, = payloads
    # GDN Q/K/V/Z/out/b/a now use ordinary W8; companion scalars stay FP16.
    assert all(args[index][0].dtype == torch.int8 for index in (25, 26, 35, 36, 37, 41, 42))
    assert args[27][0].dtype == args[28][0].dtype == torch.float16
    assert inner.layers[0].linear_attn.in_proj_qkv.weight.numel() == 0


@pytest.mark.parametrize("method,count", [(qs.W8A16, 4), (qs.W8A16, 6),
                                        (qs.W8A16, 8), (qs.W4A16_PGRP, 8)])
@pytest.mark.parametrize("fault", ["raw", "allocator", "late_pool", "no_device", None])
def test_official_cold_owner_reserves_before_mutation_and_never_bulk_uploads(monkeypatch, method, count, fault):
    from rpu_backend.adapters import qwen3_5 as adapter
    from rpu_backend.api import causal_lm
    from rpu_backend.api._execution import bind_execution_session
    from rpu_backend.graph import GraphRuntimePolicy

    config = config27(method)
    events = []

    class Inner:
        # Keep the real unstamped HF child config: quantization belongs to root.
        def __init__(self):
            self.config = config.text_config

        def to(self, *_args, **_kwargs):
            pytest.fail("modern 27B must not upload the raw text stack")

    class Model:
        def __init__(self):
            self.config = config
            self.model = SimpleNamespace(language_model=Inner())
            self._rpu_execution = {"model": {"num_cores": count}}

        def half(self):
            events.append("half")
            return self

        def get_output_embeddings(self):
            return SimpleNamespace(weight=SimpleNamespace(shape=(248320, 5120)))

    model = Model()
    owner = adapter.Qwen3_5Adapter.__new__(adapter.Qwen3_5Adapter)
    owner.model, owner._rpu_is_ready = model, False
    owner._execution_session = bind_execution_session(
        model, model._rpu_execution, entry_point="modern27 cold test", graph_mode="BOUNDED_ONESHOT",
        supported={"model": {"num_cores"}})
    owner._execution_session._bind_planner_owner(model.model.language_model)
    monkeypatch.setattr(causal_lm, "_LIVE_REF", None)
    monkeypatch.setattr(causal_lm, "_LIVE_TERMINAL_REASON", None)

    def claim(value):
        assert value is model
        causal_lm._claim_live_instance(value)
        events.append("claim")

    def release(value):
        assert value is model
        causal_lm._release_live_instance(value)
        events.append("release")

    def cleanup(*_args):
        if fault is not None:
            # The early resource handler releases before outer cleanup runs.
            assert causal_lm._LIVE_REF is None
        causal_lm._release_live_instance(model)
        events.append("cleanup")

    monkeypatch.setattr(adapter, "_claim_live_instance", claim)
    monkeypatch.setattr(adapter, "_release_live_instance", release)
    monkeypatch.setattr(adapter, "_cleanup_failed_text_install", cleanup)
    for name in ("qwen3_5_set_execution_cores", "qwen3_5_get_execution_topology",
                 "qwen3_5_vision_set_execution_cores", "qwen3_5_vision_get_execution_topology",
                 "linear_with_accumulation"):
        monkeypatch.setattr(torch.ops.rpu, name, lambda *_: None, raising=False)

    def raw_check(value):
        assert value is model
        events.append("raw-check")
        if fault == "raw":
            raise RuntimeError("bad last raw tensor")

    monkeypatch.setattr(cores, "validate_dense_large_vl_weights", raw_check)
    monkeypatch.setattr(cores, "validate_cold_weight_dtype", lambda *_a, **_kw: None)
    monkeypatch.setattr(cores, "validate_text_weight_geometry", lambda *_: None)
    monkeypatch.setattr(cores, "validate_text_mlp_padding_storage", lambda *_a, **_kw: None)

    def allocator(enabled):
        assert enabled is True
        assert causal_lm._LIVE_REF() is model
        events.append("caching-allocator")
        if fault == "allocator":
            raise RuntimeError("allocator already frozen to direct")

    def prepare(policy):
        assert policy.execution_core_count == count
        assert policy.qwen35_dense_27b_sdk_budget and not policy.qwen35_legacy_27b_sdk_budget
        assert policy.graph_arena_count == 4
        events.append("four-arenas")
        if fault == "late_pool":
            raise RuntimeError("pool already frozen")
        return fault != "no_device"

    def install(inner, passed_config, *_args, **kwargs):
        assert inner is model.model.language_model and passed_config is config
        assert kwargs.get("_cpu_stage_weights", False) is False
        assert kwargs["_public_prefill_graph_mode"] == "BOUNDED_ONESHOT"
        assert kwargs["_graph_runtime_policy"].qwen35_dense_27b_sdk_budget
        events.append("finalized-install")
        raise _StopAtNativeWeights

    monkeypatch.setattr(torch.rpu, "set_caching_allocator", allocator)
    monkeypatch.setattr(torch.rpu, "get_caching_allocator",
                        lambda: pytest.fail("cold install must claim allocator atomically"))
    monkeypatch.setattr(GraphRuntimePolicy, "prepare_arenas", prepare)
    monkeypatch.setattr(adapter, "install_qwen3_5_text_for_rpu", install)
    with pytest.raises((_StopAtNativeWeights, RuntimeError, adapter.RPUBackendError)):
        owner.to_rpu(max_seq_len=128)
    if fault is None:
        assert events == ["raw-check", "claim", "caching-allocator", "four-arenas",
                          "half", "finalized-install", "cleanup"]
    else:
        assert "half" not in events and "finalized-install" not in events
        assert not hasattr(model, "_rpu_swizzle_started")
        expected = ["raw-check"]
        if fault != "raw":
            expected += ["claim", "caching-allocator"]
            if fault != "allocator":
                expected.append("four-arenas")
            expected += ["release", "cleanup"]
        assert events == expected
    assert causal_lm._LIVE_REF is None
    # A resource conflict must not strand the process-wide model owner.
    replacement = Model()
    causal_lm._claim_live_instance(replacement)
    causal_lm._release_live_instance(replacement)


@pytest.mark.parametrize("method,count", [(qs.W8A16, 4), (qs.W8A16, 6),
                                        (qs.W8A16, 8), (qs.W4A16_PGRP, 8)])
def test_real_cache_factory_keeps_top_level_quantization_and_exact_physical_heads(method, count):
    from rpu_backend.api.qwen3_5_cache import Qwen3_5Cache

    class CaptureCache(Qwen3_5Cache):
        def __init__(self, **kwargs):
            self.geometry = kwargs

    config = config27(method)
    original_child = config.text_config.to_dict()
    model = SimpleNamespace(config=config, _rpu_execution={"model": {"num_cores": count}})
    model.model = SimpleNamespace(language_model=SimpleNamespace(
        _rpu_qwen3_5=SimpleNamespace(execution_topology=cores.core_topology(model._rpu_execution))))
    cache = CaptureCache.from_model(model, max_seq_len=8192, device="cpu")
    assert cache.geometry["num_layers"] == 64
    assert cache.geometry["num_kv_heads"] == 24
    assert cache.geometry["num_v_heads"] == 48
    assert cache.geometry["conv_dim"] == 10240
    assert cache.geometry["attn_tp"] == (8 if count == 8 else 4)
    assert cache.geometry["num_cores"] == count
    assert config.text_config.to_dict() == original_child
    assert getattr(config.text_config, "quant_config", None) is None
    if method == qs.W4A16_PGRP:
        with pytest.raises(ValueError, match="requires num_cores=8"):
            CaptureCache.from_config(config, max_seq_len=8192, device="cpu", num_cores=6)
