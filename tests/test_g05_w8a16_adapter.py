"""Host checks for the explicit G05 W8 experiment and its real install ABI."""
from copy import deepcopy
import types

import pytest
import torch

from rpu_backend.adapters.g05 import quant
from rpu_backend.adapters.qwen3_5 import text


def _toy_backbone(*, action=False):
    def linear(n, k):
        return torch.nn.Linear(k, n, bias=False, dtype=torch.float16)

    inner = torch.nn.Module()
    inner.layers = torch.nn.ModuleList()
    for full in ([True] if action else [False, True]):
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        layer.mlp.gate_proj = linear(512, 256)
        layer.mlp.up_proj = linear(512, 256)
        layer.mlp.down_proj = linear(256, 512)
        layer.input_layernorm = torch.nn.LayerNorm(256, dtype=torch.float16)
        layer.post_attention_layernorm = torch.nn.LayerNorm(256, dtype=torch.float16)
        if full:
            layer.self_attn = torch.nn.Module()
            for name, n, k in (("q_proj", 512, 256), ("k_proj", 64, 256),
                               ("v_proj", 64, 256), ("o_proj", 256, 256)):
                setattr(layer.self_attn, name, linear(n, k))
            layer.self_attn.q_norm = torch.nn.LayerNorm(32, dtype=torch.float16)
            layer.self_attn.k_norm = torch.nn.LayerNorm(32, dtype=torch.float16)
        else:
            la = layer.linear_attn = torch.nn.Module()
            for name, n in (("in_proj_qkv", 6144), ("in_proj_z", 2048),
                            ("in_proj_b", 16), ("in_proj_a", 16)):
                setattr(la, name, linear(n, 256))
            la.out_proj = linear(256, 2048)
            la.conv1d = torch.nn.Conv1d(6144, 6144, 4, groups=6144, dtype=torch.float16)
            la.A_log = torch.nn.Parameter(torch.zeros(16, dtype=torch.float16))
            la.dt_bias = torch.nn.Parameter(torch.zeros(16, dtype=torch.float16))
            la.norm = torch.nn.LayerNorm(128, dtype=torch.float16)
        inner.layers.append(layer)
    inner.norm = torch.nn.LayerNorm(256, dtype=torch.float16)
    inner.config = types.SimpleNamespace(
        adaptive_mode="adaLN" if action else None,
        layer_types=(["full_attention"] if action else ["linear_attention", "full_attention"]),
        num_hidden_layers=len(inner.layers), hidden_size=256, intermediate_size=512,
        num_attention_heads=8, num_key_value_heads=2, head_dim=32,
        linear_num_value_heads=16, linear_num_key_heads=16,
        linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6, rope_parameters={"partial_rotary_factor": 0.25},
    )
    return inner


def test_g05_w8_scope_quantization_and_fail_before_mutation():
    inner = _toy_backbone()
    original = deepcopy(inner)
    inventory = quant.quantize_g05_text_weights(inner)
    assert text._validate_projection_specs(inner, [0, 1], inner.config.quant_config)
    from rpu_backend.quant._common import quantize_linear_per_channel

    for item in inventory:
        module = inner.get_submodule(item["name"])
        source = original.get_submodule(item["name"])
        if item["method"] == "w8a16":
            expected, scale = quantize_linear_per_channel(source.weight)
            assert torch.equal(module.weight, expected)
            assert torch.equal(module.weight_scale, scale)
            assert module.weight.dtype == torch.int8
            assert module.weight_scale.dtype == torch.float16
        else:
            assert item["name"].endswith(("in_proj_a", "in_proj_b"))
            assert torch.equal(module.weight, source.weight)
    assert torch.equal(inner.norm.weight, original.norm.weight)
    with pytest.raises(RuntimeError, match="already started"):
        quant.quantize_g05_text_weights(inner)
    malformed = _toy_backbone()
    first = malformed.layers[0].mlp.gate_proj.weight
    malformed.layers[-1].self_attn.o_proj.weight.data[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        quant.quantize_g05_text_weights(malformed)
    assert malformed.layers[0].mlp.gate_proj.weight is first
    assert not hasattr(malformed, "_rpu_g05_quantization_started")


@pytest.mark.parametrize("action", [False, True])
@pytest.mark.parametrize("linear_acc32", [False, True])
@pytest.mark.parametrize("math_precision", ["base", "high"])
def test_g05_cpu_stage_actual_weight_and_scale_abi(monkeypatch, action, linear_acc32, math_precision):
    """Execute shared preparation; stop exactly at native set_weights."""
    from rpu_backend.runtime import _native_retirement
    from rpu_backend.runtime.weights import transform_linear_weight

    torch.set_num_threads(2)
    inner = _toy_backbone(action=action)
    quant.quantize_g05_text_weights(inner, action=action)
    expected = deepcopy(inner)
    original_to = torch.Tensor.to

    def host_to(value, *args, **kwargs):
        args = tuple("cpu" if isinstance(arg, str) and arg == "rpu" else arg for arg in args)
        if kwargs.get("device") == "rpu":
            kwargs["device"] = "cpu"
        return original_to(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", host_to)
    monkeypatch.setattr(text, "_refresh_qwen35_text_retirement", lambda _state: None)
    monkeypatch.setattr(_native_retirement, "_InstalledNativeResource",
                        lambda *_args, **kwargs: types.SimpleNamespace(
                            keepalive=kwargs["keepalive"], finalizer=None))
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_create", lambda: 23, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_destroy", lambda _h: None, raising=False)
    precision = []
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_linear_acc32",
                        lambda *args: precision.append(args), raising=False)
    high_calls = []
    def set_high(*args):
        assert precision == [(23, linear_acc32)]
        high_calls.append(args)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_high_precision", set_high, raising=False)
    saved = []

    class Prepared(Exception):
        pass

    def set_weights(*args):
        assert precision == [(23, linear_acc32)]
        assert high_calls == ([(23, True)] if math_precision == 'high' else [])
        saved.extend(args)
        raise Prepared()

    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_weights", set_weights, raising=False)
    with pytest.raises(Prepared):
        text._install_qwen3_5_text_for_rpu_impl(
            inner, inner.config, max_seq_len=64, chunk_size_cap=0,
            exact_chunk_size=0, padding_budget=0, padding_rows="auto",
            execution_config={"prefill": {"linear_acc32": linear_acc32}},
            control_snapshot={}, graph_cache=None,
            prefill_graph=None, cpu_stage_weights=True, g05_w8a16=True,
            g05_math_precision=math_precision,
            pending_state=types.SimpleNamespace(handle=None),
        )
    # q/k/v/o, MLP and their scales arrive at set_weights with their real ABI.
    full = 0 if action else 1
    attn = expected.layers[full].self_attn
    query, gate = text.split_q_gate(attn.q_proj.weight, 8, 32)
    qscale, gatescale = text.split_q_gate_scale(attn.q_proj.weight_scale, 8, 32)
    assert saved[1][full].dtype == torch.int8
    assert torch.equal(saved[1][full], transform_linear_weight(query, 1, 8))
    assert torch.equal(saved[7][full], transform_linear_weight(gate, 1, 8))
    assert torch.equal(saved[43][full], qscale)
    assert torch.equal(saved[47][full], gatescale)
    kv = text._replicate_kv_heads(attn.k_proj.weight, 2, 32, 4)
    kvscale = text._replicate_kv_head_scale(attn.k_proj.weight_scale, 2, 32, 4)
    assert torch.equal(saved[2][full], transform_linear_weight(kv, 1, 8))
    assert torch.equal(saved[44][full], kvscale)
    for weight_index, scale_index, name, partition in (
        (10, 48, "gate_proj", 1), (11, 49, "up_proj", 1), (12, 50, "down_proj", 0),
    ):
        module = getattr(expected.layers[full].mlp, name)
        assert saved[weight_index][full].dtype == torch.int8
        assert torch.equal(saved[weight_index][full], transform_linear_weight(module.weight, partition, 8))
        assert torch.equal(saved[scale_index][full], module.weight_scale)
    assert saved[15].dtype == torch.float16  # synthetic Action final norm stays FP16.
    if not action:
        la = expected.layers[0].linear_attn
        assert saved[35][0].dtype == torch.int8  # GDN Q split.
        assert torch.equal(saved[53][0], la.in_proj_qkv.weight_scale[:2048])
        assert saved[41][0].dtype == torch.float16  # padded GDN b.
        assert saved[56][0].numel() == 0


def test_w8_private_admission_and_precision_idempotence(monkeypatch):
    from test_g05_adapter_contracts import _load_adapter

    module = _load_adapter()
    policy = types.SimpleNamespace(_rpu_g05_policy_ready=True, _rpu_g05_weight_dtype="w8a16")
    monkeypatch.setattr(module, "_g05_policy_runtime_ready", lambda _: True)
    assert module.patch_g05_policy_for_rpu(policy, weight_dtype="w8a16") is policy
    with pytest.raises(ValueError, match="conflicts"):
        module.patch_g05_policy_for_rpu(policy, weight_dtype="fp16")
    with pytest.raises(ValueError, match="weight_dtype"):
        module.patch_g05_policy_for_rpu(policy, weight_dtype="w4a16")
    for installer in (module.patch_g05_vlm_for_rpu, module.patch_g05_vision_for_rpu):
        with pytest.raises(ValueError, match="exact K1"):
            installer(object(), weight_dtype="w8a16")
    with pytest.raises(ValueError, match="CPU-staged"):
        text.install_qwen3_5_text_for_rpu(object(), object(), _g05_w8a16=True)
