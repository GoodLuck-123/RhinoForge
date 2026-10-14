"""Real HF cold installation reaches the native ABI without device allocation.

FakeTensorMode preserves the complete 24-layer model and real packing operations.
device transport, Graph allocation and native endpoints are doubles; fake parameter
conversion avoids Torch's unsupported weakref swap. The native set_weights
endpoint deliberately fails: these are installation contracts, not execution
or numerical acceptance tests.
"""
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

import rpu_backend
from rpu_backend.adapters import qwen3_5
from rpu_backend.api import causal_lm
from rpu_backend.runtime import weights


class NativeBoundaryReached(RuntimeError):
    pass


def _model(dtype=torch.float16):
    # Official dense 0.8B geometry; no checkpoint or network is needed in CI.
    config = Qwen3_5Config(
        architectures=["Qwen3_5ForConditionalGeneration"],
        image_token_id=248056, tie_word_embeddings=True,
        text_config=dict(hidden_size=1024, intermediate_size=3584,
            num_hidden_layers=24, num_attention_heads=8, num_key_value_heads=2,
            head_dim=256, vocab_size=248320, attention_bias=False,
            hidden_act="silu", attn_output_gate=True, use_cache=True,
            full_attention_interval=4,
            layer_types=["full_attention" if i % 4 == 3 else "linear_attention"
                         for i in range(24)],
            linear_conv_kernel_dim=4, linear_num_key_heads=16,
            linear_num_value_heads=16, linear_key_head_dim=128,
            linear_value_head_dim=128, rms_norm_eps=1e-6,
            rope_parameters=dict(rope_type="default", rope_theta=10_000_000,
                partial_rotary_factor=0.25, mrope_interleaved=True,
                mrope_section=[11, 11, 10])),
        vision_config=dict(depth=12, hidden_size=768, intermediate_size=3072,
            num_heads=12, out_hidden_size=1024, in_channels=3, patch_size=16,
            temporal_patch_size=2, spatial_merge_size=2,
            num_position_embeddings=2304, hidden_act="gelu_pytorch_tanh",
            deepstack_visual_indexes=[]))
    with FakeTensorMode():
        return Qwen3_5ForConditionalGeneration._from_config(config, dtype=dtype)


@pytest.fixture
def native_boundary(monkeypatch):
    events, padded, packed, payloads = [], [], [], []
    monkeypatch.setattr(causal_lm, "_LIVE_REF", None)
    monkeypatch.setattr(causal_lm, "_LIVE_TERMINAL_REASON", None)
    from rpu_backend.api import _execution
    monkeypatch.setattr(_execution, "_UNSAFE_PROCESS_REASON", None)
    # Keep raw leaves so their logical geometry can be checked at the ABI.
    monkeypatch.setenv("RPU_QWEN3_5_FREE_HF_WEIGHTS", "0")

    tensor_to, module_to = torch.Tensor.to, torch.nn.Module.to

    def host_destination(args, kwargs):
        args, kwargs = list(args), dict(kwargs)
        if args and isinstance(args[0], (str, torch.device)) and str(args[0]).split(":")[0] in ("rpu", "privateuseone"):
            args[0] = "cpu"
        if str(kwargs.get("device", "")).split(":")[0] in ("rpu", "privateuseone"):
            kwargs["device"] = "cpu"
        return args, kwargs

    def to_cpu_tensor(tensor, *args, **kwargs):
        args, kwargs = host_destination(args, kwargs)
        return tensor_to(tensor, *args, **kwargs)

    def to_cpu_module(module, *args, **kwargs):
        target = args[0] if args else kwargs.get("device", "")
        if str(target).split(":")[0] in ("rpu", "privateuseone"):
            # Fake CPU tensors model the uploaded storage; invoking Module.to
            # would use unsupported FakeTensor parameter swapping in Torch 2.10.
            return module
        args, kwargs = host_destination(args, kwargs)
        return module_to(module, *args, **kwargs)

    def fake_half(module):
        # Torch 2.10 Module._apply always swaps FakeTensor Parameters, whose
        # memoized weakrefs make swap_tensors fail. Retain real Tensor.half
        # arithmetic and tied Parameters while replacing only that bookkeeping.
        converted = {}
        for child in module.modules():
            for name, parameter in tuple(child._parameters.items()):
                if parameter is not None:
                    if id(parameter) not in converted:
                        converted[id(parameter)] = torch.nn.Parameter(
                            parameter.half() if parameter.is_floating_point() else parameter,
                            requires_grad=parameter.requires_grad)
                    child._parameters[name] = converted[id(parameter)]
            for name, buffer in tuple(child._buffers.items()):
                if buffer is not None and buffer.is_floating_point():
                    child._buffers[name] = buffer.half()
        return module

    monkeypatch.setattr(torch.Tensor, "to", to_cpu_tensor)
    monkeypatch.setattr(torch.nn.Module, "to", to_cpu_module)
    monkeypatch.setattr(torch.nn.Module, "half", fake_half)

    class Cache:
        def __init__(self, **kwargs):
            self.runtime_policy = kwargs.get("runtime_policy") or SimpleNamespace(graph_arena_count=0)
            events.append("cache")

        def clear(self):
            events.append("clear")

        def cache_invariant_ok(self):
            return True

    class Graph:
        def __init__(self, **kwargs):
            events.append("graph")

        def invalidate(self):
            events.append("invalidate")

    monkeypatch.setattr(rpu_backend.graph, "GraphCache", Cache)
    monkeypatch.setattr(rpu_backend.graph, "Graph", Graph)
    claim = qwen3_5._claim_live_instance

    def claim_model(model):
        events.append("claim")
        return claim(model)

    monkeypatch.setattr(qwen3_5, "_claim_live_instance", claim_model)
    pad = weights.pad_decoder_mlp_projection

    def prepare(weight, **kwargs):
        padded.append((kwargs["projection_name"], tuple(weight.shape), weight.dtype,
                       weight.device.type, kwargs["logical_size"], kwargs["physical_size"]))
        return pad(weight, **kwargs)

    monkeypatch.setattr(weights, "pad_decoder_mlp_projection", prepare)
    pack = weights.transform_linear_weight

    def transform(weight, partition, num_cores=8):
        packed.append((tuple(weight.shape), partition, num_cores))
        return pack(weight, partition, num_cores)

    monkeypatch.setattr(weights, "transform_linear_weight", transform)

    def create():
        events.append("create")
        return 35

    def set_weights(*args):
        payloads.append(args)
        events.append("set_weights")
        raise NativeBoundaryReached("checked cold native set_weights boundary")

    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_create", create, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_destroy",
                        lambda handle: events.append(("destroy", handle)), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_weights", set_weights, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_execution_cores",
                        lambda handle, cores: events.append(("cores", handle, cores)), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_linear_acc32",
                        lambda handle, enabled: events.append(("acc32", handle, enabled)), raising=False)
    # Checked by the public cold ABI preflight; none may be executed here.
    def unexpected(*args, **kwargs):
        pytest.fail("execution went beyond the failing native installation boundary")
    for name in ("qwen3_5_get_execution_topology", "qwen3_5_vision_set_execution_cores",
                 "qwen3_5_vision_get_execution_topology", "linear_with_accumulation"):
        monkeypatch.setattr(torch.ops.rpu, name, unexpected, raising=False)
    return SimpleNamespace(events=events, padded=padded, packed=packed, payloads=payloads)


@pytest.mark.parametrize("cores,dtype", [(4, torch.float16), (6, torch.float16),
    (8, torch.float16), (6, torch.float32), (6, torch.bfloat16)])
def test_public_install_packs_all_layers_once_and_preserves_logical_width(native_boundary, cores, dtype):
    b = native_boundary
    model = _model(dtype)
    inner = model.model.language_model
    adapter = qwen3_5.Qwen3_5Adapter(model, rpu_execution={"model": {"num_cores": cores}})
    with inner.norm.weight.fake_mode, pytest.raises(NativeBoundaryReached):
        adapter.to_rpu(max_seq_len=64)

    assert len(b.payloads) == 1
    args = b.payloads[0]
    tp = 8 if cores == 8 else 4
    physical = 3648 if cores == 6 else 3584
    assert args[0] == 35
    assert args[17:22] == (8, tp, 256, 1024, 3584)
    assert args[30:35] == (16, 128, 128, 6144, 4)
    assert model.config.text_config.intermediate_size == inner.config.intermediate_size == 3584
    assert len(args[16]) == 24 and sum(args[16]) == 6
    for i, layer in enumerate(inner.layers):
        for index, shape in ((10, (physical, 1024)), (11, (physical, 1024)),
                             (12, (1024, physical))):
            assert len(args[index]) == 24
            assert tuple(args[index][i].shape) == shape
            assert args[index][i].dtype == torch.float16 and args[index][i].is_contiguous()
        assert layer.mlp.gate_proj.weight.shape == (3584, 1024)
        assert layer.mlp.up_proj.weight.shape == (3584, 1024)
        assert layer.mlp.down_proj.weight.shape == (1024, 3584)
        assert layer.mlp.down_proj.in_features == 3584
        # Non-MLP projections retain their existing attention/GDN preparation.
        if args[16][i]:
            assert args[1][i].shape == args[7][i].shape == (2048, 1024)
            assert args[2][i].shape == args[3][i].shape == (tp * 256, 1024)
            assert args[4][i].shape == (1024, 2048)
        else:
            assert args[25][i].shape == (2048, 1024)
            assert args[26][i].shape == (1024, 2048)
            assert all(args[index][i].shape == (2048, 1024) for index in (35, 36, 37))
            assert all(args[index][i].shape == (tp, 4, 2048 // tp) for index in (38, 39, 40))
    expected_padding = [(name, shape, torch.float16, "cpu", 3584, 3648)
                        for _ in range(24)
                        for name, shape in (("gate_proj", (3584, 1024)),
                            ("up_proj", (3584, 1024)), ("down_proj", (1024, 3584)))]
    assert b.padded == (expected_padding if cores == 6 else [])
    mlp_shapes = ((physical, 1024), (1024, physical))
    assert [call for call in b.packed if call[0] in mlp_shapes] == [
        call for _ in range(24) for call in (
            ((physical, 1024), 1, cores), ((physical, 1024), 1, cores),
            ((1024, physical), 0, cores))]
    assert b.events[-3:] == ["clear", "invalidate", ("destroy", 35)]
    assert b.events.count("claim") == b.events.count("create") == 1
    assert (("cores", 35, cores) in b.events) == (cores != 8)
    assert inner._rpu_qwen3_5_text_install_started and model._rpu_swizzle_started
    assert not hasattr(inner, "_rpu_qwen3_5")
    assert adapter._handle is None and not adapter._rpu_is_ready
    assert adapter._execution_session.stats()["state"] == "CLOSED"
    # Existing post-mutation failure policy retains the process owner until
    # the caller disposes of this irreversibly transformed model.
    assert causal_lm._LIVE_REF() is model
    assert causal_lm._LIVE_TERMINAL_REASON is None
    calls = list(b.events)
    with pytest.raises(qwen3_5.RPUBackendError, match="prior install failed"):
        adapter.to_rpu(max_seq_len=64)
    assert b.events == calls


@pytest.mark.parametrize("fault", ["noncontiguous", "bias", "integer", "non_cpu", "already_packed", "wrong_shape"])
def test_last_projection_rejected_before_public_claim_or_first_mutation(native_boundary, fault):
    b = native_boundary
    model = _model()
    inner = model.model.language_model
    adapter = qwen3_5.Qwen3_5Adapter(model, rpu_execution={"model": {"num_cores": 6}})
    first = inner.layers[0].mlp.gate_proj.weight
    last = inner.layers[-1].mlp.down_proj
    with first.fake_mode:
        if fault == "noncontiguous":
            last.weight = torch.nn.Parameter(torch.empty(3584, 1024, dtype=torch.float16).t())
        elif fault == "bias":
            last.bias = torch.nn.Parameter(torch.empty(1024, dtype=torch.float16))
        elif fault == "integer":
            last.weight = torch.nn.Parameter(torch.empty(1024, 3584, dtype=torch.int8), requires_grad=False)
        elif fault == "non_cpu":
            last.weight = torch.nn.Parameter(torch.empty(1024, 3584, dtype=torch.float16, device="meta"))
        elif fault == "already_packed":
            last._rpu_linear_partition = 0
        else:
            last.weight = torch.nn.Parameter(torch.empty(1024, 3648, dtype=torch.float16))
        with pytest.raises(ValueError, match="logical geometry|requires shape"):
            adapter.to_rpu(max_seq_len=64)
    assert b.events == b.padded == b.packed == b.payloads == []
    assert inner.layers[0].mlp.gate_proj.weight is first
    assert first.shape == (3584, 1024)
    assert not getattr(model, "_rpu_swizzle_started", False)
    assert not getattr(inner, "_rpu_qwen3_5_text_install_started", False)
    assert not hasattr(inner.layers[0].mlp.gate_proj, "_rpu_linear_partition")
