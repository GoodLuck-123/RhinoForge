"""Execute Pi's CPU noise seam; native cache coherency is checked separately."""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


class DeferredGraph:
    def __init__(self):
        self.pending = None
        self.signatures = []

    @contextmanager
    def capture(self, signature):
        assert self.pending is None
        self.signatures.append(signature)
        yield
        assert self.pending is not None
        self.pending()
        self.pending = None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "broadcast"])
@pytest.mark.parametrize("input_device", ["cpu", "rpu"])
def test_pi_cpu_x0_refreshes_all_rows_and_keeps_outputs_independent(
        monkeypatch, dtype, layout, input_device):
    spec = importlib.util.spec_from_file_location(
        "_pi05_cpu_x0_adapter_under_test",
        ROOT / "python/rpu_backend/adapters/pi05/runtime.py")
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    from rpu_backend.adapters.pi05 import weights
    import rpu_backend.graph

    graph = DeferredGraph()
    steps, cs, mad, prefix = 10, 50, 32, 400
    cfg = SimpleNamespace(chunk_size=cs, max_action_dim=mad)
    ec = SimpleNamespace(num_attention_heads=8, num_key_value_heads=8, head_dim=256)
    owner = SimpleNamespace(
        config=cfg, _rpu_fused_denoise_handle=7,
        _rpu_fused_denoise_graph_cache=graph,
        paligemma_with_expert=SimpleNamespace(
            gemma_expert=SimpleNamespace(model=SimpleNamespace(config=ec))),
        action_in_proj=SimpleNamespace(out_features=1024))
    cache = SimpleNamespace(
        k_caches=[SimpleNamespace(size=lambda axis: 128 if axis == 1 else 16)],
        v_caches=[object()])
    plan = SimpleNamespace(
        selected=SimpleNamespace(stage_tuple=SimpleNamespace(
            physical_descriptor=(11, 12), compute_chunk=64)),
        graph_key_words=lambda: (13,))
    monkeypatch.setattr(weights, "_precompute_adarms_cond_all",
                        lambda _owner, n: torch.zeros(n, 1024, dtype=torch.float16))
    monkeypatch.setattr(runtime, "_build_attention_mask_4d",
                        lambda *args, **kwargs: torch.zeros(1, 1, cs, prefix + cs))
    monkeypatch.setattr(runtime, "_set_denoise_rope_position", lambda *args: prefix - 7)
    monkeypatch.setattr(runtime, "_plan_pi05_denoise_action_execution",
                        lambda *args, **kwargs: plan)
    monkeypatch.setattr(runtime, "_fixed_plan_receipt", lambda *args, **kwargs: {})
    monkeypatch.setattr(rpu_backend.graph, "GraphSignature", lambda **kwargs: kwargs)
    monkeypatch.setattr(torch.ops.rpu, "pi05_denoise_step_get_resolved_chunk_size",
                        lambda _: 64, raising=False)
    original_empty = torch.empty
    allocations, inputs, producers = [], [], []

    def empty(*args, **kwargs):
        allocations.append(kwargs.get("device"))
        if kwargs.get("device") == "rpu":
            kwargs["device"] = "cpu"
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)

    def native(handle, x0, k, v, cond, mask, out, dt, position, n, descriptor):
        assert handle == 7 and k is cache.k_caches and v is cache.v_caches
        assert x0.device.type == "cpu" and x0.dtype == torch.float16
        assert x0.shape == (1, cs, mad) and x0.is_contiguous()
        assert dt == -0.1 and position == prefix and n == steps
        assert descriptor == (11, 12)
        assert allocations[-2:] == [input_device, "rpu"]
        inputs.append(x0)
        producers.append(out)
        # The source must survive capture exit, after the native call returns.
        graph.pending = lambda: out.copy_(x0 + torch.tensor(0.125, dtype=torch.float16))

    monkeypatch.setattr(torch.ops.rpu, "pi05_denoise_loop_forward", native, raising=False)
    width = mad * 2 if layout == "strided" else mad
    rows = 1 if layout == "broadcast" else cs
    base = torch.linspace(-1.003, 1.007, rows * width, dtype=dtype).reshape(1, rows, width)
    outputs = []
    for offset in (0.0, 0.375, 0.0):
        value = base + offset
        if layout == "strided":
            value = value[..., ::2]
            assert not value.is_contiguous()
        # The device stand-in allows dtype conversion only. A new host readback
        # on the existing device input path would fail this boundary.
        argument = (value if input_device == "cpu" else SimpleNamespace(
            device=SimpleNamespace(type="rpu"), to=lambda dtype: value.to(dtype)))
        result = runtime._run_denoise_loop(
            owner, argument, torch.ones(1, prefix), cache, steps, prefix, 1)
        expected_x0 = original_empty((1, cs, mad), dtype=torch.float16)
        expected_x0.copy_(value.to(torch.float16))
        expected = (expected_x0 + torch.tensor(0.125, dtype=torch.float16)).float()
        assert result.shape == (1, cs, mad) and result.dtype == torch.float32
        assert torch.equal(result, expected) and torch.isfinite(result).all()
        assert torch.equal(inputs[-1], expected_x0)
        assert inputs[-1].data_ptr() != value.data_ptr()
        value.fill_(9)
        inputs[-1].fill_(-7)
        producers[-1].fill_(-33)
        assert torch.equal(result, expected)
        outputs.append(result)
    assert torch.equal(outputs[0], outputs[2])
    assert not torch.equal(outputs[0], outputs[1])
    assert len({out.data_ptr() for out in outputs}) == 3
    assert len({src.data_ptr() for src in inputs}) == 3
    assert graph.signatures[0] == graph.signatures[1] == graph.signatures[2]
    assert graph.pending is None
