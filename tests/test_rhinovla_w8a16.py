"""Host contracts for expert-only quantization before irreversible swizzling."""
import pytest
import torch

from rpu_backend.adapters.rhinovla import convert


def expert_fixture():
    expert = torch.nn.Module()
    expert.layers = torch.nn.ModuleList()
    for _ in range(2):
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        for parent, name in convert._EXPERT_PROJECTIONS:
            setattr(getattr(layer, parent), name, torch.nn.Linear(256, 128, bias=False))
        layer.cond = torch.nn.Linear(256, 384)
        expert.layers.append(layer)
    return expert.half()


def test_quantization_preserves_nonprojection_weights_and_natural_scales():
    expert = expert_fixture()
    with torch.no_grad():
        expert.layers[0].self_attn.q_proj.weight[0].zero_()
    originals = {name: param.clone() for name, param in expert.named_parameters()}
    convert._quantize_expert_projections(expert)
    scales = convert._gather_expert_scales(expert)
    assert len(scales) == 7 and all(len(group) == 2 for group in scales)
    for index, layer in enumerate(expert.layers):
        for parent, name in convert._EXPERT_PROJECTIONS:
            linear = getattr(getattr(layer, parent), name)
            expected = originals[f"layers.{index}.{parent}.{name}.weight"].float()
            assert linear.weight.dtype == torch.int8 and not linear.weight.requires_grad
            assert linear.weight_scale.shape == (128,)
            assert torch.isfinite(linear.weight_scale).all() and (linear.weight_scale > 0).all()
            restored = linear.weight.float() * linear.weight_scale.float()[:, None]
            assert ((restored - expected).abs() <= linear.weight_scale.float()[:, None] * 0.501).all()
        assert torch.equal(layer.cond.weight, originals[f"layers.{index}.cond.weight"])
        assert torch.equal(layer.cond.bias, originals[f"layers.{index}.cond.bias"])
    assert not expert.layers[0].self_attn.q_proj.weight[0].any()
    with pytest.raises(ValueError, match="fresh"):
        convert._quantize_expert_projections(expert)


def test_nonfinite_last_weight_fails_without_partial_quantization():
    expert = expert_fixture()
    with torch.no_grad():
        expert.layers[-1].mlp.down_proj.weight[-1, -1] = float("inf")
    with pytest.raises(ValueError, match="nonfinite"):
        convert._quantize_expert_projections(expert)
    assert all(param.dtype == torch.float16 for param in expert.parameters())
    assert not any(hasattr(module, "weight_scale") for module in expert.modules())


@pytest.mark.parametrize("damage", ("missing", "mixed", "shape", "dtype"))
def test_scale_gather_rejects_partial_quantization(damage):
    expert = expert_fixture()
    assert convert._gather_expert_scales(expert) == ()
    convert._quantize_expert_projections(expert)
    linear = expert.layers[-1].mlp.down_proj
    if damage == "missing":
        del linear.weight_scale
    elif damage == "mixed":
        linear.weight = torch.nn.Parameter(linear.weight.half())
    elif damage == "shape":
        linear.weight_scale = linear.weight_scale[:-1]
    else:
        linear.weight_scale = linear.weight_scale.float()
    with pytest.raises(ValueError, match="uniform W8A16"):
        convert._gather_expert_scales(expert)


def test_conversion_quantizes_before_byte_width_swizzle(monkeypatch):
    from rpu_backend.runtime import weights

    expert = expert_fixture()
    events = []

    def swizzle(model, **kwargs):
        assert model.layers[0].self_attn.q_proj.weight.dtype == torch.int8
        assert model.layers[0].cond.weight.dtype == torch.float16
        assert kwargs["skip_names"] == {"cond"}
        events.append("swizzle")

    def move(device):
        assert device == "rpu" and events == ["swizzle"]
        events.append("move")
        return expert

    monkeypatch.setattr(weights, "convert_linear_weights_inplace", swizzle)
    monkeypatch.setattr(expert, "to", move)
    assert convert.convert_expert_for_rpu(expert, w8a16=True) is expert
    assert events == ["swizzle", "move"]
    with pytest.raises(RuntimeError, match="already"):
        convert.convert_expert_for_rpu(expert, w8a16=True)
