"""Frozen CPU codec geometry, overlap boundaries, and output ownership."""
import pytest
import torch
import torch.nn.functional as F

from rpu_backend.adapters.qwen3_tts.codec import (
    install_codec_activations, install_codec_convolutions,
    install_rpu_convolutions, _rpu_convolution,
)


class SnakeBeta(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.randn(5))
        self.beta = torch.nn.Parameter(torch.randn(5))
        self.no_div_by_zero = 1e-9

    def forward(self, hidden):
        alpha = self.alpha[None, :, None].exp()
        beta = self.beta[None, :, None].exp()
        return hidden + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(torch.sin(hidden * alpha), 2)


@pytest.mark.parametrize("length", [1, 7, 1024])
def test_snake_matches_frozen_formula_without_mutating_input_or_retained_output(length):
    torch.manual_seed(12)
    module = SnakeBeta().eval()
    hidden = torch.randn(1, 5, length * 2)[:, :, ::2]
    saved = hidden.clone()
    original, restore = module.forward, []
    with torch.inference_mode():
        expected = module(hidden)
        install_codec_activations(module, restore)
        result = module(hidden)
        assert torch.equal(result, expected) and torch.equal(hidden, saved)
        module(hidden + 1)
        assert torch.equal(result, expected)
        assert restore == [(module, original)]
        module.forward = restore[0][1]
        assert torch.equal(module(hidden), expected)


def test_snake_rejects_parameters_before_patching_any_module():
    decoder = torch.nn.Sequential(SnakeBeta(), SnakeBeta().half())
    originals, restore = [m.forward for m in decoder], []
    with pytest.raises(ValueError, match="CPU FP32"):
        install_codec_activations(decoder, restore)
    assert not restore and [m.forward for m in decoder] == originals


@pytest.mark.parametrize("kernel,stride", [(2, 2), (16, 8), (10, 5), (8, 4), (6, 3)])
@pytest.mark.parametrize("length", [1, 7])
@pytest.mark.parametrize("bias", [False, True])
def test_transpose_matches_previous_fold_at_boundaries(kernel, stride, length, bias):
    torch.manual_seed(12)
    module = torch.nn.ConvTranspose1d(3, 5, kernel, stride=stride, bias=bias).eval()
    hidden = torch.randn(1, 3, length * 2)[:, :, ::2]
    saved = hidden.clone()
    restore = []
    with torch.inference_mode():
        projected = hidden.transpose(1, 2) @ module.weight.flatten(1)
        expected = F.fold(projected.transpose(1, 2),
            (1, (length - 1) * stride + kernel), (1, kernel), stride=(1, stride)).squeeze(2)
        if module.bias is not None:
            expected += module.bias[None, :, None]
        install_codec_convolutions(module, restore)
        result = module(hidden)
        assert result.shape == expected.shape and result.dtype == torch.float32
        assert torch.isfinite(result).all() and torch.equal(result, expected)
        assert torch.equal(hidden, saved)
        held = result.clone()
        module(hidden + 1)
        assert torch.equal(result, held)
        with pytest.raises(ValueError, match="output size"):
            module(hidden, output_size=expected.shape)


@pytest.mark.parametrize("length", [1, 7, 70])
@pytest.mark.parametrize("bias", [False, True])
def test_pointwise_matches_original_and_restores(length, bias):
    torch.manual_seed(12)
    module = torch.nn.Conv1d(3, 5, 1, bias=bias).eval()
    hidden = torch.randn(1, 3, length)
    saved = hidden.clone()
    original = module.forward
    restore = []
    with torch.inference_mode():
        expected = module(hidden)
        previous = module.weight.squeeze(-1) @ hidden
        if module.bias is not None:
            previous = previous + module.bias[None, :, None]
        install_codec_convolutions(module, restore)
        result = module(hidden)
        assert result.shape == expected.shape and result.dtype == torch.float32
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, expected)
        assert torch.equal(result, previous) and torch.equal(hidden, saved)
        held = result.clone()
        module(hidden + 1)
        assert torch.equal(result, held)
        assert restore == [(module, original)]
        module.forward = restore[0][1]
        assert torch.equal(module(hidden), expected)


@pytest.mark.parametrize("invalid", [
    torch.nn.ConvTranspose1d(4, 4, 3, stride=2),
    torch.nn.Conv1d(4, 4, 1, stride=2),
    torch.nn.Conv1d(4, 4, 1, groups=2),
])
def test_geometry_rejected_before_any_patch(invalid):
    modules = torch.nn.Sequential(torch.nn.ConvTranspose1d(4, 4, 2, stride=2), invalid)
    originals = [m.forward for m in modules]
    restore = []
    with pytest.raises(ValueError, match="geometry"):
        install_codec_convolutions(modules, restore)
    assert restore == [] and [m.forward for m in modules] == originals


@pytest.mark.parametrize("channels,out_channels,kernel,dilation,groups", [
    (96, 96, 7, 1, 1), (96, 96, 7, 3, 1), (96, 96, 7, 9, 1),
    (16, 32, 1, 1, 1), (32, 16, 3, 1, 1), (96, 1, 7, 1, 1),
    (16, 16, 7, 1, 16),
])
def test_rpu_im2col_matches_rounded_conv_and_owns_outputs(
        monkeypatch, channels, out_channels, kernel, dilation, groups):
    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **kw:
        self if a == ("rpu",) else original_to(self, *a, **kw))
    seen = []
    def linear(hidden, weight, bias, partition, cores, acc32):
        assert hidden.dtype == weight.dtype == torch.float16
        assert (bias, partition, cores, acc32) == (None, 1, 8, True)
        seen.append(hidden.clone())
        return (hidden.float() @ weight.float().T).half()
    monkeypatch.setattr(torch.ops.rpu, "linear_with_accumulation", linear, raising=False)
    torch.manual_seed(1)
    module = torch.nn.Conv1d(channels, out_channels, kernel, dilation=dilation, groups=groups).eval()
    hidden = torch.randn(1, channels, 2 * ((kernel - 1) * dilation + 11))[:, :, ::2]
    source = hidden.clone()
    weight = module.weight.detach().flatten(1).half()
    if groups != 1:
        weight = torch.block_diag(*(row.unsqueeze(0) for row in weight))
    weight = F.pad(weight, (0, 0, 0, (-out_channels) % 128))
    stats = {"calls": 0}
    with torch.inference_mode():
        result = _rpu_convolution(module, hidden, weight, stats)
        patches = hidden.unfold(-1, (kernel - 1) * dilation + 1, 1)[..., ::dilation]
        assert torch.equal(seen[0], patches.permute(0, 2, 1, 3).reshape(11, -1).half())
        expected = F.conv1d(hidden.half().float(), module.weight.half().float(), dilation=dilation, groups=groups).half().float()
        expected += module.bias[None, :, None]
        torch.testing.assert_close(result, expected, rtol=0.002, atol=0.001)
        assert result.shape == (1, out_channels, 11) and result.dtype == torch.float32
        assert torch.equal(hidden, source)
        retained = result.clone()
        changed = _rpu_convolution(module, hidden + 1, weight, stats)
        assert torch.equal(result, retained) and not torch.equal(changed, result)
        assert stats["calls"] == 2 and not torch.equal(*seen)


def _decoder():
    decoder = torch.nn.Module()
    decoder.quantizer = torch.nn.ModuleDict({name: torch.nn.ModuleDict({
        "input_proj": torch.nn.Conv1d(512, 256, 1, bias=False),
        "output_proj": torch.nn.Conv1d(256, 512, 1, bias=False),
    }) for name in ("rvq_first", "rvq_rest")})
    decoder.convs = torch.nn.ModuleList([
        *(torch.nn.Conv1d(c, c, k, dilation=d) for c in (768, 384, 192, 96)
          for k, d in ((7, 1), (7, 3), (7, 9), (1, 1), (1, 1), (1, 1))),
        torch.nn.Conv1d(512, 1024, 3), torch.nn.Conv1d(1024, 1536, 7),
        torch.nn.Conv1d(1024, 1024, 7, groups=1024),
        torch.nn.Conv1d(1024, 1024, 7, groups=1024), torch.nn.Conv1d(96, 1, 7),
    ])
    decoder.transpose = torch.nn.ConvTranspose1d(16, 16, 2, stride=2)
    return decoder


def test_rpu_install_validates_all_layers_before_transform(monkeypatch):
    from rpu_backend.runtime import weights
    decoder = _decoder()
    decoder.convs[-1].padding = (1,)
    originals = [m.forward for m in decoder.modules()]
    monkeypatch.setattr(weights, "transform_linear_weight", lambda *a: pytest.fail("transformed before validation"))
    restore = []
    with pytest.raises(ValueError, match="geometry"):
        install_rpu_convolutions(decoder, restore, {"calls": 0})
    assert restore == [] and [m.forward for m in decoder.modules()] == originals


def test_invalid_codec_backend_rejected_before_loading(monkeypatch):
    from rpu_backend.api import Qwen3TTSModel
    from rpu_backend.adapters import qwen3_tts
    monkeypatch.setattr(qwen3_tts, "load_cpu_model", lambda *a: pytest.fail("weights loaded"))
    with pytest.raises(ValueError, match="codec_backend"):
        Qwen3TTSModel.from_pretrained("unused", source="unused", codec_backend="cuda")


def test_rpu_install_covers_decoding_convs_and_preserves_unused_layers(monkeypatch):
    from rpu_backend.runtime import weights
    decoder = _decoder()
    originals = {m: m.forward for m in decoder.modules()}
    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **kw:
        self if a == ("rpu",) else original_to(self, *a, **kw))
    converted = []
    def transform(weight, partition, cores):
        converted.append(weight.shape)
        assert weight.shape[0] % 128 == 0 and (partition, cores) == (1, 8)
        return weight
    monkeypatch.setattr(weights, "transform_linear_weight", transform)
    restore, stats = [], {"calls": 0}
    install_rpu_convolutions(decoder, restore, stats)
    assert stats["layers"] == len(restore) == len(converted) == 31
    for module in (decoder.quantizer.rvq_first.input_proj,
                   decoder.quantizer.rvq_rest.input_proj, decoder.transpose):
        assert module.forward == originals[module]
    assert all(m.weight.device.type == "cpu" and m.weight.dtype == torch.float32 for m, _ in restore)
    # The diagonal expansion is checked using the actual prepared forward weight.
    for module in (decoder.convs[-3], decoder.convs[-2]):
        weight = module.forward.__func__.__defaults__[0].view(1024, 1024, 7)
        channel = torch.arange(1024)
        assert torch.equal(weight[channel, channel], module.weight[:, 0].half())
        assert torch.count_nonzero(weight) == torch.count_nonzero(module.weight.half())
    for module, forward in reversed(restore):
        module.forward = forward
    assert all(m.forward == forward for m, forward in originals.items())
