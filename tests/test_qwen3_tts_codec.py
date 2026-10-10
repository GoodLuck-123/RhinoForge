"""Frozen CPU codec geometry, overlap boundaries, and output ownership."""
import pytest
import torch
import torch.nn.functional as F

from rpu_backend.adapters.qwen3_tts.codec import install_codec_convolutions


@pytest.mark.parametrize("kernel,stride", [(2, 2), (16, 8), (10, 5), (8, 4), (6, 3)])
@pytest.mark.parametrize("length", [1, 7])
@pytest.mark.parametrize("bias", [False, True])
def test_transpose_matches_previous_fold_at_boundaries(kernel, stride, length, bias):
    torch.manual_seed(12)
    module = torch.nn.ConvTranspose1d(3, 5, kernel, stride=stride, bias=bias).eval()
    hidden = torch.randn(1, 3, length * 2)[:, :, ::2]
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
    original = module.forward
    restore = []
    with torch.inference_mode():
        expected = module(hidden)
        install_codec_convolutions(module, restore)
        result = module(hidden)
        assert result.shape == expected.shape and result.dtype == torch.float32
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, expected)
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
