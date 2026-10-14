"""Convolution paths for the frozen 12 Hz speech decoder."""
import types

import torch


def _transpose_convolution(owner, hidden, output_size=None):
    if output_size is not None:
        raise ValueError("TTS codec uses the configured transposed-convolution output size")
    batch, _, count = hidden.shape
    stride, kernel = owner.stride[0], owner.kernel_size[0]
    projected = torch.matmul(hidden.transpose(1, 2), owner.weight.flatten(1))
    if kernel == stride:
        result = projected.view(batch, count, owner.out_channels, stride)
        result = result.transpose(1, 2).reshape(batch, owner.out_channels, count * stride)
    else:
        # The frozen decoder overlaps only two adjacent stride-sized blocks.
        chunks = projected.view(batch, count, owner.out_channels, 2, stride)
        result = hidden.new_empty((batch, owner.out_channels, count + 1, stride))
        result[:, :, 0] = chunks[:, 0, :, 0]
        result[:, :, -1] = chunks[:, -1, :, 1]
        torch.add(chunks[:, 1:, :, 0].transpose(1, 2),
                  chunks[:, :-1, :, 1].transpose(1, 2), out=result[:, :, 1:-1])
        result = result.flatten(2)
    if owner.bias is not None:
        result = result + owner.bias[None, :, None]
    return result


def _pointwise_convolution(owner, hidden):
    result = torch.matmul(owner.weight.squeeze(-1), hidden)
    if owner.bias is not None:
        result = result + owner.bias[None, :, None]
    return result


def install_codec_convolutions(decoder, restore):
    modules = [m for m in decoder.modules() if isinstance(m, torch.nn.ConvTranspose1d) or
               (isinstance(m, torch.nn.Conv1d) and m.kernel_size == (1,))]
    for module in modules:
        if module.weight.device.type != "cpu" or module.weight.dtype != torch.float32:
            raise ValueError("TTS codec convolutions require CPU FP32 weights")
        if module.padding_mode != "zeros":
            raise ValueError("TTS codec convolutions require zero padding")
        if module.groups != 1 or module.dilation != (1,) or module.padding != (0,):
            raise ValueError("TTS codec convolution geometry differs")
        if isinstance(module, torch.nn.ConvTranspose1d):
            if (module.output_padding != (0,) or
                    module.kernel_size[0] not in (module.stride[0], 2 * module.stride[0])):
                raise ValueError("TTS codec transposed-convolution geometry differs")
        elif module.stride != (1,):
            raise ValueError("TTS codec pointwise-convolution geometry differs")
    for module in modules:
        restore.append((module, module.forward))
        forward = _transpose_convolution if isinstance(module, torch.nn.ConvTranspose1d) else _pointwise_convolution
        module.forward = types.MethodType(forward, module)


def _residual_convolution(owner, hidden, weight, stats):
    if (hidden.device.type != "cpu" or hidden.dtype != torch.float32 or
            hidden.ndim != 3 or hidden.shape[:2] != (1, owner.in_channels)):
        raise ValueError("TTS RPU residual convolution requires batch-one CPU FP32 input")
    dilation = owner.dilation[0]
    count = hidden.shape[-1] - 6 * dilation
    if count <= 0:
        raise ValueError("TTS RPU residual convolution input is shorter than its kernel")
    # shortcut: CPU im2col and bias remain; fuse them only after measuring transfer cost.
    patches = hidden.unfold(-1, 6 * dilation + 1, 1)[..., ::dilation]
    patches = patches.permute(0, 2, 1, 3).reshape(count, -1).half().contiguous().to("rpu")
    result = torch.ops.rpu.linear_with_accumulation(
        patches, weight, None, 1, 8, True).cpu().float()[:, :owner.out_channels]
    if owner.bias is not None:
        result = result + owner.bias
    stats["calls"] += 1
    return result.T.unsqueeze(0).contiguous()


def install_rpu_residual_convolutions(decoder, restore, stats):
    """Opt-in FP16/ACC32 Linear for the twelve frozen residual Conv1d layers."""
    from collections import Counter
    from rpu_backend.runtime.weights import transform_linear_weight

    modules = [m for m in decoder.modules()
               if isinstance(m, torch.nn.Conv1d) and m.kernel_size == (7,) and
               m.in_channels == m.out_channels and m.in_channels in (768, 384, 192, 96)]
    expected = Counter((channels, dilation) for channels in (768, 384, 192, 96)
                       for dilation in (1, 3, 9))
    if Counter((m.in_channels, m.dilation[0]) for m in modules) != expected:
        raise ValueError("TTS RPU residual convolution geometry differs from the twelve-layer profile")
    for module in modules:
        if (module.out_channels != module.in_channels or module.groups != 1 or
                module.stride != (1,) or module.padding != (0,) or module.padding_mode != "zeros"):
            raise ValueError("TTS RPU residual convolution geometry differs")
        if (module.weight.device.type != "cpu" or module.weight.dtype != torch.float32 or
                (module.bias is not None and
                 (module.bias.device.type != "cpu" or module.bias.dtype != torch.float32))):
            raise ValueError("TTS RPU residual convolution requires CPU FP32 source weights")
    prepared = []
    for module in modules:
        weight = module.weight.detach().flatten(1).half()
        weight = torch.nn.functional.pad(weight, (0, 0, 0, (-module.out_channels) % 128))
        prepared.append(transform_linear_weight(weight, 1, 8).to("rpu"))
    for module, weight in zip(modules, prepared):
        def forward(owner, hidden, weight=weight):
            return _residual_convolution(owner, hidden, weight, stats)
        restore.append((module, module.forward))
        module.forward = types.MethodType(forward, module)
    stats["layers"] = len(modules)
