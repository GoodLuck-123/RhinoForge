"""Convolution and activation paths for the frozen 12 Hz speech decoder."""
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
        result.add_(owner.bias[None, :, None])
    return result


def _pointwise_convolution(owner, hidden):
    result = torch.matmul(owner.weight.squeeze(-1), hidden)
    if owner.bias is not None:
        result.add_(owner.bias[None, :, None])
    return result


def install_codec_activations(decoder, restore):
    modules = [m for m in decoder.modules() if type(m).__name__ == "SnakeBeta"]
    for module in modules:
        if any(p.device.type != "cpu" or p.dtype != torch.float32 or p.ndim != 1
               for p in (module.alpha, module.beta)) or module.alpha.shape != module.beta.shape:
            raise ValueError("TTS SnakeBeta requires matching CPU FP32 channel parameters")
    for module in modules:
        alpha = module.alpha.detach()[None, :, None].exp()
        scale = 1.0 / (module.beta.detach()[None, :, None].exp() + module.no_div_by_zero)

        def forward(owner, hidden, alpha=alpha, scale=scale):
            # Only the new temporary is mutated; residual inputs remain intact.
            return (hidden * alpha).sin_().pow_(2).mul_(scale).add_(hidden)

        restore.append((module, module.forward))
        module.forward = types.MethodType(forward, module)


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


def _rpu_convolution(owner, hidden, weight, stats):
    if (hidden.device.type != "cpu" or hidden.dtype != torch.float32 or
            hidden.ndim != 3 or hidden.shape[:2] != (1, owner.in_channels)):
        raise ValueError("TTS RPU convolution requires batch-one CPU FP32 input")
    dilation = owner.dilation[0]
    width = (owner.kernel_size[0] - 1) * dilation + 1
    count = hidden.shape[-1] - width + 1
    if count <= 0:
        raise ValueError("TTS RPU convolution input is shorter than its kernel")
    # shortcut: CPU im2col and bias remain; fuse them only after measuring transfer cost.
    # Round each input once, before overlapping windows multiply the copy volume.
    patches = hidden.half().unfold(-1, width, 1)[..., ::dilation]
    patches = patches.permute(0, 2, 1, 3).reshape(count, -1).contiguous().to("rpu")
    result = torch.ops.rpu.linear_with_accumulation(
        patches, weight, None, 1, 8, True).cpu().float()[:, :owner.out_channels]
    if owner.bias is not None:
        result.add_(owner.bias)
    stats["calls"] += 1
    return result.T.unsqueeze(0).contiguous()


def install_rpu_convolutions(decoder, restore, stats):
    """FP16/ACC32 Linear for every Conv1d used by the frozen decoder."""
    from collections import Counter
    from rpu_backend.runtime.weights import transform_linear_weight

    # These encoder-only projections are registered on the shared quantizer.
    unused = {"quantizer.rvq_first.input_proj", "quantizer.rvq_rest.input_proj"}
    modules = [m for name, m in decoder.named_modules()
               if isinstance(m, torch.nn.Conv1d) and name not in unused]
    expected = Counter((c, c, k, d, 1) for c in (768, 384, 192, 96)
                       for k, d in ((7, 1), (7, 3), (7, 9), (1, 1), (1, 1), (1, 1)))
    expected.update([(256, 512, 1, 1, 1)] * 2 + [(1024, 1024, 7, 1, 1024)] * 2 +
                    [(512, 1024, 3, 1, 1), (1024, 1536, 7, 1, 1), (96, 1, 7, 1, 1)])
    if Counter((m.in_channels, m.out_channels, m.kernel_size[0], m.dilation[0], m.groups)
               for m in modules) != expected:
        raise ValueError("TTS RPU convolution geometry differs from the 31-layer decode profile")
    for module in modules:
        if module.stride != (1,) or module.padding != (0,) or module.padding_mode != "zeros":
            raise ValueError("TTS RPU convolution geometry differs")
        if (module.weight.device.type != "cpu" or module.weight.dtype != torch.float32 or
                (module.bias is not None and
                 (module.bias.device.type != "cpu" or module.bias.dtype != torch.float32))):
            raise ValueError("TTS RPU convolution requires CPU FP32 source weights")
    prepared = []
    for module in modules:
        weight = module.weight.detach().half()
        if module.groups != 1:
            # shortcut: two short depthwise layers use a dense diagonal; use a
            # grouped device kernel when an admitted asset supports this profile.
            dense = weight.new_zeros(module.out_channels, module.in_channels, module.kernel_size[0])
            channels = torch.arange(module.in_channels)
            dense[channels, channels] = weight[:, 0]
            weight = dense
        weight = weight.flatten(1)
        weight = torch.nn.functional.pad(weight, (0, 0, 0, (-module.out_channels) % 128))
        prepared.append(transform_linear_weight(weight, 1, 8).to("rpu"))
    for module, weight in zip(modules, prepared):
        def forward(owner, hidden, weight=weight):
            return _rpu_convolution(owner, hidden, weight, stats)
        restore.append((module, module.forward))
        module.forward = types.MethodType(forward, module)
    stats["layers"] = len(modules)
