"""CPU FP32 transposed-convolution path for the frozen 12 Hz speech decoder."""
import types

import torch
import torch.nn.functional as F


def _transpose_convolution(owner, hidden, output_size=None):
    if output_size is not None:
        raise ValueError("TTS codec uses the configured transposed-convolution output size")
    stride, kernel = owner.stride[0], owner.kernel_size[0]
    projected = torch.matmul(hidden.transpose(1, 2), owner.weight.flatten(1))
    length = (hidden.shape[-1] - 1) * stride + kernel
    result = F.fold(projected.transpose(1, 2), (1, length), (1, kernel),
                    stride=(1, stride)).squeeze(2)
    if owner.bias is not None:
        result = result + owner.bias[None, :, None]
    return result


def install_codec_convolutions(decoder, restore):
    modules = [m for m in decoder.modules() if isinstance(m, torch.nn.ConvTranspose1d)]
    for module in modules:
        if module.weight.device.type != "cpu" or module.weight.dtype != torch.float32:
            raise ValueError("TTS codec convolutions require CPU FP32 weights")
        if module.padding_mode != "zeros":
            raise ValueError("TTS codec convolutions require zero padding")
        if (module.groups != 1 or module.dilation != (1,) or
                module.padding != (0,) or module.output_padding != (0,)):
            raise ValueError("TTS codec transposed-convolution geometry differs")
    for module in modules:
        restore.append((module, module.forward))
        module.forward = types.MethodType(_transpose_convolution, module)
