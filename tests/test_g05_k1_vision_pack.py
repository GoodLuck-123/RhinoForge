"""CPU byte/layout contracts for exact K1 B2/B3 packing, using the real function."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


_RUNTIME = Path(__file__).resolve().parents[1] / 'python/rpu_backend/adapters/g05/runtime.py'


def _pack_function():
    source = _RUNTIME.read_text()
    node = next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == '_pack_g05_vision')
    scope = {'torch': torch}
    exec(compile(ast.get_source_segment(source, node), str(_RUNTIME), 'exec'), scope)
    return scope['_pack_g05_vision']


def _vision(*, b3=True, patch=16, temporal=2, merge=2):
    return SimpleNamespace(
        patch_size=patch, spatial_merge_size=merge,
        config=SimpleNamespace(temporal_patch_size=temporal),
        _rpu_g05_vision_forward_impl=SimpleNamespace(_g05_k1_b3_wrapper=b3),
    )


def _release_reference(cameras, vision):
    # Release temporal expansion occurs in C,T order, followed by its T,C
    # reinterpretation. Keep this reference independent of the optimized path.
    chunks, grids, splits = [], [], []
    p, m, t = vision.patch_size, vision.spatial_merge_size, vision.config.temporal_patch_size
    for camera in cameras.values():
        _, frames, channels, height, width = camera.shape
        gh, gw = height // p, width // p
        images = camera.reshape(frames, channels, height, width)
        if images.dtype == torch.float32:
            images = images.half()
        chunks.append(images.unsqueeze(2).expand(-1, -1, t, -1, -1)
                      .reshape(frames, t, channels, gh // m, m, p, gw // m, m, p)
                      .permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
                      .reshape(frames * gh * gw, -1))
        grids.extend([[1, gh, gw]] * frames)
        splits.append((gh // m) * (gw // m))
    return torch.cat(chunks), torch.tensor(grids), splits, frames


def _assert_result(actual, expected):
    assert actual[0].shape == expected[0].shape
    assert actual[0].stride() == expected[0].stride()
    assert actual[0].dtype == expected[0].dtype == torch.float16
    assert torch.equal(actual[0].view(torch.int16), expected[0].view(torch.int16))
    assert torch.equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]


@pytest.mark.parametrize('adjacent', [False, True])
@pytest.mark.parametrize('count', [2, 3])
def test_k1_camera_group_pack_exact_cast_layout_live_images_and_fresh_outputs(adjacent, count):
    pack = _pack_function()
    generator = torch.Generator().manual_seed(20260930)
    images = torch.randn(count, 1, 3, 256, 256, generator=generator)
    # Exercise half rounding boundaries, signed zero, subnormal and overflow.
    edges = torch.tensor([0., -0., 2.**-25, -2.**-25, 2.**-24,
                          1.00048828125, -1.00048828125, 65504., 65520.,
                          float('inf'), float('-inf'), float('nan')])
    images.flatten()[:len(edges)] = edges
    cameras = {str(i): images[i:i+1] if adjacent else images[i:i+1].clone()
               for i in range(count)}
    vision = _vision()
    vision._rpu_g05_vision_forward_impl = SimpleNamespace(
        _g05_k1_camera_group_wrapper=True, _g05_k1_camera_batch=count,
        _g05_k1_b3_wrapper=count == 3)
    output = pack(cameras, vision)
    _assert_result(output, _release_reference(cameras, vision))
    assert output[0].shape == (256 * count, 1536)
    saved = output[0].clone()
    before = cameras['1'][0, 0, 2, 79, 113].clone()
    cameras['1'][0, 0, 2, 79, 113] = 1.25
    changed = pack(cameras, vision)
    _assert_result(changed, _release_reference(cameras, vision))
    assert not torch.equal(changed[0].view(torch.int16), saved.view(torch.int16))
    assert torch.equal(output[0].view(torch.int16), saved.view(torch.int16))
    cameras['1'][0, 0, 2, 79, 113] = before
    assert torch.equal(pack(cameras, vision)[0].view(torch.int16), saved.view(torch.int16))


def test_k1_b3_preserves_release_channel_time_order():
    camera = torch.arange(3, dtype=torch.float32).reshape(1, 1, 3, 1, 1).expand(1, 1, 3, 256, 256)
    packed, *_ = _pack_function()({str(i): camera + i * 10 for i in range(3)}, _vision())
    for camera_idx in range(3):
        source_order = packed[camera_idx * 256].reshape(6, 16, 16)[:, 0, 0].tolist()
        assert source_order == [camera_idx * 10 + i for i in [0, 1, 0, 2, 1, 2]]
        assert source_order != [camera_idx * 10 + i for i in [0, 0, 1, 1, 2, 2]]


@pytest.mark.parametrize('fallback', ['b1', 'fp16', 'k6', 'noncanonical_shape', 'two_cameras'])
def test_other_pack_profiles_keep_legacy_path(monkeypatch, fallback):
    vision = _vision(b3=fallback != 'b1')
    frames = 6 if fallback == 'k6' else 1
    height = 224 if fallback == 'noncanonical_shape' else 256
    count = 2 if fallback == 'two_cameras' else 3
    dtype = torch.float16 if fallback == 'fp16' else torch.float32
    cameras = {str(i): torch.full((1, frames, 3, height, 256), i + 0.125, dtype=dtype)
               for i in range(count)}
    expected = _release_reference(cameras, vision)

    def reject_index_select(*_args, **_kwargs):
        raise AssertionError('noncanonical profile reached the K1 B3 candidate')

    monkeypatch.setattr(torch.Tensor, 'index_select', reject_index_select)
    _assert_result(_pack_function()(cameras, vision), expected)
