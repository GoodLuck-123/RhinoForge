"""Numerical exports must not become hidden runtime-owned CPU buffers."""
import ast
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import torch


def _export_method():
    path = Path(__file__).parents[1] / "python/rpu_backend/adapters/rhinovla/pipeline.py"
    cls = next(node for node in ast.parse(path.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == "RhinoVLAOnRPU")
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in {"_export_text_prefix_kv_cpu", "export_last_prefix_kv_cpu"}]
    scope = {"torch": torch, "get_cache_layer": lambda cache, index: cache[index]}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), scope)
    return scope


@pytest.mark.parametrize("previous", [None, "inference-owned-prefix"])
def test_export_has_caller_lifetime_and_preserves_inference_state(previous):
    layers = [(torch.full((1, 2, 5, 4), float(i), dtype=torch.float16),
               torch.full((1, 2, 5, 4), float(i + 10), dtype=torch.float16)) for i in range(4)]
    cache = SimpleNamespace(position=3, to_dynamic_cache=lambda device: layers)
    owner = SimpleNamespace(_text_cache=cache, prefix_layer_offset=1,
                            cfg=SimpleNamespace(depth=2), _last_prefix_kv=previous)
    methods = _export_method()
    owner._export_text_prefix_kv_cpu = lambda: methods["_export_text_prefix_kv_cpu"](owner)
    exported = methods["export_last_prefix_kv_cpu"](owner)
    assert owner._last_prefix_kv is previous

    assert len(exported) == 2
    for index, pair in enumerate(exported, 1):
        for kind, tensor in enumerate(pair):
            assert tensor.shape == (1, 2, 3, 4)
            assert tensor.dtype == torch.float32 and tensor.is_contiguous()
            assert torch.equal(tensor, layers[index][kind][:, :, :3].float())
    # CPU exports stay independent while the device cache is refreshed.
    layers[1][0].fill_(100)
    assert bool((exported[0][0] == 1).all())
    refs = [weakref.ref(t) for pair in exported for t in pair]
    del pair, tensor, exported
    assert all(ref() is None for ref in refs)
    assert owner._last_prefix_kv is previous


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_inference_export_supports_denoise_reuse_and_tracks_mutation(dtype):
    from test_rhinovla_prefix_version_guard import _helpers

    snapshot, matches = _helpers()
    methods = _export_method()
    with torch.inference_mode():
        layers = [(torch.ones(1, 2, 3, 4, dtype=dtype),
                   torch.full((1, 2, 3, 4), 2.0, dtype=dtype))]
        owner = SimpleNamespace(
            _text_cache=SimpleNamespace(position=3, to_dynamic_cache=lambda device: layers),
            prefix_layer_offset=0, cfg=SimpleNamespace(depth=1))
        exported = methods["_export_text_prefix_kv_cpu"](owner)
        assert all(not torch.is_inference(t) for pair in exported for t in pair)
        before = snapshot(exported, 0, exported)
        assert matches(before, snapshot(exported, 0, exported))
        layers[0][0].add_(9)
        assert torch.equal(exported[0][0], torch.ones_like(exported[0][0]))
        assert matches(before, snapshot(exported, 0, exported))
        exported[0][0].add_(1)
        assert not matches(before, snapshot(exported, 0, exported))
