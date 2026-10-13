"""Cached masks retain mutation tracking under inference mode."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "python/rpu_backend"
spec = importlib.util.spec_from_file_location(
    "_versioned_masks_cpu", PYTHON / "runtime/masks.py")
masks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(masks)


def _functions(relative, names, **globals_):
    """Load actual host-only methods without importing device/model loaders."""
    source = ast.parse((PYTHON / relative).read_text())
    selected = []
    for node in ast.walk(source):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            selected.append(node)
    assert len(selected) == len(names)
    module = ast.Module(body=[ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0),
        *selected], type_ignores=[])
    env = dict(torch=torch, versioned_cpu_mask=masks.versioned_cpu_mask,
               **globals_)
    exec(compile(ast.fix_missing_locations(module), relative, "exec"), env)
    return env


@pytest.mark.parametrize("inference_source", [False, True])
def test_owned_mask_keeps_values_and_tracks_mutation(inference_source):
    with torch.inference_mode(inference_source):
        source = torch.tensor([[0., -50000., float("-inf")], [1., 2., 3.]])[:, ::2]
        owned = masks.versioned_cpu_mask(source)
        other = masks.versioned_cpu_mask(source)
        assert not owned.is_inference() and not owned.requires_grad
        assert owned.is_contiguous() and owned.dtype == torch.float16
        assert owned.device.type == "cpu"
        assert torch.equal(owned, source.half())
        assert owned.data_ptr() != source.data_ptr() != other.data_ptr()
        version = owned._version
        owned[0, 0] = 7
        assert owned._version > version
        assert source[0, 0] == 0 and other[0, 0] == 0


def test_wall_cache_preserves_owner_and_separates_same_shape_masks():
    env = _functions("adapters/wall_oss/action.py",
                     {"_build_action_mask", "_action_mask_cached"})
    owner = SimpleNamespace(_host_cache=True, _action_mask_cache={},
                            _causal_action_attention_mask=False,
                            _build_action_mask=env["_build_action_mask"])
    get = env["_action_mask_cached"]
    with torch.inference_mode():
        a = get(owner, 16, 4, 15, 0)
        assert get(owner, 16, 4, 15, 0) is a
        b = get(owner, 16, 4, 14, 0)
        c = get(owner, 16, 4, 15, 2)
        assert a.shape == b.shape == c.shape
        assert not torch.equal(a, b) and not torch.equal(a, c)
        assert get(owner, 16, 4, 15, 0) is a
        for mask in (a, b, c):
            assert not mask.is_inference()
        assert torch.isneginf(a[:, 15]).all()
        assert (a[:, :15] == 0).all()
        assert torch.isneginf(c[:2, 18:]).all()
        version = a._version
        a[0, 0] = -11
        assert get(owner, 16, 4, 15, 0)._version > version
        assert b[0, 0] == 0


@pytest.mark.parametrize("v2", [False, True])
def test_lingbot_masks_stay_on_cpu_with_padding_and_state_block(v2):
    common = _functions("adapters/lingbot_vla/runtime.py",
                        {"_make_att_2d_masks"})
    options = dict(N_ACTION=50, _make_att_2d_masks=common["_make_att_2d_masks"])
    if v2:
        # Disable optional alignment spans to isolate execution-prefix padding.
        env = _functions("adapters/lingbot_vla_v2/runtime.py",
                         {"build_suffix_mask_bool", "_build_suffix_mask"},
                         apply_suffix_prefix_blocking_=lambda *args: None,
                         **options)
        build = lambda: env["_build_suffix_mask"](16, None, real_prefix_len=14)
    else:
        env = _functions("adapters/lingbot_vla/runtime.py",
                         {"_build_suffix_mask"}, **options)
        build = lambda: env["_build_suffix_mask"](16)
    with torch.inference_mode():
        mask = build()
        assert mask.device.type == "cpu" and not mask.is_inference()
        assert tuple(mask.shape) == (51, 67)
        assert (mask[:, :14] == 0).all()
        assert mask[0, 17] < -40000  # state query cannot see action keys
        assert mask[1, 16] == 0 and mask[1, -1] == 0
        if v2:
            assert (mask[:, 14:16] < -40000).all()


def test_hy_constant_cache_changes_mask_when_same_shape_input_changes():
    drops = []
    env = _functions("adapters/hy_vla/runtime.py",
                     {"_cached", "_prefill_constants", "_denoise_constants"},
                     MASK_NEG=-50000., _rpu16=lambda x: x.half(),
                     build_rope_tables=lambda pos, cfg: (pos.clone(), pos.clone()))
    owner = SimpleNamespace(w=SimpleNamespace(cfg=SimpleNamespace(suffix_len=2)),
                            prefix_len=4, _const_key={}, _const_val={},
                            _psubs={}, _drop_psubs=lambda: drops.append(True))
    owner._cached = lambda *args: env["_cached"](owner, *args)
    with torch.inference_mode():
        pos = torch.arange(4)
        prefix = torch.ones(4, 4, dtype=torch.bool)
        prefix[-1] = False
        modality = torch.ones(4, dtype=torch.bool)
        a = env["_prefill_constants"](owner, pos, prefix, modality)[2]
        assert env["_prefill_constants"](owner, pos, prefix, modality)[2] is a
        prefix[0, 2] = False
        b = env["_prefill_constants"](owner, pos, prefix, modality)[2]
        assert a.shape == b.shape and a is not b
        assert a[0, 0, 0, 2] == 0 and b[0, 0, 0, 2] < -40000
        assert a[0, 0, -1, -1] == 0  # invalid-query diagonal preserved
        dm = torch.ones(2, 6, dtype=torch.bool)
        suffix = torch.arange(4, 6)
        c = env["_denoise_constants"](owner, 6, pos, suffix, dm)[2]
        assert env["_denoise_constants"](owner, 6, pos, suffix, dm)[2] is c
        dm[0, 1] = False
        d = env["_denoise_constants"](owner, 6, pos, suffix, dm)[2]
        assert c is not d and d[0, 0, 0, 1] < -40000
        assert all(not x.is_inference() and x.device.type == "cpu" for x in (a, b, c, d))
        assert len(drops) == 2


def test_pi05_denoise_content_cache_publishes_versioned_mask(monkeypatch):
    env = _functions("adapters/pi05/runtime.py", {"_build_attention_mask_4d"},
                     _hash_mask=lambda x: x.numpy().tobytes())
    common = _functions("adapters/lingbot_vla/runtime.py", {"_make_att_2d_masks"})
    monkeypatch.setitem(sys.modules, "lerobot.policies.pi05.modeling_pi05",
                        SimpleNamespace(make_att_2d_masks=common["_make_att_2d_masks"]))
    owner = SimpleNamespace(_rpu_denoise_mask_cache=None,
                            _prepare_attention_masks_4d=lambda x:
                            torch.where(x[:, None], 0., float("-inf")))
    get = env["_build_attention_mask_4d"]
    with torch.inference_mode():
        prefix = torch.tensor([[True, True, False]])
        a = get(owner, prefix, 3, 1, 4, torch.float32, "cpu")
        assert get(owner, prefix, 3, 1, 4, torch.float32, "cpu") is a
        prefix[0, 1] = False
        b = get(owner, prefix, 3, 1, 4, torch.float32, "cpu")
        assert a is not b and a.shape == b.shape
        assert not a.is_inference() and not b.is_inference()
        assert a[0, 0, 0, 1] == 0 and torch.isneginf(b[0, 0, 0, 1])
