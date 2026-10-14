"""Actual adapter mask identity, optional native arguments and READY misses."""

import ast
from collections import OrderedDict
from types import SimpleNamespace as NS

import pytest
import torch

from test_wall_vision_pair_252 import SOURCE, helpers


def prepared_plans():
    source = SOURCE.parents[2] / "runtime/execution_planner.py"
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == "PreparedExecutionPlans")
    scope = {"OrderedDict": OrderedDict}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    return scope["PreparedExecutionPlans"]()


@pytest.mark.parametrize("seq,batch,valid", [(672, 2, 324), (324, 1, 0)])
def test_mask_identity_invalidates_ready_on_owner_or_version(seq, batch, valid):
    identity = helpers()["_ragged_mask_plan_identity"]
    mask = torch.zeros(seq // batch, seq // batch, dtype=torch.float16)
    key = lambda value: identity(value, seq, batch, valid)
    initial = key(mask)
    assert initial is not None and initial == key(mask)
    cache = prepared_plans()
    result = object()
    assert cache.prepare(initial, lambda: result) is result
    cache.freeze()
    assert cache.prepare(key(mask), lambda: pytest.fail("READY hit rebuilt")) is result
    # A view shares the pointer, but has its own TensorImpl owner.
    for replacement in (mask.clone(), mask.view_as(mask)):
        assert key(replacement) != initial
        with pytest.raises(RuntimeError, match="READY miss"):
            cache.prepare(key(replacement), lambda: pytest.fail("READY miss built"))
    mask[0, 0] = 1
    assert key(mask) != initial
    with pytest.raises(RuntimeError, match="READY miss"):
        cache.prepare(key(mask), lambda: pytest.fail("mutated mask rebuilt"))


def test_mask_identity_excludes_unsupported_and_versionless_inputs():
    identity = helpers()["_ragged_mask_plan_identity"]
    for mask in (None, torch.zeros(324, 324), torch.zeros(324, 325, dtype=torch.float16),
                 torch.zeros(324, 324, dtype=torch.float16).t()):
        assert identity(mask, 324, 1, 0) is None
    with torch.inference_mode():
        mask = torch.zeros(324, 324, dtype=torch.float16)
    assert identity(mask, 324, 1, 0) is None
    assert identity(torch.zeros(256, 256, dtype=torch.float16), 768, 3, 0) is None


@pytest.mark.parametrize("seq,batch,valid", [(672, 2, 324), (324, 1, 0)])
def test_real_planner_passes_mask_proof_and_keys_its_owner(monkeypatch, seq, batch, valid):
    h = helpers()
    mask = torch.zeros(seq // batch, seq // batch, dtype=torch.float16)
    calls = []
    monkeypatch.setattr(torch.ops.rpu, "qwen25vl_vision_resolve_stage_domain",
                        lambda *args: calls.append(args) or [123], raising=False)
    def plan(real, physical, budget, **kwargs):
        assert real == (valid * batch if valid else seq)
        assert physical == seq and budget == 0
        expected = (batch, True, None) + ((valid,) if valid else ())
        expected += (h["_ragged_mask_plan_identity"](mask, seq, batch, valid),)
        assert kwargs["plan_signature"] == expected
        assert kwargs["resolve_stage_domain"](seq) == [123]
        kwargs["plan_result_sink"](NS(selected=NS(stage_tuple=NS(physical_descriptor=[7]))))
        return seq, seq
    h["plan_bounded_prefill_execution"] = plan
    owner = NS(_handle=5, _graph_cache=object())
    result = h["_plan_native_execution"](owner, seq, batch, mask, None,
                                         valid_patches_per_image=valid)
    assert result[0] == seq and result[2] == [7]
    assert calls[0][:-1] == (5, seq, batch, True, None, 0, valid)
    assert calls[0][-1] is mask
