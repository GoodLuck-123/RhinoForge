import os
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import rpu_backend
from rpu_backend.runtime.weights import NUM_CORES, SKIPPED_PARTITION, convert_linear_weights_inplace, get_linear_partition, transform_linear_weight
HIDDEN = 1024
Q_DIM = 2048
INTER = 3072
SEQ = 128

def _cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()

def _container(child_name, in_features, out_features):
    mod = nn.Module()
    mod.add_module(child_name, nn.Linear(in_features, out_features, bias=False))
    return mod.half()

@pytest.mark.parametrize('name,in_features,out_features', [('o_proj', Q_DIM, HIDDEN), ('down_proj', INTER, HIDDEN)])
def test_name_rule_and_shape_rule_actually_disagree(name, in_features, out_features):
    info = convert_linear_weights_inplace(_container(name, in_features, out_features))
    assert info == {f'{name}.weight': 0}, 'by-name pin must still produce ROW'
    assert get_linear_partition(in_features, out_features, 2) == 1, 'shape rule must still produce COL -- if it now says ROW the pin is redundant for this shape; re-audit before deleting it'

def test_converter_records_the_layout_it_chose():
    mod = _container('o_proj', Q_DIM, HIDDEN)
    convert_linear_weights_inplace(mod)
    assert mod.o_proj._rpu_linear_partition == 0
    assert mod.o_proj._rpu_linear_num_cores == NUM_CORES

def test_agreeing_linear_keeps_the_stock_forward():
    mod = _container('q_proj', HIDDEN, Q_DIM)
    info = convert_linear_weights_inplace(mod)
    assert info == {'q_proj.weight': 1}
    assert 'forward' not in vars(mod.q_proj), 'a Linear whose recorded layout matches the eager re-derivation must not be wrapped'

def test_diverging_linear_gets_a_guard():
    mod = _container('o_proj', Q_DIM, HIDDEN)
    convert_linear_weights_inplace(mod)
    assert 'forward' in vars(mod.o_proj)

@pytest.mark.skipif(os.environ.get('RPU_HW_AVAILABLE') != '1', reason='requires device hardware (set RPU_HW_AVAILABLE=1 and run with sudo)')
def test_attn_num_cores_divergence_uses_explicit_recorded_layout():
    mod = _container('q_proj', HIDDEN, Q_DIM)
    x = torch.randn(1, SEQ, HIDDEN, dtype=torch.float16)
    expected = F.linear(x.float(), mod.q_proj.weight.detach().float())
    convert_linear_weights_inplace(mod, attn_num_cores=4)
    assert mod.q_proj._rpu_linear_num_cores == 4
    mod = mod.to('rpu')
    actual = mod.q_proj(x.to('rpu')).cpu().float()
    assert _cos(actual, expected) >= 0.999
    assert (actual - expected).norm().item() / expected.norm().item() <= 0.03
DENSE_OUT = 3 * HIDDEN

@pytest.mark.parametrize('name,skip_names', [('dense', None), ('cond', {'cond'}), ('q_proj', {'q_proj'}), ('lm_head', {'lm_head'})])
def test_skip_is_recorded_on_the_module_not_only_in_the_returned_dict(name, skip_names):
    mod = _container(name, HIDDEN, DENSE_OUT)
    info = convert_linear_weights_inplace(mod, skip_names=skip_names)
    assert info == {f'{name}.weight': SKIPPED_PARTITION}
    child = getattr(mod, name)
    assert child._rpu_linear_partition == SKIPPED_PARTITION
    assert child._rpu_linear_num_cores is None, 'nothing was swizzled, so there is no core count to record'

def test_the_skipped_weight_really_is_the_hazard():
    mod = _container('dense', HIDDEN, DENSE_OUT)
    raw = mod.dense.weight.detach().clone()
    convert_linear_weights_inplace(mod)
    assert torch.equal(mod.dense.weight, raw), 'a skip must not touch the weight'
    assert get_linear_partition(HIDDEN, DENSE_OUT, 2) == 1, 'if the shape rule ever returns -1 here the hazard is gone for this geometry and this test needs a new one'

def test_skipped_linear_refuses_eager_rpu_dispatch():
    mod = _container('dense', HIDDEN, DENSE_OUT)
    convert_linear_weights_inplace(mod)
    x = torch.randn(1, SEQ, HIDDEN, dtype=torch.float16)
    mod = mod.to('rpu')
    with pytest.raises(RuntimeError, match='SKIP_LINEAR_NAMES'):
        mod.dense(x.to('rpu'))

def test_skipped_linear_keeps_stock_cpu_dispatch():
    mod = _container('cond', HIDDEN, DENSE_OUT)
    x = torch.randn(1, SEQ, HIDDEN, dtype=torch.float16)
    ref = mod.cond(x)
    convert_linear_weights_inplace(mod, skip_names={'cond'})
    assert torch.equal(mod.cond(x), ref)

def test_a_companion_can_still_swizzle_the_skipped_weight():
    torch.manual_seed(42)
    x = torch.randn(1, SEQ, HIDDEN, dtype=torch.float16)
    mod = _container('q_proj', HIDDEN, Q_DIM)
    ref = mod.q_proj(x)
    raw = mod.q_proj.weight.detach().clone()
    convert_linear_weights_inplace(mod, skip_names={'q_proj'})
    assert torch.equal(mod.q_proj.weight, raw)
    companion_w = transform_linear_weight(mod.q_proj.weight.data, 1, NUM_CORES)
    out = F.linear(x.to('rpu'), companion_w.to('rpu')).cpu()
    cos = _cos(ref, out)
    assert cos > 0.999, f'companion-swizzled copy cos {cos:.6f}'
    mod = mod.to('rpu')
    with pytest.raises(RuntimeError, match='SKIP_LINEAR_NAMES'):
        mod.q_proj(x.to('rpu'))

def test_bare_linear_is_refused_not_silently_skipped():
    with pytest.raises(TypeError, match='named_children'):
        convert_linear_weights_inplace(nn.Linear(HIDDEN, Q_DIM, bias=False).half())

@pytest.mark.parametrize('name,in_features', [('o_proj', Q_DIM), ('down_proj', INTER)])
def test_row_pinned_linear_is_numerically_right_through_eager_dispatch(name, in_features):
    torch.manual_seed(42)
    x = torch.randn(1, SEQ, in_features, dtype=torch.float16)
    mod = _container(name, in_features, HIDDEN)
    ref = getattr(mod, name)(x)
    convert_linear_weights_inplace(mod)
    mod = mod.to('rpu')
    out = getattr(mod, name)(x.to('rpu')).cpu()
    assert out.shape == ref.shape
    cos = _cos(ref, out)
    assert cos > 0.999, f'{name} eager row dispatch cos {cos:.6f}'

def test_public_linear_into_matches_fp32(record_property):
    from rpu_backend.tests.conformance._numeric import format_metrics, require_same_semantic
    gen = torch.Generator().manual_seed(20260908)
    for (in_features, out_features) in ((128, 256), (128, 16)):
        logical_weight = (torch.randn(out_features, in_features, generator=gen) * 0.05).half()
        partition = get_linear_partition(in_features, out_features)
        assert partition == (1 if out_features == 256 else 0)
        weight = transform_linear_weight(logical_weight.clone(), partition).to('rpu')
        for leading in ((), (17,), (2, 16)):
            x = (torch.randn(*leading, in_features, generator=gen) * 0.75).half()
            for with_bias in (False, True):
                bias = (torch.randn(out_features, generator=gen) * 0.2).half() if with_bias else None
                expected = F.linear(x.float(), logical_weight.float(), None if bias is None else bias.float())
                out = torch.full_like(expected.half(), float('nan')).to('rpu')
                pointer = out.data_ptr()
                torch.ops.rpu.linear_into(x.to('rpu'), weight, None if bias is None else bias.to('rpu'), out)
                assert out.data_ptr() == pointer
                metrics = require_same_semantic(out.cpu(), expected, actual_dtype=torch.float16, cosine_min=0.999, relative_l2_max=0.02, row_relative_l2_max=0.02, label=f'linear_into partition={partition} shape={tuple(x.shape)} bias={with_bias}')
                print(format_metrics(metrics))
