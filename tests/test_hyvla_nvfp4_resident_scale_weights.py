"""Actual cold plan and loader append GU4/GU8 scales without changing GU24 codes."""
import dataclasses
import pytest
import torch
from rpu_backend.adapters.hy_vla import weights
from rpu_backend.quant.hyvla_nvfp4_resident_scales import unpack_hyvla_nvfp4_resident_scales

KEY='RPU_HY_VLA_EXPERT_RESIDENT_FP8_SCALE_LAYERS'
ENV={'RPU_HY_VLA_EXPERT_NVFP4':'1','RPU_HY_VLA_W8A16':'vit,vlm',
     'RPU_HY_VLA_ATTN_TP8':'1','RPU_HY_VLA_EXPERT_GATEUP_RESIDENT':'1',
     'RPU_HY_VLA_EXPERT_GATEUP_RESIDENT_LAYERS':'24'}
PROJECTIONS=('q','k','v','o','gate','up','down')


@pytest.mark.parametrize("layers",[4,8])
def test_actual_scale_cold_plan_strict_zero_four_eight_and_frozen_dependencies(layers):
    default=weights._HyVlaWeightColdPlan.from_environment(ENV)
    assert default.resident_fp8_scale_layers==0 and default.gateup_resident_layers==24
    environment=ENV|{KEY:str(layers)}
    enabled=weights._HyVlaWeightColdPlan.from_environment(environment)
    environment[KEY]='0'
    assert enabled.resident_fp8_scale_layers==layers
    with pytest.raises(dataclasses.FrozenInstanceError):enabled.resident_fp8_scale_layers=0
    assert weights._HyVlaWeightColdPlan.from_environment(environment).resident_fp8_scale_layers==0
    for value in ('1','3','5','7','12','24','04','08','4.0','8.0','true','false','off','-1'):
        with pytest.raises(ValueError,match=KEY):
            weights._HyVlaWeightColdPlan.from_environment(ENV|{KEY:value})
    for override in ({'RPU_HY_VLA_EXPERT_GATEUP_RESIDENT_LAYERS':'12'},
                     {'RPU_HY_VLA_EXPERT_GATEUP_RESIDENT':'0'},
                     {'RPU_HY_VLA_EXPERT_NVFP4':'0'},
                     {'RPU_HY_VLA_EXPERT_O_RESIDENT_LAYERS':'12'},
                     {'RPU_HY_VLA_ATTN_TP8':'0'},
                     {'RPU_HY_VLA_ACTION_MLP_MC':'0'}):
        with pytest.raises(ValueError):
            weights._HyVlaWeightColdPlan.from_environment(ENV|{KEY:str(layers)}|override)
    # Existing FP16/W8 loaders keep their default zero without GU24 constraints.
    assert weights._HyVlaWeightColdPlan.from_environment({}).resident_fp8_scale_layers==0
    assert weights._HyVlaWeightColdPlan.from_environment({'RPU_HY_VLA_W8A16':'all'}).resident_fp8_scale_layers==0


@pytest.mark.parametrize("layers",[0,4,8])
def test_actual_loader_appends_only_existing_scale_bytes_and_uploads_one_carrier(monkeypatch,layers):
    quant_calls,carrier_uploads=[],[]
    class Source:
        def __getitem__(self,key):return torch.tensor([int(key.split('.')[1])+1.])
    packed_template=(torch.arange(2048*512,dtype=torch.int32)%256).byte().reshape(2048,512)
    scale_template=(torch.arange(64*2048,dtype=torch.int32)*29%256).byte().reshape(64,2048)
    def quant(raw,partition,cores):
        index=len(quant_calls)
        quant_calls.append((partition,cores))
        name=PROJECTIONS[index%7]
        packed=(packed_template+index).contiguous() if name in ('gate','up') else torch.tensor([index%256],dtype=torch.uint8)
        scale=(scale_template+index).contiguous() if name in ('gate','up') else torch.tensor([index%256],dtype=torch.uint8)
        return packed,scale,torch.tensor(index+.25,dtype=torch.float32)
    def rpu16(tensor):
        if tensor.ndim==2 and tensor.shape[0]==8 and tensor.numel()>1_000_000:
            carrier_uploads.append((tuple(tensor.shape),tensor.dtype))
        return tensor.half().clone()
    monkeypatch.setattr(weights,'quantize_expert_nvfp4_cpu',quant)
    monkeypatch.setattr(weights,'_rpu_nvfp4',lambda tensor:tensor.clone())
    monkeypatch.setattr(weights,'_rpu16',rpu16)
    monkeypatch.setattr(weights,'_dup_kv',lambda value,*args,**kw:value)
    def forbidden(*args,**kwargs):pytest.fail('NVFP4 resident scales must not requantize or invoke W8 packing')
    monkeypatch.setattr(weights,'_quant',forbidden)
    monkeypatch.setattr(weights,'_pack_expert_gateup_weight',forbidden)
    monkeypatch.setattr(weights,'_pack_expert_gateup_w8_installed',forbidden)
    cold=weights._HyVlaWeightColdPlan.from_environment(ENV|{KEY:str(layers)})
    result=weights._build_expert(Source(),Source(),weights.HyVlaConfig(),cold)
    assert len(quant_calls)==32*7
    assert carrier_uploads==[((8,3145728+layers*16384),torch.float16)]
    carrier,=result['gateup_resident']
    assert carrier.shape==(8,(6*1024*1024+layers*32768)//2)
    code_bytes=carrier[:,:3145728].contiguous().view(torch.uint8).reshape(8,24,2,131072)
    for layer in range(24):
        for projection,name in enumerate(('gate','up')):
            expected=result[name+'_w'][layer].reshape(16,16,8,512)
            for core in range(8):assert torch.equal(code_bytes[core,layer,projection],expected[:,:,core].reshape(-1))
    tail=carrier[:,3145728:].contiguous()
    if layers:
        restored=unpack_hyvla_nvfp4_resident_scales(tail)
        for projection,name in enumerate(('gate','up')):
            assert len(restored[projection])==layers
            for layer in range(layers):assert torch.equal(restored[projection][layer],result[name+'_ws'][layer])
    else:
        assert tail.numel()==0
    for i,name in enumerate(PROJECTIONS):
        table,=result[name+'_ts']
        assert torch.equal(table,torch.arange(32,dtype=torch.float32)*7+i+.25)


def test_actual_loader_rejects_direct_invalid_scale_plan_before_quantization(monkeypatch):
    cold=weights._HyVlaWeightColdPlan.from_environment(ENV|{KEY:'8'})
    def forbidden(*args,**kwargs):pytest.fail('invalid scale plan reached quantization')
    monkeypatch.setattr(weights,'quantize_expert_nvfp4_cpu',forbidden)
    for override in ({'resident_fp8_scale_layers':7},{'resident_fp8_scale_layers':4.0},
                     {'resident_fp8_scale_layers':True},{'gateup_resident_layers':12},
                     {'gateup_resident':False},{'expert_nvfp4':False},{'o_resident_layers':12}):
        with pytest.raises(ValueError,match='resident FP8 scales'):
            weights._build_expert({}, {},weights.HyVlaConfig(),dataclasses.replace(cold,**override))


@pytest.mark.parametrize("layers",[4,8])
def test_runtime_cold_plan_keeps_explicit_scale_count_and_unroll_dependencies(monkeypatch,layers):
    from rpu_backend.adapters.hy_vla import runtime
    from test_hyvla_lazy_weights import _clear_hyvla_cold_environment
    _clear_hyvla_cold_environment(monkeypatch)
    environment=ENV|{KEY:str(layers)}
    plan=runtime.prepare_hy_vla_cold_plan(environment)
    environment[KEY]='0'
    assert plan.weights.resident_fp8_scale_layers==layers
    assert plan.weights.gateup_resident_layers==24 and plan.weights.expert_nvfp4
    assert plan.runner.unroll and plan.native.expert.silu_mul
    with pytest.raises(dataclasses.FrozenInstanceError):plan.weights.resident_fp8_scale_layers=0
    for override in ({'RPU_HY_VLA_DENOISE_UNROLL':'0'},{'RPU_HY_VLA_SILU_MUL':'vlm'}):
        with pytest.raises(ValueError,match='ten-step unroll and expert SILU_MUL'):
            runtime.prepare_hy_vla_cold_plan(ENV|{KEY:str(layers)}|override)
