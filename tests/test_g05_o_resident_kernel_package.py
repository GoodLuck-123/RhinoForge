"""Native registry and packer permutation preserve O's row-TP8 bytes."""
from pathlib import Path
import re
import numpy as np
from test_siglip_cold_profile import _body

ROOT=Path(__file__).resolve().parents[1]


def test_g05_o_resident_registry_matches_native_route():
    declarations=(ROOT/'src/core/rpu_kernel_cache.h').read_text()
    registry=(ROOT/'src/core/rpu_kernel_cache.inc').read_text()
    assert 'G05_O_RESIDENT_W8A16_ACC16_M32N1024K256' in declarations
    assert 'g05_o_resident_w8a16_acc16_m32n1024k256' in registry


def test_actual_o_packer_permutation_preserves_each_core_k_slice_and_four_owners():
    source=(ROOT/'src/ops/rpu_linear.cpp').read_text();body=_body(source,'at::Tensor rpu_pack_g05_o_resident_w8a16(')
    chain=re.search(r'core_weights.push_back\(weight.cpu\(\)(.*?)\);',body,re.S).group(1)
    calls=re.findall(r'\.(view|permute|contiguous)\((\{[\d, ]+\})?\)',chain)
    assert calls and 'at::cat(core_weights, 1)' in body
    labels=np.arange(1024*2048,dtype=np.uint32).reshape(1024,2048)
    installed=labels.reshape(64,16,8,8,32).transpose(0,3,2,1,4).copy().reshape(1024,2048)
    packed=[]
    for layer in range(4):
        data=installed+layer*labels.size
        for op,args in calls:
            dims=tuple(map(int,re.findall(r'\d+',args)))
            data=data.reshape(dims)if op=='view'else data.transpose(dims)if op=='permute'else data.copy()
        packed.append(data)
    actual=np.concatenate(packed,axis=1)
    assert actual.shape==(8,1048576)
    for core in range(8):
        for layer in range(4):
            shard=actual[core,layer*262144:(layer+1)*262144]
            restored=shard.reshape(64,8,16,32).transpose(0,2,1,3).reshape(1024,256)
            np.testing.assert_array_equal(restored,labels[:,core*256:(core+1)*256]+layer*labels.size)
