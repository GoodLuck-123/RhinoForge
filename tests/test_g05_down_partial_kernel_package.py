"""Verify the actual cold N672 byte permutation."""
import re
from pathlib import Path
import numpy as np
from test_siglip_cold_profile import _body
ROOT=Path(__file__).resolve().parents[1]


def test_actual_partial_down_packing_preserves_signed_owner_slices():
    source=(ROOT/'src/ops/rpu_linear.cpp').read_text()
    body=_body(source,'at::Tensor rpu_pack_g05_down_partial_w8a16(')
    chain=body.split('auto packed = weight.cpu()',1)[1].split('.to(weight.device())',1)[0]
    calls=re.findall(r'\.(view|narrow|permute|contiguous)\(([^)]*)\)',chain)
    labels=np.arange(1024*4096,dtype=np.uint32).reshape(1024,4096)
    installed=labels.reshape(64,16,8,16,32).transpose(0,3,2,1,4).copy().reshape(1024,4096)
    actual=installed
    for op,args in calls:
        dims=tuple(map(int,re.findall(r'\d+',args)))
        if op=='view':actual=actual.reshape(dims)
        elif op=='permute':actual=actual.transpose(dims)
        elif op=='contiguous':actual=actual.copy()
        else:
            axis,start,count=dims;slices=[slice(None)]*actual.ndim;slices[axis]=slice(start,start+count);actual=actual[tuple(slices)]
    assert actual.shape==(8,344064)
    for core in range(8):
        restored=actual[core].reshape(42,16,16,32).transpose(0,2,1,3).reshape(672,512)
        np.testing.assert_array_equal(restored,labels[:672,core*512:(core+1)*512])
