from __future__ import annotations
import os
import pytest
import torch
from rpu_backend.tests.conformance._numeric import format_metrics, numeric_metrics, require_same_semantic
pytestmark = pytest.mark.skipif(os.environ.get('RPU_HW_AVAILABLE') != '1', reason='requires RPU hardware (set RPU_HW_AVAILABLE=1 and run with sudo)')
COSINE_MIN = 0.99999
RELATIVE_L2_MAX = 0.002
ROW_RELATIVE_L2_MAX = 0.002
CASES = (('ple_gate', 256, 2560, 1, 8), ('ple_projection', 2560, 256, 0, 8), ('pi05_state_projection', 1024, 16, 1, 8), ('row_depth4', 256, 896, 0, 8), ('col_depth8', 2048, 240, 1, 8), ('tp2_col', 512, 256, 1, 2), ('tp2_row_small_k', 512, 64, 0, 2), ('rhinovla_adarms', 3072, 1024, 0, 8), ('rhinovla_fused_adarms', 6144, 1024, 0, 8), ('core4_mlp_gate', 3072, 1024, 1, 4), ('core4_mlp_down', 1024, 3072, 0, 4), ('core6_mlp_gate', 3072, 1024, 1, 6), ('core6_mlp_down', 1024, 3072, 0, 6), ('core6_lm_head', 151936, 1024, 1, 4))

@pytest.mark.parametrize('name,n,k,partition,num_cores', tuple((case for case in CASES if case[0].startswith(('core4_', 'core6_')))))
def test_reduced_core_prefill_projection_matches_cpu(name, n, k, partition, num_cores):
    from rpu_backend.runtime.weights import transform_linear_weight
    m = 16
    gen = torch.Generator(device='cpu').manual_seed(24576 + n + partition)
    x = (torch.rand((m, k), generator=gen) * 0.2 - 0.1).half()
    weight = (torch.rand((n, k), generator=gen) * 0.2 - 0.1).half()
    golden = x.float().matmul(weight.float().t())
    x_rpu = x.to('rpu')
    w_rpu = transform_linear_weight(weight, partition, num_cores).to('rpu')
    ddr = torch.ops.rpu.linear_with_core_count(x_rpu, w_rpu, None, partition, num_cores).cpu()
    ddr_metrics = require_same_semantic(ddr, golden, cosine_min=COSINE_MIN, relative_l2_max=RELATIVE_L2_MAX, row_relative_l2_max=ROW_RELATIVE_L2_MAX, label=f'{name}: explicit DDR ACC32', actual_dtype=torch.float16)
    print(f'{name}: DDR[{format_metrics(ddr_metrics)}]')
