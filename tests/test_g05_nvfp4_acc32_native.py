"""Execute ACC32 production launchers with typed owners."""
from pathlib import Path

import pytest
from test_g05_q_gate_native import _compile_run
from test_siglip_cold_profile import _body

ROOT = Path(__file__).resolve().parents[1]


def _function(source, name):
    marker = 'void ' + name + '('
    start = source.index(marker)
    return source[start:source.index('{', start)] + _body(source, marker)


@pytest.mark.parametrize('kind', ['paired', 'swiglu', 'o', 'down'])
def test_acc32_actual_launcher_boundaries_owners_and_replay(tmp_path, monkeypatch, kind):
    if kind in ('paired', 'swiglu'):
        import test_g05_nvfp4_pair_native as fixture
        test = fixture.test_pair_typed_four_owners_and_absolute_spm_boundaries
        old_name = 'rpu_launch_g05_gate_up_paired_nvfp4_acc16_kernel'
        new_name = f'rpu_launch_g05_gate_up_{kind}_nvfp4_acc32_kernel'
    elif kind == 'o':
        import test_g05_nvfp4_o_resident as fixture
        test = fixture.test_o_nvfp4_original_owner_validation_typed_preload_and_spm_registers
        old_name = 'rpu_launch_g05_o_resident_nvfp4_acc16_kernel'
        new_name = 'rpu_launch_g05_o_resident_nvfp4_acc32_kernel'
    else:
        import test_g05_nvfp4_down_resident as fixture
        test = fixture.test_down_nvfp4_original_owner_validation_typed_preload_and_spm_registers
        old_name = 'rpu_launch_g05_down_resident_nvfp4_acc16_kernel'
        new_name = 'rpu_launch_g05_down_resident_nvfp4_acc32_kernel'
    captured = []
    monkeypatch.setattr(fixture, '_compile_run', lambda _p, _n, cpp: captured.append(cpp))
    test(tmp_path)
    source = (ROOT / 'src/ops/rpu_linear.cpp').read_text()
    cpp = captured[0]
    old, new = _function(source, old_name), _function(source, new_name)
    assert cpp.count(old) == 1
    cpp = cpp.replace(old, new).replace(old_name, new_name)
    if kind in ('paired', 'swiglu'):
        pair_id = 'G05_GATE_UP_PAIRED_NVFP4_ACC32_M32N512K1024_N128'
        fused_id = 'G05_GATE_UP_SWIGLU_NVFP4_ACC32_M32N512K1024_N128'
        start = cpp.index('enum class KernelId{')
        end = cpp.index('};', start)+2
        cpp = cpp[:start] + f'enum class KernelId{{{pair_id},{fused_id},G05_GATE_UP_SWIGLU_NVFP4_ACC16_M32N512K1024_N128}};' + cpp[end:]
        expected = fused_id if kind == 'swiglu' else pair_id
        cpp = cpp.replace('stage_kernel_ddr_registers(KernelId,std::vector',
                          'stage_kernel_ddr_registers(KernelId id,std::vector')
        cpp = cpp.replace('assert(!pending&&!written&&v.size()==4);',
                          f'assert(id==KernelId::{expected});assert(!pending&&!written&&v.size()==4);')
        if kind == 'swiglu':
            cpp = cpp.replace('++launches;g.pending=g.written=false;',
                              'for(int r:{128,129,130,131})assert(g.kernel.regs.at(r)==0);'
                              'g.kernel.regs[128]=8;g.kernel.regs[130]=8;'
                              '++launches;g.pending=g.written=false;')
    else:
        id32 = ('G05_O_RESIDENT_NVFP4_ACC32_M32N1024K256_N80' if kind == 'o'
                else 'G05_DOWN_RESIDENT_NVFP4_ACC32_M32N1024K512_N64')
        cpp = cpp.replace('enum class KernelId{', f'enum class KernelId{{{id32},', 1)
        cpp = cpp.replace('Kernel* get_kernel(KernelId){++lookups;',
                          f'Kernel* get_kernel(KernelId id){{assert(id==KernelId::{id32});++lookups;')
    _compile_run(tmp_path, 'g05_acc32_' + kind + '_owners', cpp)
