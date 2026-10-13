from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys
import pytest
ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = (('linear_partition', 'test_linear_col_partition.py'), ('linear_partition', 'test_linear_row_partition.py'), ('linear_partition', 'test_linear_spm_chunking.py'), ('linear_boundary', 'test_linear_k_boundary.py'), ('linear_boundary', 'test_linear_n_boundary.py'), ('kv_insert', 'test_insert_kcache.py'), ('kv_insert', 'test_insert_vcache.py'), ('kv_insert', 'test_vcache_comprehensive.py'), ('kv_insert', 'test_vcache_padding.py'), ('isolated_runtime', 'test_cast_uint8_fp16_isolated.py'), ('ring_gelu', 'test_gelu.py'), ('dma_owned', 'test_graph_replay_dma_ordering'), ('fmb_ring', 'test_spm_pipeline_contract'))

@pytest.mark.skipif(os.environ.get('RPU_HW_AVAILABLE') != '1', reason='requires device hardware')
@pytest.mark.parametrize('group,relative_path', SCRIPTS, ids=lambda value: Path(value).stem if isinstance(value, str) else str(value))
def test_hardware_script_entrypoint(group: str, relative_path: str) -> None:
    environment = os.environ.copy()
    timeout = 600
    if group in {'dma_owned', 'fmb_ring'}:
        assert relative_path == {'dma_owned': 'test_graph_replay_dma_ordering', 'fmb_ring': 'test_spm_pipeline_contract'}[group]
        candidates = [path for depth in range(3) for path in (ROOT / 'build').glob('*/' * depth + relative_path) if path.is_file()]
        assert len(candidates) == 1, f'expected one built {relative_path}, found {candidates}'
        path = candidates[0]
        command = [str(path)]
        environment['LKN_SKIP'] = '1'
        timeout = 300
    else:
        path = (Path(__file__).parent / relative_path).resolve()
        assert path.is_file(), f'hardware script is missing: {path.relative_to(ROOT)}'
        command = [sys.executable, '-B']
        command.append(str(path))
    try:
        result = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f'{path.name} exceeded {timeout} seconds: {exc}')
    if result.stdout:
        print(result.stdout, end='')
    if result.stderr:
        print(result.stderr, end='', file=sys.stderr)
    assert result.returncode == 0, f'{path.name} exited {result.returncode}; see captured output above'
