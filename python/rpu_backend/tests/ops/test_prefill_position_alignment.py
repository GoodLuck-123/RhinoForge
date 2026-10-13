import os
import pytest
pytestmark = pytest.mark.skipif(os.environ.get('RPU_HW_AVAILABLE') != '1', reason='requires the native extension (set RPU_HW_AVAILABLE=1)')
GEOM = dict(num_q_heads=16, num_kv_heads=8, num_cores=8, head_dim=128, attn_mask_type=1)
LENGTHS = (528, 544, 560, 576, 592)

def _valid_chunks(seq_len: int, position: int) -> list:
    from rpu_backend._cpp_ext import sdpa_helpers as h
    return [cs for cs in range(16, seq_len + 1, 16) if h.is_valid_chunk_size(cs=cs, seq_len=seq_len, position=position, **GEOM)]

def test_aligned_positions_have_plans():
    for position in (0, 512, 1024):
        for seq_len in LENGTHS:
            assert _valid_chunks(seq_len, position), f'position={position}, seq_len={seq_len}: no valid chunk size, but an aligned resume is the SUPPORTED path'

def test_qwen_logical_length_may_be_off_grid_but_physical_plan_may_not():
    assert not _valid_chunks(999, 0)
    assert _valid_chunks(1024, 0)

def test_an_unaligned_position_has_no_plan_at_any_length():
    for position in (500, 501, 515, 1000):
        for seq_len in LENGTHS:
            assert not _valid_chunks(seq_len, position), f"position={position} is off the 16 grid yet seq_len={seq_len} reports valid splits — the alignment rule changed, so update plan_bounded_prefill_execution's error hint"

def test_the_rule_binds_multi_token_forwards_only():
    import ast
    import pathlib
    decoder = pathlib.Path(__file__).resolve().parents[2] / 'runtime' / 'decoder.py'
    tree = ast.parse(decoder.read_text(), filename=str(decoder))
    forward = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_run_causal_decoder_forward'))
    guards = [n for n in ast.walk(forward) if isinstance(n, ast.If)
              and any(isinstance(stmt, ast.Assign)
                      and isinstance(stmt.value, ast.Call)
                      and isinstance(stmt.value.func, ast.Name)
                      and stmt.value.func.id == '_text_prefill_execution_plan'
                      for stmt in n.body)]
    assert len(guards) == 1
    guard, = guards
    assert 'seq_len > 1' in ast.unparse(guard.test), 'the prefill planner must be reached only for multi-token forwards; a 1-token decode step is never chunked and must not be planned'
if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
