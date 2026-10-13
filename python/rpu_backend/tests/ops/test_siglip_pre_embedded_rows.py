from __future__ import annotations
import ast
import pathlib
import pytest
from rpu_backend.adapters.siglip import SIGLIP_SEQ_LEN, check_pre_embedded_rows
_ADAPTER = pathlib.Path(__file__).resolve().parents[1].parent / 'adapters' / 'siglip.py'

def test_the_certified_row_count_is_256():
    assert SIGLIP_SEQ_LEN == 256

def test_exactly_256_rows_is_accepted():
    check_pre_embedded_rows((1, 256, 1152))

@pytest.mark.parametrize('rows', [1, 128, 255, 257, 512, 1564])
def test_any_other_row_count_is_refused(rows):
    with pytest.raises(ValueError, match='must have exactly 256 rows'):
        check_pre_embedded_rows((1, rows, 1152))

def test_the_refusal_points_at_halo_rather_than_just_saying_no():
    with pytest.raises(ValueError) as exc:
        check_pre_embedded_rows((1, 1564, 1152))
    msg = str(exc.value)
    assert 'halo/vit_cache.py' in msg
    assert 'REPLAY' in msg, 'the message must say WHAT goes wrong, not just that it is disallowed'

def test_the_4d_patch_embed_branch_is_untouched():
    check_pre_embedded_rows((3, 3, 224, 224))
    check_pre_embedded_rows((1, 3, 224, 224))

def test_a_2d_shape_is_not_this_rules_business():
    check_pre_embedded_rows((256, 1152))

def test_production_forward_actually_calls_the_guard():
    tree = ast.parse(_ADAPTER.read_text(encoding='utf-8'))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and (node.func.id == 'check_pre_embedded_rows')]
    assert calls, 'adapters/siglip.py defines check_pre_embedded_rows but never calls it — the 3D signature trap is open again (MR-B/T2)'
    srcs = {ast.dump(c.args[0]) if c.args else '' for c in calls}
    assert any(('shape' in s for s in srcs)), f"check_pre_embedded_rows must be called with the input tensor's .shape; got args {srcs}"
