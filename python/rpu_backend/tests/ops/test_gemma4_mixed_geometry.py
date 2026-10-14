import ast
import pathlib
import sys
import rpu_backend
from rpu_backend._cpp_ext import sdpa_helpers as h
(NQ, NKV) = (8, 2)
ATTN_TP = min(8, NKV)
(HD_SLIDING, HD_GLOBAL) = (256, 512)
(LTM, MASK_2D) = (1, 4)
_ROOT = pathlib.Path(__file__).resolve().parents[4]
_MODEL = _ROOT / 'src' / 'fused' / 'rpu_gemma4_model.cpp'
_ADAPTER = _ROOT / 'python' / 'rpu_backend' / 'adapters' / 'gemma4' / '__init__.py'
FAILED = 0

def check(cond, msg):
    global FAILED
    if cond:
        print(f'  ok   {msg}')
    else:
        FAILED += 1
        print(f'  FAIL {msg}')

def ok(head_dim, mask_type, cs, seq_len, position=0):
    return h.is_valid_chunk_size(num_q_heads=NQ, num_kv_heads=NKV, num_cores=ATTN_TP, head_dim=head_dim, attn_mask_type=mask_type, cs=cs, seq_len=seq_len, position=position)

def gemma4_valid(cs, seq_len, windowed):
    v = ok(HD_GLOBAL, LTM, cs, seq_len) and ok(HD_SLIDING, LTM, cs, seq_len)
    if windowed:
        v = v and ok(HD_SLIDING, MASK_2D, cs, seq_len)
    return v

def main() -> int:
    print('[1] the geometries genuinely disagree — the AND is load-bearing')
    for (seq_len, witnesses) in ((512, (224,)), (1024, (160, 192))):
        for cs in witnesses:
            check(ok(HD_SLIDING, LTM, cs, seq_len) and (not ok(HD_GLOBAL, LTM, cs, seq_len)), f'seq={seq_len} cs={cs}: sliding-hd256 legal, global-hd512 NOT')
            check(not gemma4_valid(cs, seq_len, windowed=False), f'seq={seq_len} cs={cs}: the conjunction rejects it')
    for seq_len in (512, 1024):
        check(gemma4_valid(128, seq_len, windowed=False),
              f'seq={seq_len}: the common 128-row chunk is admitted')
    print('[2] the current tiling policy admits large aligned chunks')
    for (seq_len, expected_max) in ((500, 496), (512, 512), (1024, 1024)):
        admissible = [cs for cs in range(16, seq_len + 1, 16) if gemma4_valid(cs, seq_len, windowed=False)]
        check(bool(admissible), f'seq={seq_len}: some chunk size is legal at all')
        check(max(admissible) == expected_max, f'seq={seq_len}: largest legal chunk is {expected_max} (got {max(admissible)}; {len(admissible)} legal sizes)')
    print('[3] the windowed build cannot be laxer than the unwindowed one')
    for seq_len in (500, 512, 1024, 2048):
        for cs in range(16, min(seq_len, 512) + 1, 16):
            if gemma4_valid(cs, seq_len, windowed=True):
                if not gemma4_valid(cs, seq_len, windowed=False):
                    check(False, f'seq={seq_len} cs={cs}: windowed admits what the unwindowed build rejects')
                    break
        else:
            check(True, f'seq={seq_len}: windowed set is a subset of LTM-only')
    print('[4] gemma4 actually ANDs those modes (source contract)')
    src = _MODEL.read_text()
    fn = src[src.index('bool subclass_chunk_size_valid'):]
    fn = fn[:fn.index('\n    }')]
    check('head_dim()' in fn and 'max_sliding_hd_' in fn, 'validates BOTH the global head_dim and the widest sliding head_dim')
    check('ok(hd_g, 1)' in fn and 'ok(max_sliding_hd_, 1)' in fn, 'both LTM checks are unconditional (they run for every build)')
    check('attention_mask.has_value()' in fn and 'ok(max_sliding_hd_, 4)' in fn, 'MASK_2D is added exactly when the build is windowed')
    check('&&' in fn and '||' in fn, 'combined as a conjunction, not a first-match')
    print('[5] the geometry this test hardcodes is the certified profile')
    prof = _ADAPTER.read_text()
    tree = ast.parse(prof, filename=str(_ADAPTER))
    node = next((n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == '_SUPPORTED_TEXT_PROFILE'))
    values = [v.value for v in node.value.elts]
    check(values[3] == NQ and values[4] == NKV, f'num_q_heads={NQ}, num_kv_heads={NKV} match the adapter profile')
    check(values[5] == HD_SLIDING and values[6] == HD_GLOBAL, f'head_dim {HD_SLIDING} sliding / {HD_GLOBAL} global match the profile')
    check('std::min(8, static_cast<int>(num_kv_heads))' in (_ROOT / 'src' / 'core' / 'fused_model_base.cpp').read_text(), f'attn_tp is still min(8, num_kv_heads) => {ATTN_TP} here')
    print(f"\n{('PASS' if FAILED == 0 else 'FAIL')}: {FAILED} failed")
    return 1 if FAILED else 0
if __name__ == '__main__':
    sys.exit(main())

def test_native_selection_contract():
    global FAILED
    FAILED = 0
    assert main() == 0
