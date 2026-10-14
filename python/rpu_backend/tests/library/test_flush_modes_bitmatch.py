from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path
import pytest
import torch
pytestmark = pytest.mark.skipif(os.environ.get('RPU_HW_AVAILABLE') != '1', reason='requires device hardware (set RPU_HW_AVAILABLE=1)')
_HERE = Path(__file__).resolve().parent
_PROBE = _HERE / 'probe_rpu_logits_bitmatch.py'
_PARAMS = [pytest.param('0.6b', 256, 16, 0, id='0.6b_smoke')]
if os.environ.get('RPU_BITMATCH_FULL') == '1':
    _PARAMS.append(pytest.param('8b', 4096, 64, 128, id='8b_production'))

def _run_probe(mode: str, out_path: Path, size: str, prefill: int, decode: int, chunk_size: int) -> None:
    cmd = [sys.executable, str(_PROBE), mode, str(out_path), '--size', size, '--prefill', str(prefill), '--decode', str(decode), '--chunk-size', str(chunk_size)]
    env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    if res.returncode != 0:
        pytest.fail(f"probe '{mode}' failed (rc={res.returncode}):\n--- stdout ---\n{res.stdout[-2000:]}\n--- stderr ---\n{res.stderr[-2000:]}")

def _capture_pair(tmp_path, size, prefill, decode, chunk_size, tag):
    on_path = tmp_path / f'logits_{size}_flush_on_{tag}.pt'
    off_path = tmp_path / f'logits_{size}_flush_off_{tag}.pt'
    _run_probe('flush_on', on_path, size, prefill, decode, chunk_size)
    _run_probe('flush_off', off_path, size, prefill, decode, chunk_size)
    return (torch.load(on_path, weights_only=False), torch.load(off_path, weights_only=False))

@pytest.mark.parametrize('size,prefill,decode,chunk_size', _PARAMS)
def test_ddr_flush_modes_bit_identical(tmp_path, size, prefill, decode, chunk_size):
    (a, b) = _capture_pair(tmp_path, size, prefill, decode, chunk_size, 'try1')
    assert torch.equal(a['input_ids'], b['input_ids']), 'input_ids differ between probe runs — non-determinism in tokenizer or padding path'
    (la, lb) = (a['logits'], b['logits'])
    assert la.shape == lb.shape, f'logits shape mismatch: flush_on={tuple(la.shape)}, flush_off={tuple(lb.shape)}'
    if torch.equal(la, lb):
        return
    (a2, b2) = _capture_pair(tmp_path, size, prefill, decode, chunk_size, 'try2')
    (la2, lb2) = (a2['logits'], b2['logits'])
    if torch.equal(la2, lb2):
        diff = (la - lb).abs()
        print(f'[flush-bitmatch] TRANSIENT first-try mismatch on {size}/prefill={prefill}/decode={decode}/chunk={chunk_size}: max_abs_diff={diff.max().item():.3g}, retry bit-matched.', flush=True)
        return
    diff1 = (la - lb).abs()
    diff2 = (la2 - lb2).abs()
    use_first = diff1.max().item() >= diff2.max().item()
    (la_show, lb_show) = (la, lb) if use_first else (la2, lb2)
    diff = diff1 if use_first else diff2
    am_on = la_show[0].argmax(-1).tolist()
    am_off = lb_show[0].argmax(-1).tolist()
    argmax_match = sum((1 for (x, y) in zip(am_on, am_off) if x == y))
    n = len(am_on)
    first_diff_pos = next((i for i in range(n) if not torch.equal(la_show[0, i], lb_show[0, i])), -1)
    pytest.fail(f'device logits NOT bit-identical under ddr_flush={{True,False}} on TWO consecutive attempts (not a transient):\n  config           : size={size} prefill={prefill} decode={decode} chunk_size={chunk_size}\n  max abs diff     : {diff.max().item():.6g}\n  mean abs diff    : {diff.mean().item():.6g}\n  argmax agreement : {argmax_match}/{n}\n  first diff pos   : {first_diff_pos} (out of {n}Check the flush control path and rerun the regression.')
