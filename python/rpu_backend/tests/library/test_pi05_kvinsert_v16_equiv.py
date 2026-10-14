from __future__ import annotations
import os, random, subprocess, sys, tempfile
from rpu_backend.model_registry import model_path
from pathlib import Path
import pytest
pytestmark = pytest.mark.skipif(os.environ.get('RPU_HW_AVAILABLE') != '1', reason='requires device hardware (set RPU_HW_AVAILABLE=1 and run with sudo)')
REPO = Path(__file__).resolve().parents[4]
W4 = Path(os.environ.get('RPU_PI05_W4A16_MODEL_DIR', str(model_path('pi05-libero-finetuned-w4real-kvint8'))))
SEED = 1234

def _make_batch(path: Path) -> None:
    import numpy as np
    import torch
    from rpu_backend.tests.library.test_pi05_e2e import _prepare_dummy_inputs
    import lerobot.policies.pi05.modeling_pi05
    from lerobot.configs.policies import PreTrainedConfig
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    cfg = PreTrainedConfig.from_pretrained(W4)
    cfg.dtype = 'float32'
    cfg.device = 'cpu'
    torch.save(_prepare_dummy_inputs(cfg, 32), path)
_RUNNER = '\nimport os, random, torch, rpu_backend  # noqa\nimport numpy as np\nfrom rpu_backend.api import Pi05Policy\nimport lerobot.policies.pi05.modeling_pi05  # noqa\nW4 = os.environ["W4_DIR"]\nseed = int(os.environ.get("SEED", "1234"))\nrandom.seed(seed); np.random.seed(seed); torch.manual_seed(seed)\ntorch.rpu.set_caching_allocator(True); torch.rpu.set_ddr_flush(True)\nbatch = torch.load(os.environ["BATCH"], map_location="cpu")\npol = Pi05Policy.from_pretrained(W4, dtype=torch.float16).to("rpu")\npol._lerobot_policy.model._rpu_noise_seed = 0\na = pol._forward_rpu(batch, return_chunk=True, num_steps=5).cpu().float()\ntorch.save(a, os.environ["OUT"])\nprint("RAN", os.environ.get("RPU_KVINSERT_V16"))\n'

def _run(v16: str, batch: Path, out: Path) -> str:
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1', LKN_RPU_FREQ_MHZ='750', RPU_LOG_LEVEL='1', RPU_KVINSERT_V16=v16, RPU_KVINSERT_V16_TRACE='1', W4_DIR=str(W4), BATCH=str(batch), OUT=str(out), SEED=str(SEED))
    c = subprocess.run([sys.executable, '-c', _RUNNER], cwd=str(REPO), env=env, text=True, capture_output=True, timeout=900)
    assert c.returncode == 0 and out.is_file(), f'runner v16={v16} failed:\n{c.stdout[-2000:]}\n{c.stderr[-2000:]}'
    return c.stdout + c.stderr

def _trace_lines(log: str) -> list[str]:
    return [ln for ln in log.splitlines() if '[KVINSERT_TRACE]' in ln]

def test_kvinsert_v16_matches_v2() -> None:
    if not (W4 / 'config.json').is_file():
        pytest.skip(f'w4a16 checkpoint missing: {W4}')
    import torch
    with tempfile.TemporaryDirectory() as td:
        batch = Path(td) / 'batch.pt'
        (a16, a2) = (Path(td) / 'v16.pt', Path(td) / 'v2.pt')
        _make_batch(batch)
        log16 = _run('1', batch, a16)
        log2 = _run('0', batch, a2)
        x16 = torch.load(a16)
        x2 = torch.load(a2)
    (t16, t2) = (_trace_lines(log16), _trace_lines(log2))
    print('\n'.join(t16[:20]))
    assert any(('kind=K path=v16' in ln for ln in t16)), f'K v16 never fired:\n{log16[-4000:]}'
    assert any(('kind=V path=v16' in ln for ln in t16)), f'V v16 never fired:\n{log16[-4000:]}'
    assert not any(('path=v16' in ln for ln in t2)), f'kill-switch still hit v16:\n{log2[-4000:]}'
    assert any(('kind=K path=v2' in ln for ln in t2)), f'K v2 trace missing:\n{log2[-4000:]}'
    assert any(('kind=V path=v2' in ln for ln in t2)), f'V v2 trace missing:\n{log2[-4000:]}'
    assert any(('path=v2' in ln for ln in t16)), f'v16-on run never exercised fallback:\n{log16[-4000:]}'
    mse = float(((x16 - x2) ** 2).mean())
    maxd = float((x16 - x2).abs().max())
    print(f'KVINSERT v16-vs-v2 MSE={mse:.3e} max_abs={maxd:.3e}')
    assert torch.equal(x16, x2), f'v16 diverged from v2 on the same batch (MSE {mse:.3e}, max {maxd:.3e})'
