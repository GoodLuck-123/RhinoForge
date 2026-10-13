import pytest
import torch
import rpu_backend
pytestmark = pytest.mark.hardware

def _rpu_hw_probe() -> bool:
    try:
        torch.zeros(1, device='rpu')
        return True
    except Exception:
        return False

def test_build_matches_replay(request):
    if request.session.items and request.session.items[0] is not request.node:
        pytest.skip('not first in this pytest process — this reproducer needs the COLD first-in-process device-VA layout and passes for the wrong reason from any other position. Run this file on its own (see the dedicated invocation in .gitlab-ci.yml).')
    if not _rpu_hw_probe():
        pytest.skip('needs device device')
    from transformers import AutoTokenizer
    from rpu_backend import RPUModelForCausalLM, RPUCache
    from rpu_backend.model_registry import model_path
    path = model_path('qwen3-0.6b')
    if not path.exists():
        pytest.skip(f'missing model cache: {path}')
    tok = AutoTokenizer.from_pretrained(str(path))
    ids = tok('The quick brown fox jumps over the lazy dog and then keeps on running through the quiet field until the sun goes down at last.', return_tensors='pt').input_ids[:, :10]
    rpu = RPUModelForCausalLM.from_pretrained(str(path), dtype=torch.float16).to('rpu')
    cache = RPUCache.from_model(rpu, max_seq_len=16)
    outs = []
    for i in range(3):
        if i:
            cache.reset_to_position(0)
        with torch.no_grad():
            outs.append(rpu(ids.to('rpu'), past_key_values=cache).logits.cpu())
    assert torch.equal(outs[1], outs[2]), 'two REPLAYs disagree — different defect'
    assert torch.equal(outs[0], outs[1]), 'BUILD (forward #1) differs from REPLAY (forward #2): the first forward of this shape returns a different answer than every subsequent one. Regression of the hidden_states use-after-free — check that FusedModelBase::run_all_layers still calls RpuKernelGraph::active().keep_alive(hidden_states) next to the hidden_in_src_base_ assignment (a device address is not a reference)'

@pytest.mark.parametrize(('n_tok', 'cache_rows', 'expect_padding'), [pytest.param(16, 16, False, id='unpadded-same-capacity-control'), pytest.param(10, 32, True, id='padded-different-layout-control')])
def test_padded_prefill_matches_cpu(n_tok, cache_rows, expect_padding):
    if not _rpu_hw_probe():
        pytest.skip('needs device device')
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from rpu_backend import RPUModelForCausalLM, RPUCache
    from rpu_backend.model_registry import model_path
    path = model_path('qwen3-0.6b')
    if not path.exists():
        pytest.skip(f'missing model cache: {path}')
    path = str(path)
    tok = AutoTokenizer.from_pretrained(path)
    ids = tok('The quick brown fox jumps over the lazy dog and then keeps on running through the quiet field until the sun goes down at last.', return_tensors='pt').input_ids[:, :n_tok]
    assert ids.shape[1] == n_tok
    assert (n_tok % 16 != 0) == expect_padding, 'shape sanity: padding expected?'
    cpu = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16).eval()
    with torch.no_grad():
        cpu_logits = cpu(ids).logits.detach()
    del cpu
    rpu = RPUModelForCausalLM.from_pretrained(path, dtype=torch.float16).to('rpu')
    cache = RPUCache.from_model(rpu, max_seq_len=cache_rows)
    with torch.no_grad():
        rpu_logits = rpu(ids.to('rpu'), past_key_values=cache).logits.cpu()
    cos = torch.nn.functional.cosine_similarity(rpu_logits.float().flatten().unsqueeze(0), cpu_logits.float().flatten().unsqueeze(0)).item()
    assert cos >= 0.999, f'n_tok={n_tok} cache_rows={cache_rows} (sKeyVx={-(-cache_rows // 16)}) cos={cos:.6f}'
