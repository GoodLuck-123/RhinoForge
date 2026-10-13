import pytest
import torch
import rpu_backend
from rpu_backend import RPUCache, RPUModelForCausalLM
from rpu_backend.model_registry import model_path
pytestmark = pytest.mark.skipif(not model_path('qwen3-0.6b').exists(), reason='qwen3-0.6b not in model cache; set RPU_MODEL_CACHE to its parent dir')

@pytest.fixture(scope='module')
def model_and_cache():
    model = RPUModelForCausalLM.from_pretrained(str(model_path('qwen3-0.6b')), dtype=torch.float16, device='rpu', local_files_only=True)
    torch.ops.rpu.causal_decoder_set_chunk_envelope(model.model._rpu_decoder_handle, 4096, 960)
    cache = RPUCache.from_model(model, max_seq_len=2048)
    return (model, cache)

def _set_chunk(model, cs):
    torch.ops.rpu.causal_decoder_set_chunk_size_override(model.model._rpu_decoder_handle, cs)

@pytest.fixture(autouse=True)
def reset_override(model_and_cache):
    (model, _) = model_and_cache
    _set_chunk(model, 0)
    yield
    _set_chunk(model, 0)

def _run(model, cache, seq_len: int):
    cache.reset()
    input_ids = torch.zeros((1, seq_len), dtype=torch.long).to('rpu')
    with torch.no_grad():
        return model(input_ids=input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)

def _resolve(model, seq_len: int):
    return torch.ops.rpu.causal_decoder_resolve_prefill_chunk_size(model.model._rpu_decoder_handle, seq_len, 0)

def test_chunk_override_cs80_NOW_VALID_under_runtime_alignment(model_and_cache):
    (model, cache) = model_and_cache
    _set_chunk(model, 80)
    out = _run(model, cache, seq_len=80)
    assert out is not None

def test_chunk_override_sync_rejected(model_and_cache):
    (model, _) = model_and_cache
    _set_chunk(model, 640)
    with pytest.raises(RuntimeError, match='chunk_size_override'):
        _resolve(model, seq_len=640)

def test_chunk_override_sync_rejected_modulo_8(model_and_cache):
    (model, _) = model_and_cache
    _set_chunk(model, 960)
    with pytest.raises(RuntimeError, match='chunk_size_override'):
        _resolve(model, seq_len=4096)

def test_chunk_override_valid_passes(model_and_cache):
    (model, cache) = model_and_cache
    _set_chunk(model, 128)
    out = _run(model, cache, seq_len=128)
    assert out is not None
