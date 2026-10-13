from __future__ import annotations
import sys

def test_install_causal_decoder_forward_in_runtime_decoder():
    from rpu_backend.runtime.decoder import _install_causal_decoder_forward
    import inspect
    sig = inspect.signature(_install_causal_decoder_forward)
    assert 'arch' in sig.parameters, f'arch keyword missing: {sig}'

def test_run_causal_decoder_forward_in_runtime_decoder():
    from rpu_backend.runtime.decoder import _run_causal_decoder_forward
    import inspect
    sig = inspect.signature(_run_causal_decoder_forward)
    assert 'model' in sig.parameters
    assert 'handle' in sig.parameters

def test_chunk_size_helper_in_runtime_decoder():
    from rpu_backend.runtime.decoder import chunk_policy_key
    assert callable(chunk_policy_key)

def test_qwen3_class_patch_sentinels_set():
    import rpu_backend.adapters.qwen3
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3RotaryEmbedding
    assert getattr(Qwen3RMSNorm, '_rpu_patched_rmsnorm', False), 'Qwen3RMSNorm._rpu_patched_rmsnorm sentinel not set after adapters/qwen3.py import'
    assert getattr(Qwen3RotaryEmbedding, '_rpu_patched_rotary', False), 'Qwen3RotaryEmbedding._rpu_patched_rotary sentinel not set after adapters/qwen3.py import'

def test_llama_class_patch_sentinels_set():
    import rpu_backend.adapters.llama
    from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding
    assert getattr(LlamaRMSNorm, '_rpu_patched_rmsnorm', False), 'LlamaRMSNorm._rpu_patched_rmsnorm sentinel not set after adapters/llama.py import'
    assert getattr(LlamaRotaryEmbedding, '_rpu_patched_rotary', False), 'LlamaRotaryEmbedding._rpu_patched_rotary sentinel not set after adapters/llama.py import'

def test_no_internal_imports_at_module_load():
    import rpu_backend.adapters.qwen3
    import rpu_backend.adapters.llama
    leaked = [m for m in sys.modules if m.startswith('rpu_backend._internal')]
    assert not leaked, f'_internal modules leaked: {leaked}'

def test_idempotency_guard_fires_on_repeated_call():
    import rpu_backend.adapters.qwen3 as qwen3_mod
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
    forward_before = Qwen3RMSNorm.forward
    qwen3_mod._idempotent_patch_qwen3_rmsnorm(Qwen3RMSNorm)
    assert Qwen3RMSNorm.forward is forward_before, 'Idempotency guard FAILED: Qwen3RMSNorm.forward was re-patched on second call'
