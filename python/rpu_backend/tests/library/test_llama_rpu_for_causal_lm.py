from __future__ import annotations
import argparse
import gc
import json as _json
import os
import sys
from pathlib import Path
import pytest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from rpu_backend import RPUModelForCausalLM, RPUCache
from rpu_backend.api.errors import UnsupportedModelError
from rpu_backend.model_registry import model_path
pytestmark = pytest.mark.skipif(os.environ.get('RPU_HW_AVAILABLE') != '1', reason='requires device hardware (set RPU_HW_AVAILABLE=1)')
_MODEL_KEY = 'llama-3.2-1b'
_HF_NAME = 'unsloth/Llama-3.2-1B'
_PROMPT = 'The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. How vexingly quick daft zebras jump. Sphinx of black quartz, judge my vow! Two driven jocks help fax my big quiz. The five boxing wizards jump quickly. Jackdaws love my big sphinx of quartz.'

def _compute_local_fingerprint(model_dir: str) -> dict:
    import hashlib
    md = Path(model_dir)
    out = {}
    for name in ('config.json', 'tokenizer.json'):
        p = md / name
        if not p.exists():
            raise FileNotFoundError(f'Llama fingerprint: {p} missing')
        out[f"{name.split('.')[0]}_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    weights_index = md / 'model.safetensors.index.json'
    weights_single = md / 'model.safetensors'
    if weights_index.exists():
        idx_bytes = weights_index.read_bytes()
        out['weights_sha256'] = hashlib.sha256(idx_bytes).hexdigest()[:16]
        try:
            idx = _json.loads(idx_bytes)
            shard_files = sorted(set((idx.get('weight_map') or {}).values()))
            shards = []
            for shard_name in shard_files:
                sp = md / shard_name
                if sp.exists():
                    shards.append((shard_name, sp.stat().st_size))
                else:
                    shards.append((shard_name, -1))
            manifest_blob = _json.dumps(shards, sort_keys=True).encode('utf-8')
            out['weights_shard_manifest_sha256'] = hashlib.sha256(manifest_blob).hexdigest()[:16]
        except (_json.JSONDecodeError, KeyError, OSError) as e:
            out['weights_shard_manifest_sha256'] = f'MANIFEST_ERR:{type(e).__name__}'
    elif weights_single.exists():
        out['weights_sha256'] = hashlib.sha256(weights_single.read_bytes()).hexdigest()[:16]
        single_manifest = _json.dumps([('model.safetensors', weights_single.stat().st_size)], sort_keys=True).encode('utf-8')
        out['weights_shard_manifest_sha256'] = hashlib.sha256(single_manifest).hexdigest()[:16]
    else:
        raise FileNotFoundError(f'Llama fingerprint: neither model.safetensors.index.json nor model.safetensors at {md} (R1-03: .bin pickle weights are NOT accepted; the test requires safetensors-only weights for security defense-in-depth).')
    return out

def _resolve_model_dir() -> tuple[str, str, dict]:
    env_override = os.environ.get('RPU_LLAMA_MODEL_PATH')
    if env_override:
        model_dir = env_override
    else:
        p = model_path(_MODEL_KEY)
        if not (p / 'config.json').exists():
            pytest.skip(f'Llama-3.2-1B weights not found at {p} — fetch it into $RPU_MODEL_CACHE via the normal HuggingFace flow first.')
        model_dir = str(p)
    sha = 'UNPINNED'
    try:
        refs_main = Path(model_dir) / '.cache' / 'huggingface' / 'refs' / 'main'
        if refs_main.exists():
            sha = refs_main.read_text().strip()[:12]
    except Exception:
        pass
    try:
        fingerprint = _compute_local_fingerprint(model_dir)
    except FileNotFoundError as e:
        pytest.skip(f'Llama-3.2-1B model files incomplete: {e}. Re-fetch it into $RPU_MODEL_CACHE via the normal HuggingFace flow.')
        fingerprint = {}
    return (model_dir, sha, fingerprint)

@torch.no_grad()
def _run_cpu_prefill(model_cpu, input_ids):
    out = model_cpu(input_ids=input_ids, use_cache=False)
    logits = out.logits.float()
    argmaxes = logits[0].argmax(dim=-1).tolist()
    return (logits, argmaxes)

@torch.no_grad()
def _run_rpu_prefill(model_rpu, input_ids_cpu):
    device = torch.device('rpu')
    ids_rpu = input_ids_cpu.to(device)
    cache = RPUCache.from_model(model_rpu, input_ids=ids_rpu, max_new_tokens=4)
    out = model_rpu(input_ids=ids_rpu, past_key_values=cache, use_cache=True)
    assert out.logits.dtype == torch.float16, f'REN-05: expected device logits dtype=float16, got {out.logits.dtype}'
    logits = out.logits.cpu().float()
    argmaxes = logits[0].argmax(dim=-1).tolist()
    return (logits, argmaxes, cache)

@torch.no_grad()
def _run_rpu_prefill_with_cache(model_rpu, input_ids_cpu, cache):
    device = torch.device('rpu')
    ids_rpu = input_ids_cpu.to(device)
    out = model_rpu(input_ids=ids_rpu, past_key_values=cache, use_cache=True)
    assert out.logits.dtype == torch.float16, f'BC2-01 Run 2: expected fp16 logits, got {out.logits.dtype}'
    logits = out.logits.cpu().float()
    argmaxes = logits[0].argmax(dim=-1).tolist()
    return (logits, argmaxes, cache)

@torch.no_grad()
def _run_rpu_decode_step(model_rpu, last_token_id_cpu, cache, *, expected_position):
    device = torch.device('rpu')
    ids = torch.tensor([[last_token_id_cpu]], dtype=torch.long, device=device)
    actual_position = getattr(cache, 'position', None)
    assert actual_position is not None, 'CR3-WC2-01: RPUCache does NOT expose .position public attribute. RPUCache.position is part of the public API contract (api/cache.py:114,129,134,142).'
    assert actual_position == expected_position, f'WC2-01 (a): cache.position {actual_position} != expected {expected_position}'
    out = model_rpu(input_ids=ids, past_key_values=cache, use_cache=True)
    decode_dtype_correct = out.logits.dtype == torch.float16
    assert decode_dtype_correct, f'WC2-01 (b): device decode logits dtype must be fp16, got {out.logits.dtype}'
    logits_cpu = out.logits.cpu().float()
    assert not torch.isnan(logits_cpu).any() and (not torch.isinf(logits_cpu).any()), 'WC2-01 (c): NaN/Inf detected in device decode logits'
    return (logits_cpu, int(logits_cpu[0, -1].argmax().item()), decode_dtype_correct)

@torch.no_grad()
def _run_cpu_decode_step(model_cpu, full_input_ids):
    out = model_cpu(input_ids=full_input_ids, use_cache=False)
    last_logits = out.logits[:, -1:, :].float()
    return (last_logits, int(last_logits[0, 0].argmax().item()))

def _read_resolved_chunk_size(model_rpu) -> int:
    inner_decoder = model_rpu.model if hasattr(model_rpu, 'model') else model_rpu
    handle = getattr(inner_decoder, '_rpu_decoder_handle', None)
    assert handle is not None, "REN-05: inner_decoder._rpu_decoder_handle is unset — model.to('rpu') must be called before _read_resolved_chunk_size"
    return int(torch.ops.rpu.causal_decoder_get_resolved_chunk_size(handle))

@pytest.fixture(scope='module')
def llama_run_data():
    (model_dir, model_revision_sha, fingerprint) = _resolve_model_dir()
    seq_len = 64
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    enc = tokenizer(_PROMPT, return_tensors='pt', padding=False, truncation=True, max_length=seq_len)
    input_ids = enc.input_ids[:, :seq_len]
    assert input_ids.shape[1] == seq_len, f'B-03 invariant: deterministic prompt must produce exactly {seq_len} tokens; got {input_ids.shape[1]}.'
    print(f'[CPU] Loading Llama-3.2-1B fp16 from {model_dir} (mirror SHA: {model_revision_sha}; local fingerprint: {fingerprint})...')
    model_cpu = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, device_map='cpu', low_cpu_mem_usage=True, local_files_only=True, trust_remote_code=False).eval()
    (cpu_logits, cpu_argmaxes) = _run_cpu_prefill(model_cpu, input_ids)
    cpu_decode_argmax = int(cpu_argmaxes[-1])
    extended_ids = torch.cat([input_ids, torch.tensor([[cpu_decode_argmax]], dtype=torch.long)], dim=1)
    (cpu_decode_logits, cpu_step2_argmax) = _run_cpu_decode_step(model_cpu, extended_ids)
    del model_cpu
    gc.collect()
    print(f'[device] BC2-01 Step 1 — auto-pick prefill...')
    model_rpu = RPUModelForCausalLM.from_pretrained(model_dir, dtype=torch.float16, local_files_only=True, trust_remote_code=False).to('rpu')
    (rpu_logits, rpu_argmaxes, rpu_cache) = _run_rpu_prefill(model_rpu, input_ids)
    chunk_size_resolved = _read_resolved_chunk_size(model_rpu)
    max_diff_auto = float((cpu_logits - rpu_logits).abs().max().item())
    rpu_decode_argmax_step1 = int(rpu_argmaxes[-1])
    (rpu_decode_logits, rpu_step2_argmax, decode_dtype_correct) = _run_rpu_decode_step(model_rpu, rpu_decode_argmax_step1, rpu_cache, expected_position=seq_len)
    decode_max_diff = float((cpu_decode_logits - rpu_decode_logits).abs().max().item())
    decode_chunk_size_resolved = _read_resolved_chunk_size(model_rpu)
    print(f'[device] BC2-01 Step 2 — repeatability prefill (same model, fresh cache)...')
    rpu_cache_run2 = RPUCache.from_model(model_rpu, input_ids=input_ids.to('rpu'), max_new_tokens=4)
    (rpu_logits_run2, rpu_argmaxes_run2, _) = _run_rpu_prefill_with_cache(model_rpu, input_ids, rpu_cache_run2)
    chunk_size_run_2 = _read_resolved_chunk_size(model_rpu)
    max_diff_run_2 = float((cpu_logits - rpu_logits_run2).abs().max().item())
    argmax_match_count_run_2 = sum((1 for (a, b) in zip(cpu_argmaxes, rpu_argmaxes_run2) if a == b))
    print(f'[device] BC2-01 Step 3 — explicit-pin parity (re-instantiate, pin chunk_size={chunk_size_resolved})...')
    del model_rpu, rpu_cache, rpu_cache_run2
    gc.collect()
    pin_value = chunk_size_resolved
    model_rpu_pinned = RPUModelForCausalLM.from_pretrained(model_dir, dtype=torch.float16, local_files_only=True, trust_remote_code=False, rpu_execution={'prefill': {'chunk_size': pin_value}}).to('rpu')
    (rpu_logits_pinned, rpu_argmaxes_pinned, _) = _run_rpu_prefill(model_rpu_pinned, input_ids)
    chunk_size_after_explicit_pin = _read_resolved_chunk_size(model_rpu_pinned)
    max_diff_explicit = float((cpu_logits - rpu_logits_pinned).abs().max().item())
    logits_match = bool(torch.allclose(rpu_logits_pinned, rpu_logits, atol=0.0, rtol=0.0))
    argmax_match = bool(rpu_argmaxes_pinned == rpu_argmaxes)
    chunk_size_explicit_pin_ok = bool(chunk_size_after_explicit_pin == chunk_size_resolved and logits_match and argmax_match)
    n = seq_len
    match_count = sum((1 for (a, b) in zip(cpu_argmaxes, rpu_argmaxes) if a == b))
    max_diff = max_diff_auto
    has_nan_or_inf = bool(torch.isnan(rpu_logits).any().item() or torch.isinf(rpu_logits).any().item())
    decode_argmax_match = bool(rpu_step2_argmax == cpu_step2_argmax)
    result = {'size': _MODEL_KEY, 'seq_len': int(n), 'argmax_match_count': int(match_count), 'argmax_correct_count': int(match_count), 'top1_match': bool(match_count == n), 'max_diff': float(max_diff), 'no_nan_or_inf': not has_nan_or_inf, 'logits_argmax_first': int(rpu_argmaxes[0]), 'logits_argmax_last': int(rpu_argmaxes[-1]), 'decode_argmax_match': decode_argmax_match, 'decode_max_diff': float(decode_max_diff), 'decode_dtype_correct': bool(decode_dtype_correct), 'decode_chunk_size_resolved': int(decode_chunk_size_resolved), 'auto_chunk_size': int(chunk_size_resolved), 'chunk_size_resolved': int(chunk_size_resolved), 'chunk_size_run_2': int(chunk_size_run_2), 'chunk_size_explicit_pin_ok': bool(chunk_size_explicit_pin_ok), 'max_diff_run_2': float(max_diff_run_2), 'max_diff_explicit': float(max_diff_explicit), 'argmax_match_count_run_2': int(argmax_match_count_run_2), 'explicit_pin_logits_match': logits_match, 'explicit_pin_argmax_match': argmax_match, 'model_revision_sha': str(model_revision_sha), 'config_sha256': str(fingerprint.get('config_sha256', '')), 'tokenizer_sha256': str(fingerprint.get('tokenizer_sha256', '')), 'weights_sha256': str(fingerprint.get('weights_sha256', '')), 'weights_shard_manifest_sha256': str(fingerprint.get('weights_shard_manifest_sha256', ''))}
    print(f'RESULT: {_json.dumps(result, sort_keys=True)}')
    yield {'n': n, 'match_count': match_count, 'max_diff': max_diff, 'has_nan_or_inf': has_nan_or_inf, 'decode_argmax_match': decode_argmax_match, 'decode_max_diff': decode_max_diff, 'decode_dtype_correct': decode_dtype_correct, 'chunk_size_resolved': chunk_size_resolved, 'chunk_size_run_2': chunk_size_run_2, 'chunk_size_explicit_pin_ok': chunk_size_explicit_pin_ok, 'argmax_match_count_run_2': argmax_match_count_run_2, 'logits_match': logits_match, 'argmax_match': argmax_match, 'result': result}

def test_llama_3_2_1b_chunk_size_observable(llama_run_data):
    assert llama_run_data['chunk_size_resolved'] > 0, f"TASK-1.5: _read_resolved_chunk_size returned {llama_run_data['chunk_size_resolved']}; expected > 0. Either compute_chunks_impl didn't write last_resolved_chunk_size_, or binary search failed."

def test_llama_3_2_1b_chunk_size_repeatability(llama_run_data):
    assert llama_run_data['chunk_size_run_2'] == llama_run_data['chunk_size_resolved'], f"BC2-01: chunk_size drift {llama_run_data['chunk_size_resolved']} → {llama_run_data['chunk_size_run_2']}"
    assert llama_run_data['argmax_match_count_run_2'] == llama_run_data['match_count'], f"BC2-01: argmax drift Run 1={llama_run_data['match_count']} vs Run 2={llama_run_data['argmax_match_count_run_2']}"

def test_llama_3_2_1b_explicit_pin_logits_parity(llama_run_data):
    assert llama_run_data['chunk_size_explicit_pin_ok'], f"BC2-01 Step 3 (R1-02): explicit-pin numerics differ from auto-pick. Same chunk_size in both runs MUST produce bit-exact logits. Drift = bug. chunk_size_resolved={llama_run_data['chunk_size_resolved']}; logits_match={llama_run_data['logits_match']}; argmax_match={llama_run_data['argmax_match']}. See RESULT JSON keys explicit_pin_logits_match + explicit_pin_argmax_match."

def test_llama_3_2_1b_cache_position_invariant(llama_run_data):
    assert llama_run_data is not None

def test_llama_3_2_1b_dtype_and_no_nan(llama_run_data):
    assert llama_run_data['decode_dtype_correct'], 'WC2-01 (b): decode dtype != fp16'
    assert not llama_run_data['has_nan_or_inf'], 'REN-05: NaN/Inf in device prefill logits'

@pytest.mark.xfail(reason='Source prefill numerical parity remains a strict expected failure.', strict=True)
def test_llama_3_2_1b_prefill_numerical_parity(llama_run_data):
    n = llama_run_data['n']
    match_count = llama_run_data['match_count']
    max_diff = llama_run_data['max_diff']
    assert match_count == n, f'REN-05 prefill: argmax_correct_count {match_count} != seq_len {n}'
    assert max_diff <= 0.05, f'REN-05 prefill: max_diff {max_diff:.4f} > 0.05 threshold'

@pytest.mark.xfail(reason='Source decode numerical parity remains a strict expected failure.', strict=True)
def test_llama_3_2_1b_decode_numerical_parity(llama_run_data):
    decode_argmax_match = llama_run_data['decode_argmax_match']
    decode_max_diff = llama_run_data['decode_max_diff']
    assert decode_argmax_match, 'REN-05 decode: decode_argmax_match == False'
    assert decode_max_diff <= 0.05, f'REN-05 decode: decode_max_diff {decode_max_diff:.4f} > 0.05 threshold'

def _main_cli() -> int:
    parser = argparse.ArgumentParser(description='Llama-3.2-1B device prefill+decode harness (REN-05 + W-01) — split into 7 tests (5 PASS + 2 XFAIL strict per R1-01)')
    parser.add_argument('--model', default=None, help='Override model dir (defaults to model_registry).')
    args = parser.parse_args()
    if args.model:
        os.environ['RPU_LLAMA_MODEL_PATH'] = args.model
    import pytest as _pytest
    return _pytest.main([__file__, '-q', '-s'])
if __name__ == '__main__':
    sys.exit(_main_cli())
