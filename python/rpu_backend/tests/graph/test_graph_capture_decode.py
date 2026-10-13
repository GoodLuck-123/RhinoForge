import argparse
import os
import sys
import time
import gc
import ctypes
import torch
import torch.nn as nn
import rpu_backend
from rpu_backend import RPUCache
from rpu_backend.runtime.decoder import _install_rmsnorm_class_swap as patch_rmsnorm_class, _install_rotary_class_swap as patch_rotary_embedding, _install_causal_decoder_forward
from rpu_backend.runtime.weights import _collect_embedding_data_ptrs, get_linear_partition, transform_linear_weight
from rpu_backend.model_registry import model_path
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3RotaryEmbedding
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding
from rpu_backend.adapters.qwen3 import lookup_causal_decoder
MODEL_NAME_MAP = {'qwen3-0.6b': 'Qwen/Qwen3-0.6B', 'qwen3-1.7b': 'Qwen/Qwen3-1.7B', 'qwen3-4b': 'Qwen/Qwen3-4B', 'qwen3-8b': 'Qwen/Qwen3-8B', 'llama-3.2-1b': 'meta-llama/Llama-3.2-1B'}
ARCH_MAP = {'qwen3-0.6b': 'qwen3', 'qwen3-1.7b': 'qwen3', 'qwen3-4b': 'qwen3', 'qwen3-8b': 'qwen3', 'llama-3.2-1b': 'llama'}
MODEL_CONFIG = {'name': 'Qwen/Qwen3-0.6B', 'cache_dir': str(model_path('qwen3-0.6b')), 'arch': 'qwen3'}
DECODE_STEPS = 16
TEXT = 'Hello from Hugging Face and PyTorch! This is a test for Qwen model.'

def setup_rpu():
    torch.rpu.set_profile(False)
    torch.rpu.set_caching_allocator(True)
    torch.rpu.set_ddr_flush(True)
    torch.rpu.set_debug_export(False)
    torch.rpu.set_spm_debug(False)
    torch.rpu.set_cross_layer_batch_prefill(True)
    torch.rpu.set_cross_layer_batch_size(36)

def patch_classes():
    arch = MODEL_CONFIG.get('arch', 'qwen3')
    if arch == 'qwen3':
        patch_rmsnorm_class(Qwen3RMSNorm)
        patch_rotary_embedding(Qwen3RotaryEmbedding)
    elif arch == 'llama':
        patch_rmsnorm_class(LlamaRMSNorm)
        patch_rotary_embedding(LlamaRotaryEmbedding)
    else:
        raise ValueError(f'patch_classes: unsupported arch={arch!r}')

def load_model():
    (name, cache_dir) = (MODEL_CONFIG['name'], MODEL_CONFIG['cache_dir'])
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
    tokenizer = AutoTokenizer.from_pretrained(cache_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(cache_dir, torch_dtype=torch.float16, device_map='cpu', low_cpu_mem_usage=True, local_files_only=True)
    model.eval()
    return (tokenizer, model)

def convert_to_rpu(model):
    device = torch.device('rpu')
    embedding_ptrs = _collect_embedding_data_ptrs(model)

    def _conv_move(module):
        for (n, child) in module.named_modules():
            if isinstance(child, nn.Linear):
                if child.weight.data.device != torch.device('cpu'):
                    continue
                (in_f, out_f) = (child.in_features, child.out_features)
                leaf = n.split('.')[-1]
                part = 0 if leaf in ('o_proj', 'down_proj') else get_linear_partition(in_f, out_f, child.weight.element_size())
                if part >= 0:
                    is_tied = child.weight.data_ptr() in embedding_ptrs
                    with torch.no_grad():
                        if is_tied:
                            child.weight = nn.Parameter(child.weight.data.clone())
                        child.weight.data = transform_linear_weight(child.weight.data, part)
        module.to(device)

    def _free():
        gc.collect()
        try:
            ctypes.CDLL('libc.so.6').malloc_trim(0)
        except:
            pass
    backbone = model.model
    if hasattr(backbone, 'embed_tokens'):
        backbone.embed_tokens.to(device)
        _free()
    if hasattr(backbone, 'layers'):
        for (i, layer) in enumerate(backbone.layers):
            _conv_move(layer)
            _free()
    for (name, child) in backbone.named_children():
        if name not in ('embed_tokens', 'layers'):
            _conv_move(child)
            _free()
    for (_, p) in list(backbone.named_parameters(recurse=False)):
        p.data = p.data.to(device)
    for (_, b) in list(backbone.named_buffers(recurse=False)):
        b.data = b.data.to(device)
    for (name, child) in model.named_children():
        if name != 'model':
            _conv_move(child)
            _free()
    for (_, p) in list(model.named_parameters(recurse=False)):
        p.data = p.data.to(device)
    for (_, b) in list(model.named_buffers(recurse=False)):
        b.data = b.data.to(device)
    _free()
    model.eval()
    return model

def _make_decode_sig(hidden: int) -> 'rpu_backend.graph.GraphSignature':
    return rpu_backend.graph.GraphSignature(op_id='qwen3_decode', shapes=[1, 1], dyn_dims=[hidden], dtypes=[torch.float16])

@torch.no_grad()
def run_decode(model, input_ids, decode_steps, use_graph=False):
    device = torch.device('rpu')
    config = model.config
    head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
    seq_len = input_ids.shape[1]
    backbone = model.model
    lm_head = model.lm_head
    ids_rpu = input_ids.to(device)
    cache = RPUCache(num_layers=config.num_hidden_layers, batch_size=1, max_seq_len=seq_len + decode_steps + 16, num_kv_heads=config.num_key_value_heads, head_dim=head_dim, device='rpu', dtype=torch.float16)
    mode_str = 'GRAPH' if use_graph else 'PASSTHROUGH'
    gcache = rpu_backend.graph.GraphCache() if use_graph else None
    sig = _make_decode_sig(hidden=config.hidden_size) if use_graph else None
    cache.reset()
    t0 = time.perf_counter()
    outputs = backbone(input_ids=ids_rpu, past_key_values=cache, use_cache=True)
    logits = lm_head(outputs.last_hidden_state[:, -1:, :])
    t_prefill = (time.perf_counter() - t0) * 1000
    next_id = logits[0, -1, :].cpu().float().argmax().item()
    token_ids = [next_id]
    print(f'  [{mode_str}] Prefill: {t_prefill:.1f} ms')
    state_log = []
    last_graph_handle = None
    decode_times = []
    body_times = []
    end_times = []
    lm_head_times = []
    for step in range(decode_steps - 1):
        inp = torch.tensor([[next_id]], dtype=torch.long).to(device)
        t_step_start = time.perf_counter()
        if use_graph:
            with gcache.capture(sig) as g:
                in_scope_state = g.state()
                outputs = backbone(input_ids=inp, past_key_values=cache, use_cache=True)
                t_body_end = time.perf_counter()
            t_end_done = time.perf_counter()
            state_log.append(in_scope_state)
            last_graph_handle = g
        else:
            outputs = backbone(input_ids=inp, past_key_values=cache, use_cache=True)
            t_body_end = time.perf_counter()
            t_end_done = t_body_end
        t_lm_start = time.perf_counter()
        logits = lm_head(outputs.last_hidden_state[:, -1:, :])
        t_lm_end = time.perf_counter()
        decode_times.append((t_lm_end - t_step_start) * 1000)
        body_times.append((t_body_end - t_step_start) * 1000)
        end_times.append((t_end_done - t_body_end) * 1000)
        lm_head_times.append((t_lm_end - t_lm_start) * 1000)
        next_id = logits[0, -1, :].cpu().float().argmax().item()
        token_ids.append(next_id)
    avg = sum(decode_times) / len(decode_times) if decode_times else 0
    print(f'  [{mode_str}] Decode avg: {avg:.1f} ms/tok')
    if decode_times:
        lm_head_avg = sum(lm_head_times) / len(lm_head_times)
        if use_graph and len(decode_times) >= 2:
            record_ms = decode_times[0]
            build_ms = end_times[0]
            replay_avg = sum(decode_times[1:]) / len(decode_times[1:])
            print(f'  [{mode_str}] record_ms (iter 0 full step):   {record_ms:.2f}')
            print(f'  [{mode_str}] build_ms  (iter 0 graph.end()): {build_ms:.2f}')
            print(f'  [{mode_str}] replay_avg (iter 1+ mean):      {replay_avg:.2f}')
            print(f'  [{mode_str}] lm_head_avg (all iters):        {lm_head_avg:.2f}')
        else:
            body_avg = sum(body_times) / len(body_times)
            print(f'  [{mode_str}] body_avg (backbone per step):   {body_avg:.2f}')
            print(f'  [{mode_str}] lm_head_avg (all iters):        {lm_head_avg:.2f}')
    if use_graph and last_graph_handle is not None:
        g = last_graph_handle
        state_names = {0: 'PASSTHROUGH', 1: 'RECORDING', 2: 'BUILT', 3: 'REPLAYING'}
        print(f'  [{mode_str}] Final: state={state_names.get(g.state(), g.state())}, graph_size={g.graph_size()}, replayable={g.replayable()}')
        log_str = [state_names.get(s, str(s)) for s in state_log]
        print(f'  [{mode_str}] In-scope state per step: {log_str[:5]}...')
        print(f'  [{mode_str}] GraphCache size: {gcache.size()}')
    return token_ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--decode-steps', type=int, default=DECODE_STEPS)
    parser.add_argument('--model', type=str, default='qwen3-0.6b', help='qwen3-0.6b / qwen3-1.7b / qwen3-4b / qwen3-8b / llama-3.2-1b')
    args = parser.parse_args()
    arch = ARCH_MAP.get(args.model, 'qwen3')
    global MODEL_CONFIG
    MODEL_CONFIG = {'name': MODEL_NAME_MAP.get(args.model, f'Qwen/{args.model}'), 'cache_dir': str(model_path(args.model)), 'arch': arch}
    print('=' * 60)
    print(f'KernelGraph Capture/Replay Decode Test — {args.model} (arch={arch}, v5)')
    print(f'  decode_steps={args.decode_steps}')
    print('=' * 60)
    setup_rpu()
    patch_classes()
    (tokenizer, model) = load_model()
    model = convert_to_rpu(model)
    _install_causal_decoder_forward(model.model, arch=arch, chunk_envelope_for=lookup_causal_decoder)
    inputs = tokenizer(TEXT, return_tensors='pt')
    input_ids = inputs['input_ids']
    print(f'Input: "{TEXT}"')
    print(f'Input length: {input_ids.shape[1]}, Decode steps: {args.decode_steps}')
    print(f'\n--- Run 1: PASSTHROUGH (baseline) ---')
    baseline_ids = run_decode(model, input_ids, args.decode_steps, use_graph=False)
    print(f'\n--- Run 2: Graph capture/replay ---')
    graph_ids = run_decode(model, input_ids, args.decode_steps, use_graph=True)
    print(f"\n{'=' * 60}")
    print('Comparison')
    print(f"{'=' * 60}")
    match_count = sum((a == b for (a, b) in zip(baseline_ids, graph_ids)))
    total = len(baseline_ids)
    print(f'  Token match: {match_count}/{total}')
    print(f'  Baseline IDs: {baseline_ids}')
    print(f'  Graph    IDs: {graph_ids}')
    if match_count == total:
        print(f'\n  ALL tokens identical!')
        decoded_text = tokenizer.decode(baseline_ids, skip_special_tokens=True)
        print(f'  Text: {decoded_text}')
        print(f"\n{'=' * 60}")
        print('TEST PASSED')
        print(f"{'=' * 60}")
        return 0
    else:
        print(f'\n  First divergence:')
        print(f"  {'Step':>5} | {'Baseline':>10} | {'Graph':>10} | Match")
        print(f"  {'-' * 42}")
        for (i, (a, b)) in enumerate(zip(baseline_ids, graph_ids)):
            m = 'Y' if a == b else 'N'
            print(f'  {i:5d} | {a:10d} | {b:10d} |     {m}')
            if i >= 9:
                break
        print(f"\n{'=' * 60}")
        print(f'TEST FAILED — {total - match_count}/{total} tokens differ')
        print(f"{'=' * 60}")
        return 1
if __name__ == '__main__':
    sys.exit(main())
