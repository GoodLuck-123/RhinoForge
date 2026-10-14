from __future__ import annotations
import argparse
import os
import sys
import torch
SEED = 42

def parse_args(argv: list[str] | None=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Capture device prefill+decode logits for bit-match diffing.')
    p.add_argument('mode', choices=['flush_on', 'flush_off'])
    p.add_argument('out_path')
    p.add_argument('--size', default='0.6b')
    p.add_argument('--prefill', type=int, default=256)
    p.add_argument('--decode', type=int, default=16)
    p.add_argument('--chunk-size', type=int, default=0)
    return p.parse_args(argv)

def main(argv: list[str] | None=None) -> int:
    args = parse_args(argv)
    if args.mode == 'flush_on':
        (ddr_flush, ddr_flush_force) = (True, True)
    else:
        (ddr_flush, ddr_flush_force) = (False, True)
    import rpu_backend
    from rpu_backend import RPUModelForCausalLM, RPUCache
    from rpu_backend.model_registry import model_path
    from transformers import AutoTokenizer
    torch.rpu.set_caching_allocator(True)
    torch.rpu.set_ddr_flush(ddr_flush)
    torch.rpu.set_ddr_flush_force(ddr_flush_force)
    print(f'[{args.mode}] ddr_flush={torch.rpu.get_ddr_flush()}, ddr_flush_force={torch.rpu.get_ddr_flush_force()}, chunk_size={args.chunk_size}, size={args.size}, prefill={args.prefill}, decode={args.decode}', flush=True)
    torch.manual_seed(SEED)
    cache_dir = str(model_path(f'qwen3-{args.size}'))
    tok = AutoTokenizer.from_pretrained(cache_dir, cache_dir=os.environ.get('HF_HOME'))
    model = RPUModelForCausalLM.from_pretrained(cache_dir).to('rpu')
    model.eval()
    prompt = 'The quick brown fox jumps over the lazy dog. ' * 4
    input_ids = tok(prompt, return_tensors='pt').input_ids[:, :args.prefill]
    if input_ids.shape[1] < args.prefill:
        pad = args.prefill - input_ids.shape[1]
        pad_id = tok.pad_token_id or tok.eos_token_id
        input_ids = torch.nn.functional.pad(input_ids, (0, pad), value=pad_id)
    cache = RPUCache.from_model(model, max_seq_len=args.prefill + args.decode + 8)
    ids_rpu = input_ids.to('rpu')
    _ = model.model(input_ids=ids_rpu, past_key_values=cache, use_cache=True)
    cache.reset()
    ids_rpu = input_ids.to('rpu')
    inner_out = model.model(input_ids=ids_rpu, past_key_values=cache, use_cache=True)
    first_logits = model.lm_head(inner_out.last_hidden_state[:, -1:, :])
    next_id = first_logits.argmax(-1)
    captured = [first_logits.detach()]
    for _ in range(args.decode):
        out = model(input_ids=next_id.to('rpu'), past_key_values=cache, use_cache=True)
        captured.append(out.logits[:, -1:, :].detach())
        next_id = out.logits[:, -1:, :].argmax(-1)
    logits = torch.cat(captured, dim=1).to('cpu', dtype=torch.float32)
    torch.save({'mode': args.mode, 'ddr_flush': ddr_flush, 'ddr_flush_force': ddr_flush_force, 'size': args.size, 'prefill': args.prefill, 'decode': args.decode, 'chunk_size': args.chunk_size, 'logits': logits, 'input_ids': input_ids}, args.out_path)
    print(f'[{args.mode}] saved logits shape={tuple(logits.shape)} dtype={logits.dtype} → {args.out_path}', flush=True)
    return 0
if __name__ == '__main__':
    sys.exit(main())
