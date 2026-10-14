from __future__ import annotations
import sys
import torch
import torch.nn.functional as F
import rpu_backend
SEED = 20260702

def _check_output(name: str, x_u8: torch.Tensor, out: torch.Tensor) -> bool:
    ref = x_u8.float()
    assert out.dtype == torch.float16, f'{name}: expected fp16, got {out.dtype}'
    out_cpu = out.float().cpu()
    assert torch.isfinite(out_cpu).all(), f'{name}: non-finite output'
    assert out_cpu.shape == ref.shape, f'{name}: shape {tuple(out_cpu.shape)} != {tuple(ref.shape)}'
    cos = F.cosine_similarity(out_cpu.reshape(-1), ref.reshape(-1), dim=0).item()
    max_abs = (out_cpu - ref).abs().max().item()
    ok = max_abs == 0.0
    print(f"  [{name}] numel={x_u8.numel()} max_abs={max_abs:.3e} cos~{cos:.5f} -> {('PASS' if ok else 'FAIL')}")
    return ok

def _run_case(name: str, x_u8: torch.Tensor) -> bool:
    x_rpu = x_u8.to('rpu')
    ok = _check_output(name, x_u8, torch.ops.rpu.cast_uint8_fp16(x_rpu))
    out = torch.full_like(x_u8, -1, dtype=torch.float16).to('rpu')
    torch.ops.rpu.cast_uint8_fp16_into(x_rpu, out)
    ok &= _check_output(name + '/into', x_u8, out)
    changed = 255 - x_u8
    torch.ops.rpu.cast_uint8_fp16_into(changed.to('rpu'), out)
    ok &= _check_output(name + '/into-refresh', changed, out)
    return ok

def main() -> int:
    torch.manual_seed(SEED)
    ok = True
    ok &= _run_case('vision_1img', torch.randint(0, 256, (256, 1184), dtype=torch.uint8))
    ok &= _run_case('vision_3img', torch.randint(0, 256, (768, 1184), dtype=torch.uint8))
    ok &= _run_case('small_flat', torch.randint(0, 256, (16000,), dtype=torch.uint8))
    x = torch.randint(0, 256, (16128 + 16,), dtype=torch.uint8)
    x[0] = 0
    x[1] = 255
    ok &= _run_case('boundary', x)
    print(f"\n{('ALL PASS' if ok else 'FAILURES')}")
    return 0 if ok else 1
if __name__ == '__main__':
    sys.exit(main())
