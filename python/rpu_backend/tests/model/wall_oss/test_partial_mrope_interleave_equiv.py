from __future__ import annotations
import sys
import os
import importlib.util
import torch
_rp_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../runtime/rope_partial.py'))
_rp_spec = importlib.util.spec_from_file_location('wall_oss_rope_partial_cpu', _rp_path)
_rp_mod = importlib.util.module_from_spec(_rp_spec)
_rp_spec.loader.exec_module(_rp_mod)
build_interleaved_mrope_cos_sin = _rp_mod.build_interleaved_mrope_cos_sin
HEAD_DIM = 128
ROPE_THETA = 1000000.0
MROPE_SECTION = [16, 24, 24]

def _inv_freq(rotary_dim: int) -> torch.Tensor:
    return 1.0 / ROPE_THETA ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim)

def _axis_of_lane(half: int, mrope_section: list[int]) -> list[int]:
    axis = [0] * half
    for i in range(mrope_section[1]):
        pos = 1 + i * 3
        if pos < half:
            axis[pos] = 1
    for i in range(mrope_section[2]):
        pos = 2 + i * 3
        if pos < half:
            axis[pos] = 2
    return axis

def _reference_interleaved_cos_sin(position_ids: torch.Tensor, head_dim: int, rotary_dim: int, mrope_section: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    half = rotary_dim // 2
    inv_freq = _inv_freq(rotary_dim)
    max_pos = int(position_ids.max().item()) + 1
    positions = torch.arange(max_pos, dtype=torch.float64)
    freqs = positions[:, None] * inv_freq[None, :]
    cos_tbl = freqs.cos().to(torch.float16)
    sin_tbl = freqs.sin().to(torch.float16)
    axis = _axis_of_lane(half, mrope_section)
    seq = position_ids.size(0)
    cos_il = torch.empty((seq, half), dtype=torch.float16)
    sin_il = torch.empty((seq, half), dtype=torch.float16)
    for t in range(seq):
        for j in range(half):
            p = int(position_ids[t, axis[j]].item())
            cos_il[t, j] = cos_tbl[p, j]
            sin_il[t, j] = sin_tbl[p, j]
    return (cos_il, sin_il)

def _make_position_ids() -> dict[str, torch.Tensor]:
    cases = {}
    seq = 40
    p = torch.arange(seq, dtype=torch.long)
    cases['text_only'] = p[:, None].expand(seq, 3).contiguous()
    rows = []
    for i in range(5):
        rows.append([i, i, i])
    base = 5
    for h in range(4):
        for w in range(4):
            rows.append([base, base + h, base + w])
    nxt = base + 4
    for i in range(5):
        rows.append([nxt + i, nxt + i, nxt + i])
    cases['mixed'] = torch.tensor(rows, dtype=torch.long)
    return cases

def run() -> int:
    rotary_dim = HEAD_DIM
    ok = True
    for (name, pos) in _make_position_ids().items():
        (ref_cos, ref_sin) = _reference_interleaved_cos_sin(pos, HEAD_DIM, rotary_dim, MROPE_SECTION)
        (out_cos, out_sin) = build_interleaved_mrope_cos_sin(pos, head_dim=HEAD_DIM, rope_theta=ROPE_THETA, mrope_section=MROPE_SECTION, rotary_dim=rotary_dim)
        out_cos = out_cos.to(torch.float16).cpu()
        out_sin = out_sin.to(torch.float16).cpu()
        shape_ok = out_cos.shape == (pos.size(0), rotary_dim // 2)
        cos_eq = torch.equal(out_cos, ref_cos)
        sin_eq = torch.equal(out_sin, ref_sin)
        case_ok = shape_ok and cos_eq and sin_eq
        ok = ok and case_ok
        print(f"[{name}] shape={out_cos.shape} shape_ok={shape_ok} cos_bit_exact={cos_eq} sin_bit_exact={sin_eq} -> {('PASS' if case_ok else 'FAIL')}")
        if not cos_eq:
            d = (out_cos.float() - ref_cos.float()).abs()
            bad = torch.nonzero(d > 0)
            print(f'    cos mismatch: {bad.size(0)} lanes, max|d|={d.max().item():.2e}')
    print(f"\n[summary] {('ALL PASS' if ok else 'FAIL')}")
    return 0 if ok else 1

def test_interleaved_mrope_matches_reference():
    run()
