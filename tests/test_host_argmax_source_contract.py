"""Dispatch boundaries not exercised by the CPU-native semantic tests."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = (ROOT / "src/core/rpu_backend.cpp").read_text(encoding="utf-8")
DISPATCH = (ROOT / "src/core/rpu_dispatch_registrations.inc").read_text(
    encoding="utf-8"
)
CMAKE = (ROOT / "src/CMakeLists.txt").read_text(encoding="utf-8")
HOST = (ROOT / "src/core/rpu_host_argmax.h").read_text(encoding="utf-8")
FAKE = (ROOT / "python/rpu_backend/graph/fake_dispatch.py").read_text(
    encoding="utf-8"
)


def test_neon_scanner_is_shared_by_production_and_branch_compatibility_ops():
    assert '#include "rpu_host_argmax.h"' in BACKEND
    assert "rpu::host_argmax::reduce_cpu_rows(cpu_logits)" in BACKEND
    assert "rpu::host_argmax::argmax_lastdim_cpu(logits)" in BACKEND
    assert "rpu_lm_head_logits_top1" in BACKEND
    assert "rpu_argmax_lastdim_host" in BACKEND
    assert "rpu_argmax_host.cpp" not in CMAKE


def test_compatibility_op_accepts_cpu_or_rpu_but_production_top1_requires_rpu():
    assert "on_rpu || logits.device().is_cpu()" in BACKEND
    assert "? rpu_to_cpu_zerocopy(logits, /*flush=*/true)" in BACKEND
    production = BACKEND[BACKEND.index("at::Tensor rpu_lm_head_logits_top1") :]
    assert "logits must be on device" in production
    assert 'm.impl("argmax_lastdim_host", TORCH_FN(' in DISPATCH
    assert "TORCH_LIBRARY_IMPL(rpu, CPU, m)" in DISPATCH
    assert "_meta_argmax_lastdim_host" in FAKE


def test_cpu_fp32_extension_does_not_relax_rpu_or_production_top1_dtype():
    shared = BACKEND.split("at::Tensor host_top1_impl(", 1)[1].split("}  // namespace", 1)[0]
    assert "TORCH_CHECK(logits.scalar_type() == at::kHalf," in shared
    assert "host top1: logits must be fp16" in shared
    cpu_entry = HOST.split("inline at::Tensor argmax_lastdim_cpu(", 1)[1]
    assert 'TORCH_CHECK(logits.device().is_cpu(),' in cpu_entry
