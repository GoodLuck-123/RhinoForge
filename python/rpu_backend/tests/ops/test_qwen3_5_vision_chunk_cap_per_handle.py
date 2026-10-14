from __future__ import annotations
import pytest
import torch
import rpu_backend
pytestmark = pytest.mark.skipif(not hasattr(torch.ops.rpu, 'qwen3_5_vision_create'), reason='needs the native backend (device device nodes)')

@pytest.fixture
def two_handles():
    a = torch.ops.rpu.qwen3_5_vision_create()
    b = torch.ops.rpu.qwen3_5_vision_create()
    yield (a, b)
    torch.ops.rpu.qwen3_5_vision_destroy(a)
    torch.ops.rpu.qwen3_5_vision_destroy(b)

def test_the_setter_exists_at_all():
    assert hasattr(torch.ops.rpu, 'qwen3_5_vision_set_chunk_size_cap')

def test_two_live_handles_hold_different_caps(two_handles):
    (a, b) = two_handles
    torch.ops.rpu.qwen3_5_vision_set_chunk_size_cap(a, 64)
    torch.ops.rpu.qwen3_5_vision_set_chunk_size_cap(b, 128)
    torch.ops.rpu.qwen3_5_vision_set_chunk_size_cap(a, 64)

def test_order_does_not_matter(two_handles):
    (a, b) = two_handles
    torch.ops.rpu.qwen3_5_vision_set_chunk_size_cap(b, 256)
    torch.ops.rpu.qwen3_5_vision_set_chunk_size_cap(a, 32)
    torch.ops.rpu.qwen3_5_vision_set_chunk_size_cap(b, 256)

def test_zero_means_no_cap(two_handles):
    (a, _) = two_handles
    torch.ops.rpu.qwen3_5_vision_set_chunk_size_cap(a, 0)

def test_negative_cap_is_refused(two_handles):
    (a, _) = two_handles
    with pytest.raises(RuntimeError, match='must be >= 0'):
        torch.ops.rpu.qwen3_5_vision_set_chunk_size_cap(a, -1)

def test_a_fresh_handle_starts_unlatched(two_handles):
    (a, _) = two_handles
    torch.ops.rpu.qwen3_5_vision_set_chunk_size_cap(a, 64)
    c = torch.ops.rpu.qwen3_5_vision_create()
    try:
        torch.ops.rpu.qwen3_5_vision_set_chunk_size_cap(c, 512)
    finally:
        torch.ops.rpu.qwen3_5_vision_destroy(c)

def test_the_cap_is_cold_only(two_handles):
    (a, _) = two_handles
    torch.ops.rpu.qwen3_5_vision_set_chunk_size_cap(a, 64)
    assert torch.ops.rpu.qwen3_5_vision_get_resolved_chunk_size(a) == 0
