"""MR-C / T31 (B3) — prefix_scatter takes retained Tensors, not raw addresses.

THE C-1 SHAPE THIS REMOVES
    The op used to be

        qwen3vl_vision_set_prefix_scatter(int handle, int inputs_embeds_addr,
                                          int[] dense_addrs, ...)

    and the model did `reinterpret_cast<c10::Half*>(addr)`, holding NO reference
    to the storage behind it. Whether those pointers stayed valid depended on the
    caller keeping its own tensors alive — rhino_e2e.py does so via
    cache["inputs_embeds_persist"] / cache["deepstack_dense_persist"], purely by
    convention, with nothing enforcing or even checking it. A caller that dropped
    them would have the merger DMA its output into reused memory: no crash, no
    exception, silently corrupted neighbours.

WHAT THE TENSOR FORM BUYS BEYOND LIFETIME
    An integer can only be checked against 0. A Tensor can be checked for device,
    dtype and contiguity — every one of which the scatter DMA already assumed and
    none of which was verifiable before.

Host-free where possible; the cases below need the native backend because they
exercise the real op boundary. They do NOT build a full RhinoVLA: the end-to-end
BUILD -> REPLAY lifecycle for this path is rhino_e2e.py itself, and the
argument-contract half is what had no coverage at all.
"""
from __future__ import annotations

import os

import pytest
import torch

import rpu_backend  # noqa: F401  (loads the native extension)

pytestmark = pytest.mark.skipif(
    not hasattr(torch.ops.rpu, "qwen3vl_vision_create"),
    reason="needs the native backend (device device nodes)")

OUT_HIDDEN = 2048


@pytest.fixture
def handle():
    h = torch.ops.rpu.qwen3vl_vision_create()
    yield h
    torch.ops.rpu.qwen3vl_vision_destroy(h)


def _target(rows=64, cols=OUT_HIDDEN):
    return torch.zeros(rows, cols, dtype=torch.float16, device="rpu")


def test_the_schema_takes_tensors_not_addresses():
    """The whole point of B3. If someone reverts to int addresses this fails
    before any of the behavioural cases below get a chance to."""
    schema = str(torch.ops.rpu.qwen3vl_vision_set_prefix_scatter.default._schema)
    assert "Tensor inputs_embeds" in schema, schema
    assert "Tensor[] dense" in schema, schema
    assert "_addr" not in schema, f"integer-address parameters are back: {schema}"


def test_accepts_valid_rpu_targets(handle):
    torch.ops.rpu.qwen3vl_vision_set_prefix_scatter(
        handle, _target(), [_target(), _target()], [0, 16], [8, 8])


def test_a_cpu_target_is_refused(handle):
    """Unverifiable in the integer form — a CPU pointer is just another int, and
    the scatter DMA would have written to a host address."""
    cpu = torch.zeros(64, OUT_HIDDEN, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="must be on the device device"):
        torch.ops.rpu.qwen3vl_vision_set_prefix_scatter(
            handle, cpu, [_target()], [0], [8])


def test_a_wrong_dtype_target_is_refused(handle):
    fp32 = torch.zeros(64, OUT_HIDDEN, dtype=torch.float32, device="rpu")
    with pytest.raises(RuntimeError, match="must be fp16"):
        torch.ops.rpu.qwen3vl_vision_set_prefix_scatter(
            handle, fp32, [_target()], [0], [8])


def test_a_noncontiguous_target_is_refused(handle):
    """The DMA writes a flat run; a strided view would scatter into the wrong
    rows. Also invisible to the old integer form, which saw only the base
    pointer."""
    view = _target(64, OUT_HIDDEN * 2)[:, ::2]
    assert not view.is_contiguous()
    with pytest.raises(RuntimeError, match="must be contiguous"):
        torch.ops.rpu.qwen3vl_vision_set_prefix_scatter(
            handle, view, [_target()], [0], [8])


def test_a_bad_dense_target_is_refused_too(handle):
    """The list elements get the same validation as the pooler target — the old
    form checked each dense address only for != 0."""
    cpu = torch.zeros(64, OUT_HIDDEN, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="deepstack dense target"):
        torch.ops.rpu.qwen3vl_vision_set_prefix_scatter(
            handle, _target(), [_target(), cpu], [0], [8])


def test_run_bounds_are_still_enforced(handle):
    """Pre-existing validation that must survive the signature change."""
    with pytest.raises(RuntimeError, match="run start/length"):
        torch.ops.rpu.qwen3vl_vision_set_prefix_scatter(
            handle, _target(), [_target()], [0], [0])
    with pytest.raises(RuntimeError, match="size mismatch"):
        torch.ops.rpu.qwen3vl_vision_set_prefix_scatter(
            handle, _target(), [_target()], [0, 16], [8])


def test_the_model_keeps_the_targets_alive(handle):
    """The property the whole change exists for.

    Register targets, then drop every Python reference and force a collection.
    Under the old integer API the model's pointers would now be dangling; the
    allocator is free to hand that storage to the next tensor. Holding the
    Tensors means the storage cannot be reused, which we check by confirming a
    freshly allocated tensor does not land on the retained data_ptr.

    Not a proof of absence — an allocator could coincidentally place elsewhere —
    but it fails loudly in the case that actually bit: immediate reuse of
    just-freed storage.
    """
    import gc

    ie = _target()
    dense = [_target()]
    ie_ptr = ie.data_ptr()
    dense_ptr = dense[0].data_ptr()
    torch.ops.rpu.qwen3vl_vision_set_prefix_scatter(
        handle, ie, dense, [0], [8])

    del ie, dense
    gc.collect()

    fresh = [_target() for _ in range(8)]
    landed = {t.data_ptr() for t in fresh}
    assert ie_ptr not in landed, (
        "the pooler target's storage was recycled — the model is not retaining it")
    assert dense_ptr not in landed, (
        "a deepstack target's storage was recycled — the model is not retaining it")


@pytest.mark.skipif(
    os.environ.get("RPU_HW_AVAILABLE") != "1",
    reason="requires device hardware (set RPU_HW_AVAILABLE=1)",
)
@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("operation", ["scatter", "gather", "zero_blocks"])
def test_host_visible_writers_mutate_values_views_and_versions(operation, inference):
    """Check real dispatcher writes and shared versions, including inference tensors."""
    with torch.inference_mode(inference):
        original = torch.full((4, 64), -2.0, dtype=torch.float16)
        destination = original.to("rpu")
        view = destination.view(1, 4, 64)
        before = None if inference else destination._version
        expected = original.clone()
        if operation == "scatter":
            source = torch.arange(128, dtype=torch.float16).reshape(2, 64)
            indices = torch.tensor([0, 2])
            torch.ops.rpu.qwen3vl_scatter_row_parts_unflushed_(
                destination, indices, [source[:1], source[1:]])
            expected[indices] = source
        elif operation == "gather":
            vocabulary = torch.arange(8 * 64, dtype=torch.float16).reshape(8, 64)
            indices = torch.tensor([1, -1, 7, 0])
            torch.ops.rpu.qwen3vl_gather_embedding_from_rpu_unflushed_(
                destination, vocabulary.to("rpu"), indices, -1)
            expected[[0, 2, 3]] = vocabulary[[1, 7, 0]]
        else:
            torch.ops.rpu.zero_contiguous_tensor_blocks_sized([view], 2)
            expected[2].zero_()
        assert torch.equal(destination.cpu(), expected)
        assert torch.equal(view.cpu().reshape(4, 64), expected)
        if not inference:
            assert destination._version == before + 1
            assert view._version == destination._version


@pytest.mark.skipif(
    os.environ.get("RPU_HW_AVAILABLE") != "1",
    reason="requires device hardware (set RPU_HW_AVAILABLE=1)",
)
@pytest.mark.parametrize("operation,rows", [("gather", 1), ("scatter", 16)])
def test_published_host_inputs_reach_device_dma_without_cpu_readback(operation, rows):
    """A tiny identity GEMM reads the same DDR that the decoder DMA consumes.

    No destination .cpu()/copy_ runs between an unflushed write and this device
    read. The separately uploaded control exercises the identical kernel, so
    the comparison isolates publication rather than model numerical accuracy.
    """
    from rpu_backend.runtime.weights import transform_linear_weight

    old_flush = torch.rpu.get_ddr_flush()
    old_force = torch.rpu.get_ddr_flush_force()
    torch.rpu.set_ddr_flush(False)
    torch.rpu.set_ddr_flush_force(True)
    try:
        width = 256
        weight = transform_linear_weight(torch.eye(width).half(), 1, 8).to("rpu")
        vocabulary_cpu = (torch.arange(4 * width).reshape(4, width) % 31 - 15).half() / 16
        vocabulary = vocabulary_cpu.to("rpu")
        owner = torch.full((rows + 2, width), -7., dtype=torch.float16, device="rpu")
        destination = owner[1:rows + 1]
        snapshots = []
        for value in (0, 2, 0):
            if operation == "gather":
                expected = vocabulary_cpu[value:value + 1].clone()
            else:
                expected = torch.full((rows, width), -3.).half()
                expected[::2] = vocabulary_cpu[value]
            control = expected.to("rpu")  # independent, already published owner
            if operation == "gather":
                torch.ops.rpu.qwen3vl_gather_embedding_from_rpu_unflushed_(
                    destination, vocabulary, torch.tensor([value]), -1)
            else:
                # Two partial scatters fully prepare the owner before publication.
                for begin in (0, 1):
                    indices = torch.arange(begin, rows, 2)
                    torch.ops.rpu.qwen3vl_scatter_row_parts_unflushed_(
                        destination, indices, [expected[begin::2].contiguous()])
            torch.ops.rpu.qwen3vl_publish_inputs([destination])
            actual = torch.ops.rpu.linear_with_accumulation(
                destination, weight, None, 1, 8, False).cpu().clone()
            reference = torch.ops.rpu.linear_with_accumulation(
                control, weight, None, 1, 8, False).cpu().clone()
            assert torch.equal(actual, reference)
            assert torch.equal(reference, expected)
            snapshots.append(actual)
        assert torch.equal(snapshots[0], snapshots[2])
        assert not torch.equal(snapshots[0], snapshots[1])
        neighbours = owner.cpu()
        assert torch.equal(neighbours[0], torch.full((width,), -7.).half())
        assert torch.equal(neighbours[-1], neighbours[0])
    finally:
        torch.rpu.set_ddr_flush(old_flush)
        torch.rpu.set_ddr_flush_force(old_force)
