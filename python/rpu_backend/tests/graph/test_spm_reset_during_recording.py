from __future__ import annotations
import sys
import torch
import rpu_backend

def _make_sig(op_id: str) -> 'rpu_backend.graph.GraphSignature':
    return rpu_backend.graph.GraphSignature(op_id=op_id, shapes=[1, 1], dyn_dims=[1], dtypes=[torch.float16])

def test_passthrough_reset_no_op_on_graph() -> None:
    torch.ops.rpu.spm_alloc_reset_temporary()
    print('[PASS] PASSTHROUGH: reset_temporary host-side ran without error')

def test_recording_reset_marks_non_replayable() -> None:
    gcache = rpu_backend.graph.GraphCache()
    sig = _make_sig('spm_reset_recording_test')
    with gcache.capture(sig) as g:
        assert g.state() == 1, f'expected RECORDING (1), got state={g.state()}'
        torch.ops.rpu.spm_alloc_reset_temporary()
        assert g.state() == 1, f'state must remain RECORDING, got {g.state()}'
        assert not g.replayable(), 'Graph must be marked non_replayable after spm reset during RECORDING'
    snap = gcache.snapshot()
    for entry in snap:
        reason = entry.non_replayable_reason
        if reason:
            print(f'[INFO] snapshot non_replayable_reason: {reason!r}')
            assert 'spm' in reason.lower() or 'reset' in reason.lower(), f"non_replayable_reason should mention 'spm'/'reset', got: {reason!r}"
            break
    print('[PASS] RECORDING: spm_alloc_reset_temporary marked graph non_replayable')

def test_recording_reset_idempotent() -> None:
    gcache = rpu_backend.graph.GraphCache()
    sig = _make_sig('spm_reset_idempotent_test')
    with gcache.capture(sig) as g:
        torch.ops.rpu.spm_alloc_reset_temporary()
        first_replayable = g.replayable()
        torch.ops.rpu.spm_alloc_reset_temporary()
        second_replayable = g.replayable()
        assert not first_replayable and (not second_replayable), 'Both resets should leave replayable=False'
    print('[PASS] RECORDING: idempotent reset (no crash on second call)')

def main() -> int:
    test_passthrough_reset_no_op_on_graph()
    test_recording_reset_marks_non_replayable()
    test_recording_reset_idempotent()
    print('\n============================================================')
    print('ALL TESTS PASSED')
    print('============================================================')
    return 0
if __name__ == '__main__':
    sys.exit(main())
