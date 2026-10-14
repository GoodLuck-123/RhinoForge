from __future__ import annotations
import sys
import torch
import rpu_backend
SEED = 20260702
(SEQ, PEK) = (256, 1184)

def main() -> int:
    from rpu_backend.tests._hw_closure_evidence import stable_replay
    torch.manual_seed(SEED)
    gc = rpu_backend.graph.GraphCache()
    gc.begin_warmup()
    in_rpu = torch.empty(SEQ, PEK, dtype=torch.uint8, device='rpu')
    out_rpu = torch.empty(SEQ, PEK, dtype=torch.float16, device='rpu')
    ok = True
    try:
        for it in range(4):
            ubuf = torch.randint(0, 256, (SEQ, PEK), dtype=torch.uint8)
            in_rpu.copy_(ubuf)
            sig = rpu_backend.graph.GraphSignature(op_id='test_patch_cast', shapes=[SEQ, PEK], dtypes=[torch.float16])
            before = gc.snapshot()
            with gc.capture(sig):
                torch.ops.rpu.cast_uint8_fp16_into(in_rpu, out_rpu)
            after = gc.snapshot()
            assert len(after) == gc.size() == 1 and gc.cache_invariant_ok()
            entry = after[0]
            assert entry.recapture_count == 0 and entry.kernel_count > 0
            assert entry.data_node_count >= 2 and (not entry.non_replayable_reason)
            if it == 0:
                assert not before and entry.replay_count == 0
                assert entry.graph_lifetime_id > 0 and entry.build_generation > 0
                assert entry.execution_ordinal == 1
                gc.freeze()
            else:
                assert gc.is_frozen() and stable_replay(gc, before, after, 1)
                assert entry.execution_ordinal == before[0].execution_ordinal + 1
            out_cpu = out_rpu.cpu()
            ref = ubuf.half()
            assert out_cpu.shape == ref.shape and out_cpu.dtype == ref.dtype
            assert out_cpu.numel() and torch.isfinite(out_cpu).all()
            max_abs = (out_cpu - ref).abs().max().item()
            phase = 'BUILD' if it == 0 else 'REPLAY'
            good = torch.equal(out_cpu, ref)
            ok &= good
            print(f"  [it{it} {phase}] max_abs={max_abs:.3e} -> {('PASS' if good else 'FAIL (stale read?)')}")
    finally:
        gc.clear()
        assert gc.size() == 0 and gc.cache_invariant_ok()
    print(f"\n{('ALL PASS' if ok else 'FAILURES')}")
    return 0 if ok else 1
if __name__ == '__main__':
    sys.exit(main())

def test_cast_uint8_fp16_graph_replay(record_property):
    assert main() == 0
