import gc
import sys
import threading
import pytest
import torch.nn as nn

def test_two_live_instances_raises():
    from rpu_backend.api.causal_lm import _claim_live_instance, _release_live_instance, _LIVE_REF
    from rpu_backend.api.errors import RPUSingleHandleError
    try:
        _release_live_instance()
    except Exception:
        pass
    m1 = nn.Linear(16, 16)
    m2 = nn.Linear(16, 16)
    _claim_live_instance(m1)
    with pytest.raises(RPUSingleHandleError):
        _claim_live_instance(m2)
    _release_live_instance(m1)
    _claim_live_instance(m2)
    _release_live_instance(m2)

def test_idempotent_same_model():
    from rpu_backend.api.causal_lm import _claim_live_instance, _release_live_instance
    m = nn.Linear(16, 16)
    try:
        _release_live_instance(m)
    except Exception:
        pass
    _claim_live_instance(m)
    _claim_live_instance(m)
    _release_live_instance(m)

def test_stale_gc_callback_does_not_release_new_owner():
    import rpu_backend.api.causal_lm as lifecycle
    old = nn.Linear(16, 16)
    new = nn.Linear(16, 16)
    lifecycle._claim_live_instance(old)
    lifecycle._release_live_instance(old)
    lifecycle._claim_live_instance(new)
    del old
    gc.collect()
    assert lifecycle._LIVE_REF is not None
    assert lifecycle._LIVE_REF() is new
    lifecycle._release_live_instance(new)

def test_concurrent_claim_has_exactly_one_winner():
    from rpu_backend.api.causal_lm import _claim_live_instance, _release_live_instance
    from rpu_backend.api.errors import RPUSingleHandleError
    models = [nn.Linear(16, 16) for _ in range(8)]
    barrier = threading.Barrier(len(models))
    winners = []
    rejected = []

    def claim(model):
        barrier.wait()
        try:
            _claim_live_instance(model)
            winners.append(model)
        except RPUSingleHandleError:
            rejected.append(model)
    threads = [threading.Thread(target=claim, args=(model,)) for model in models]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert len(winners) == 1
    assert len(rejected) == len(models) - 1
    _release_live_instance(winners[0])

def main() -> int:
    test_two_live_instances_raises()
    test_idempotent_same_model()
    print('PASS (UNIT): single-live-instance invariant; E2E case requires hardware (pytest -m hardware)')
    return 0
if __name__ == '__main__':
    sys.exit(main())

@pytest.mark.hardware
def test_api08_two_live_models_raises_e2e():
    from rpu_backend import RPUModelForCausalLM
    from rpu_backend.api.errors import RPUSingleHandleError
    from rpu_backend.api.causal_lm import _release_live_instance
    try:
        _release_live_instance()
    except Exception:
        pass
    from rpu_backend.model_registry import model_path
    TINY = str(model_path('qwen3-0.6b'))
    m1 = RPUModelForCausalLM.from_pretrained(TINY).to('rpu')
    try:
        with pytest.raises(RPUSingleHandleError):
            m2 = RPUModelForCausalLM.from_pretrained(TINY).to('rpu')
    finally:
        _release_live_instance()
