from __future__ import annotations
import gc
import pytest

@pytest.fixture(autouse=True)
def _reset_live_ref_between_graph_tests():
    import rpu_backend.api.causal_lm as _cl
    _cl._LIVE_REF = None
    gc.collect()
    yield
    try:
        import torch
        torch._dynamo.reset()
    except Exception:
        pass
    _cl._LIVE_REF = None
    gc.collect()
