"""Execute both fused Python entries with real decode planning and host I/O."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import rpu_backend.graph as graph
from rpu_backend.adapters import qwen3
from rpu_backend.api.cache import RPUCache

from test_execution_planner import _native_candidate_v2


@pytest.fixture(params=["lm_head", "top1"])
def fused_decode(request, monkeypatch):
    events = []
    signatures = []
    consumed = []
    inputs = []
    state = SimpleNamespace(active_capture=False, reject=False, route_id=101)

    def native_decode(handle):
        assert handle == 7 and not state.active_capture
        events.append("plan")
        if state.reject:
            raise RuntimeError("synthetic decode admission rejected")
        return _native_candidate_v2(
            1, (16,) * 3, descriptor_version=3,
            routes=((1, state.route_id, 2, 0, (), 0),))

    @contextmanager
    def capture(signature):
        signatures.append(signature)
        events.append("capture")
        state.active_capture = True
        try:
            yield
        finally:
            state.active_capture = False

    graph_cache = graph.GraphCache()
    monkeypatch.setattr(graph_cache, "capture", capture)
    inner = SimpleNamespace(
        _rpu_decoder_handle=7, _rpu_decoder_num_layers=1,
        _rpu_decoder_hidden_size=16, _rpu_decoder_deepstack_hash=0,
        _rpu_batch_decode_enabled=False, _rpu_execution_generation=0,
        _rpu_execution={"prefill": {"chunk_size": 128}},
        _rpu_decode_stage_descriptor=(99,),
        _rpu_decoder_graph_cache=graph_cache)
    model = SimpleNamespace(model=inner,
        config=SimpleNamespace(hidden_size=16, vocab_size=32))
    cache = RPUCache(num_layers=1, batch_size=1, max_seq_len=16,
                     num_kv_heads=1, head_dim=16, device="cpu")
    hidden = torch.zeros(1, 1, 16, dtype=torch.float16)
    logits = torch.zeros(1, 1, 32, dtype=torch.float16)

    def run(owner, handle, **kwargs):
        assert owner is inner and handle == 7 and state.active_capture
        consumed.append(owner._rpu_decode_stage_descriptor)
        inputs.append(kwargs.get("input_ids"))
        assert consumed[-1] is owner._decoder_decode_plan[1].selected.stage_tuple.physical_descriptor
        events.append("native")
        return logits, kwargs["past_key_values"]

    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_resolve_decode_stage_descriptor",
                        native_decode, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_planner_cache_identity",
                        lambda handle: (handle, state.route_id), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_get_chunk_size_override",
                        lambda handle: 128, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "lm_head_logits_top1",
                        lambda raw: raw.argmax(-1), raising=False)
    monkeypatch.setattr(graph, "GraphSignature", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(qwen3, "_push_lm_head_weight", lambda *_: None)
    monkeypatch.setattr(qwen3, "_run_causal_decoder_forward", run)
    qwen3._apply_fused_lm_head_for_rpu(model)

    def invoke(input_ids=None):
        kwargs = dict(past_key_values=cache, use_cache=True)
        kwargs["inputs_embeds" if input_ids is None else "input_ids"] = (
            hidden if input_ids is None else input_ids)
        if request.param == "top1":
            return qwen3._run_fused_lm_head_decode_top1(model, **kwargs)
        return model.forward(**kwargs)

    return SimpleNamespace(invoke=invoke, events=events, state=state,
                           signatures=signatures, inner=inner, consumed=consumed,
                           inputs=inputs)


def test_fused_decode_keeps_changed_cpu_ids_for_host_embedding(fused_decode, monkeypatch):
    original_to = torch.Tensor.to
    def no_upload(tensor, *args, **kwargs):
        if (args and args[0] == "rpu") or kwargs.get("device") == "rpu":
            pytest.fail("CPU token IDs were uploaded before host embedding")
        return original_to(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", no_upload)
    first, second = torch.tensor([[3]]), torch.tensor([[7]])
    fused_decode.invoke(first)
    fused_decode.invoke(second)
    assert fused_decode.inputs == [first, second]
    assert fused_decode.inputs[0] is first and fused_decode.inputs[1] is second
    assert fused_decode.signatures[0].dyn_dims == fused_decode.signatures[1].dyn_dims


def test_fused_decode_admission_rejects_before_capture(fused_decode):
    # Installation already prepared a descriptor. A changed native admission
    # must advance identity before the next production request re-queries it.
    fused_decode.state.reject = True
    fused_decode.state.route_id += 1
    fused_decode.events.clear()
    with pytest.raises(RuntimeError, match="decode admission rejected"):
        fused_decode.invoke()
    assert fused_decode.events == ["plan"]
    assert fused_decode.signatures == []


def test_fused_decode_graph_key_tracks_actual_planner_descriptor(fused_decode):
    fused_decode.invoke()
    first = fused_decode.inner._rpu_decode_stage_descriptor
    assert first != (99,)
    assert fused_decode.events == ["plan", "capture", "native"]
    first_words = fused_decode.inner._decoder_decode_plan[1].graph_key_words()
    assert len(first_words) == 8 < len(first)
    assert fused_decode.signatures[0].dyn_dims[-8:] == list(first_words)

    fused_decode.invoke()
    assert fused_decode.events.count("plan") == 1
    assert fused_decode.signatures[1].dyn_dims == fused_decode.signatures[0].dyn_dims
    assert fused_decode.consumed[1] is first

    # The shared planner can re-admit after a cost/admission change while
    # the handle, generation and installed prefill override stay constant.
    fused_decode.state.route_id = 102
    fused_decode.invoke()
    second = fused_decode.inner._rpu_decode_stage_descriptor
    assert second != first
    second_words = fused_decode.inner._decoder_decode_plan[1].graph_key_words()
    assert second_words != first_words
    assert fused_decode.signatures[2].dyn_dims[-8:] == list(second_words)
    assert fused_decode.signatures[0].dyn_dims != fused_decode.signatures[2].dyn_dims
    assert fused_decode.signatures[0].dyn_dims[:-8] == \
        fused_decode.signatures[2].dyn_dims[:-8]
    assert fused_decode.consumed[2] is second
    assert fused_decode.inner._rpu_execution == {"prefill": {"chunk_size": 128}}
