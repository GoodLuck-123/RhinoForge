"""Host checks for adapter continuation semantics, independent of device import."""
import importlib.util
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "causal_append_under_test", ROOT / "python/rpu_backend/runtime/causal_append.py")
APPEND = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(APPEND)


@pytest.mark.parametrize("position,length,expected", [
    (31, 33, ((0, 1), (1, 32))),
    (128, 33, ((0, 32), (32, 1))),
    (161, 33, tuple((i, 1) for i in range(15)) + ((15, 16), (31, 1), (32, 1))),
])
def test_zero_budget_keeps_real_token_positions(position, length, expected):
    assert APPEND.continuation_segments(position, length,
        {"chunk_size": "auto", "padding_budget": 0}, capacity=512) == expected


@pytest.mark.parametrize("stage,length", [
    ({"chunk_size": 32, "padding_rows": 0}, 32),
    ({"chunk_size": 64, "padding_rows": 0}, 64),
    ({"chunk_size": 32, "padding_budget": 0}, 33),
    ({"padding_rows": 15}, 33),
    ({"padding_rows": "auto", "padding_budget": 16}, 33),
])
def test_exact_requests_and_auto_padding_keep_the_native_plan(stage, length):
    assert APPEND.continuation_segments(128, length, stage, capacity=512) is None


def test_no_segmentation_of_initial_prefill_or_logically_aligned_append():
    for position, length in ((0, 4097), (128, 32), (128, 64), (31, 1)):
        assert APPEND.continuation_segments(position, length,
            {"padding_budget": 0}, capacity=8192) is None
    with pytest.raises(ValueError, match="horizon"):
        APPEND.continuation_segments(128, 33, {"padding_budget": 0}, capacity=160)


class Cache:
    def __init__(self, position):
        self.position = position

    def reset_to_position(self, position):
        self.position = position


@pytest.mark.parametrize("position,spans", [(31, ((0, 1), (1, 32))), (128, ((0, 32), (32, 1)))])
@pytest.mark.parametrize("keep", [0, 1, 5, torch.tensor([0, 32])])
def test_composition_preserves_logits_positions_and_segment_receipts(position, spans, keep):
    owner = SimpleNamespace()
    cache = Cache(position)
    inputs = torch.arange(33).reshape(1, 33)
    positions = torch.arange(position, position + 33).reshape(1, 1, 33).expand(3, 1, 33)
    calls = []

    def forward(**arguments):
        count = arguments["input_ids"].shape[1]
        assert arguments["logits_to_keep"] in (0, 1)
        assert arguments["marker"] == "preserved"
        torch.testing.assert_close(arguments["position_ids"][0, 0],
            torch.arange(cache.position, cache.position + count))
        assert arguments["attention_mask"].shape == (1, count)
        calls.append(count)
        owner._rpu_last_execution_plan = dict(position=cache.position,
            logical_len=count, execution_len=count, chunk_size=32 if count > 1 else 16,
            padding_rows=0, physical_descriptor=(count, cache.position))
        cache.position += count
        values = arguments["input_ids"].float().unsqueeze(-1)
        if arguments["logits_to_keep"] == 1:
            values = values[:, -1:]
        return SimpleNamespace(logits=values, past_key_values=cache)

    result = APPEND.run_continuation_segments(forward, cache, owner, spans, dict(
        input_ids=inputs, past_key_values=cache, attention_mask=torch.ones(1, 33),
        position_ids=positions, cache_position=torch.arange(position, position + 33),
        logits_to_keep=keep, marker="preserved"))
    selection = slice(-keep, None) if isinstance(keep, int) and keep else (
        slice(None) if isinstance(keep, int) else keep)
    torch.testing.assert_close(result.logits, inputs.float().unsqueeze(-1)[:, selection])
    assert calls == [count for _, count in spans]
    assert cache.position == position + 33
    receipt = owner._rpu_last_execution_plan
    assert receipt["position"] == position and receipt["logical_len"] == 33
    assert [part["position"] for part in receipt["segments"]] == [position + off for off, _ in spans]
    assert all(part["physical_descriptor"] for part in receipt["segments"])


def test_failed_later_segment_restores_logical_position():
    previous = {"stage": "prefill", "position": 0, "logical_len": 31}
    cache = Cache(31)
    owner = SimpleNamespace(_rpu_last_execution_plan=previous)
    def forward(**arguments):
        if cache.position != 31:
            raise RuntimeError("native failure")
        owner._rpu_last_execution_plan = dict(position=31, logical_len=1,
            execution_len=1, chunk_size=16)
        cache.position += 1
        return SimpleNamespace(logits=torch.zeros(1, 1, 4))
    with pytest.raises(RuntimeError, match="native failure"):
        APPEND.run_continuation_segments(forward, cache, owner, ((0, 1), (1, 32)),
            dict(input_ids=torch.ones(1, 33), logits_to_keep=1))
    assert cache.position == 31
    assert owner._rpu_last_execution_plan is previous


def test_composite_receipt_uses_internal_publication_on_validated_module():
    from rpu_backend.runtime.hw_attrs import install_hw_attr_validator
    from rpu_backend.api.errors import RPUConfigError

    owner = torch.nn.Module()
    install_hw_attr_validator(owner)
    with pytest.raises(RPUConfigError, match="Unknown hardware attribute"):
        owner._rpu_last_execution_plan = {}
    cache = Cache(31)

    def forward(**arguments):
        count = arguments["input_ids"].shape[1]
        vars(owner)["_rpu_last_execution_plan"] = {
            "position": cache.position, "logical_len": count,
            "execution_len": count, "padding_rows": 0,
            "chunk_size": 32 if count > 1 else 16,
        }
        cache.position += count
        return SimpleNamespace(logits=torch.ones(1, 1, 4))

    output = APPEND.run_continuation_segments(
        forward, cache, owner, ((0, 1), (1, 32)),
        dict(input_ids=torch.ones(1, 33), logits_to_keep=1))
    assert output.logits.shape == (1, 1, 4)
    assert cache.position == 64
    assert owner._rpu_last_execution_plan["execution_kind"] == "segmented_append"
    with pytest.raises(RPUConfigError, match="Unknown hardware attribute"):
        owner._rpu_unsupported_user_setting = 1


def _qwen3_forward_double(*, fail_body=False, stage=None):
    """Execute the real adapter closure with only native dispatch doubled."""
    source = (ROOT / "python/rpu_backend/adapters/qwen3.py").read_text()
    node = next(n for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.FunctionDef) and n.name == "fused_forward")
    decoder_source = (ROOT / "python/rpu_backend/runtime/decoder.py").read_text()
    controls = next(n for n in ast.parse(decoder_source).body
                    if isinstance(n, ast.FunctionDef)
                    and n.name == "_canonicalize_plain_text_controls")
    events = []
    def plan(owner, handle, cache, count):
        events.append(("plan", cache.position, count))
        if fail_body:
            raise RuntimeError("body has no native descriptor")
        return count, 32, SimpleNamespace(graph_key_words=lambda: (count, cache.position))
    def runner(owner, handle, *, input_ids, past_key_values, **kwargs):
        count = input_ids.shape[1]
        events.append(("run", past_key_values.position, count))
        owner._rpu_last_execution_plan = dict(position=past_key_values.position,
            logical_len=count, execution_len=count, padding_rows=0,
            chunk_size=32 if count > 1 else 16, physical_descriptor=(count,))
        past_key_values.position += count
        # Reproduce the different native decode and prefill result contracts.
        return input_ids.float().unsqueeze(-1).expand(1, count, 4 if count == 1 else 2), past_key_values
    namespace = dict(torch=torch, topology=None, _RPUCache_for_sig=Cache,
        execution_serialized=lambda function: function,
        validate_decoder_cache_topology=lambda *a, **k: None,
        _is_batched_prefill=lambda *a: False,
        _reject_unconsumed_text_padding=lambda *a: None,
        _text_prefill_execution_plan=plan,
        _text_decode_execution_plan=lambda *a: SimpleNamespace(graph_key_words=lambda: (1,) * 8),
        continuation_segments=APPEND.continuation_segments,
        run_continuation_segments=APPEND.run_continuation_segments,
        _GraphSignature=lambda **kw: kw, chunk_policy_key=lambda h: 0,
        prefill_position_key=lambda length, position: position if length > 1 else 0,
        _run_causal_decoder_forward=runner, CausalLMOutputWithPast=SimpleNamespace)
    exec(compile(ast.Module(body=[controls, node], type_ignores=[]), "qwen3-forward", "exec"), namespace)
    class Head:
        weight = torch.empty(1, dtype=torch.float16)
        def __call__(self, hidden):
            return hidden[..., :1].expand(*hidden.shape[:-1], 4)
    owner = SimpleNamespace(_rpu_decoder_handle=1,
        _rpu_execution={"prefill": stage or {"padding_budget": 0}},
        _rpu_decoder_hidden_size=2, _rpu_decoder_num_layers=1,
        _rpu_decoder_deepstack_hash=0, _rpu_batch_decode_enabled=False,
        _rpu_decode_stage_descriptor=(1,),
        _rpu_decoder_graph_cache=SimpleNamespace(capture=lambda signature: nullcontext()))
    model = SimpleNamespace(model=owner, config=SimpleNamespace(hidden_size=2, vocab_size=4), lm_head=Head())
    return namespace["fused_forward"], model, events


def test_real_qwen3_forward_preflights_then_composes_decode_logits_with_prefill_logits():
    forward, model, events = _qwen3_forward_double()
    cache = Cache(31)
    cache.max_seq_len = 128
    result = forward(model, input_ids=torch.arange(33).reshape(1, 33), past_key_values=cache,
        attention_mask=torch.ones(1, 64), position_ids=torch.arange(31, 64).reshape(1, 33),
        cache_position=torch.arange(31, 64), logits_to_keep=0)
    assert events[0] == ("plan", 32, 32)
    assert [event for event in events if event[0] == "run"] == [("run", 31, 1), ("run", 32, 32)]
    torch.testing.assert_close(result.logits[..., 0], torch.arange(33).float().reshape(1, 33))
    assert result.logits.shape == (1, 33, 4) and cache.position == 64
    assert model.model._rpu_last_execution_plan["execution_kind"] == "segmented_append"


@pytest.mark.parametrize("invalid", ["mask", "positions", "cache_positions", "body", "horizon"])
def test_real_qwen3_forward_rejects_before_any_append_write(invalid):
    forward, model, events = _qwen3_forward_double(fail_body=invalid == "body")
    cache = Cache(31)
    cache.max_seq_len = 63 if invalid == "horizon" else 128
    arguments = dict(input_ids=torch.arange(33).reshape(1, 33), past_key_values=cache,
        attention_mask=torch.ones(1, 64), position_ids=torch.arange(31, 64),
        cache_position=torch.arange(31, 64), logits_to_keep=1)
    if invalid == "mask":
        arguments["attention_mask"][0, -1] = 0
    if invalid == "positions":
        arguments["position_ids"][2] += 1
    if invalid == "cache_positions":
        arguments["cache_position"] = torch.arange(33)
    with pytest.raises((ValueError, NotImplementedError, RuntimeError)):
        forward(model, **arguments)
    assert cache.position == 31
    assert not any(event[0] == "run" for event in events)


@pytest.mark.parametrize("chunk", [32, 64])
def test_real_qwen3_aligned_exact_chunk_keeps_one_forward(chunk):
    forward, model, events = _qwen3_forward_double(stage={"chunk_size": chunk, "padding_rows": 0})
    cache = Cache(128)
    cache.max_seq_len = 256
    forward(model, input_ids=torch.ones(1, chunk, dtype=torch.long),
            past_key_values=cache, logits_to_keep=1)
    assert [event for event in events if event[0] == "run"] == [("run", 128, chunk)]
    assert "segments" not in model.model._rpu_last_execution_plan


@pytest.mark.parametrize("invalid", [None, "mask", "positions", "cache_positions", "body"])
def test_real_vl_forward_preserves_supplied_embeddings_and_preflights_all_segments(monkeypatch, invalid):
    from rpu_backend.adapters import qwen3_vl as q
    from rpu_backend.api.cache import RPUCache
    original_to = torch.Tensor.to
    def cpu_to(tensor, *args, **kwargs):
        if args and str(args[0]) == "rpu":
            args = ("cpu", *args[1:])
        if str(kwargs.get("device")) == "rpu":
            kwargs["device"] = "cpu"
        return original_to(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    events = []
    state = {"chunk": 0}
    def plan(owner, cache, length, position):
        events.append(("plan", position, length))
        if invalid == "body":
            raise RuntimeError("body has no native descriptor")
        return length, 32, SimpleNamespace(
            selected=SimpleNamespace(stage_tuple=SimpleNamespace(physical_descriptor=(length,))),
            graph_mode="RETAINED_CACHE", selection_scope="NATIVE", physical_plan_digest="physical",
            plan_digest="plan")
    def runner(owner, handle, *, inputs_embeds, past_key_values, **kwargs):
        count = inputs_embeds.shape[1]
        events.append(("run", past_key_values.position, count))
        state["chunk"] = 32 if count > 1 else 16
        owner._rpu_last_execution_plan = dict(position=past_key_values.position,
            logical_len=count, execution_len=count, padding_rows=0,
            chunk_size=state["chunk"], physical_descriptor=(count,), graph_mode="RETAINED_CACHE")
        past_key_values.update_position(count)
        raw = inputs_embeds if count > 1 else inputs_embeds[..., :1].expand(1, 1, 4)
        return raw, past_key_values
    monkeypatch.setattr(q, "_qwen3_vl_prefill_plan", plan)
    monkeypatch.setattr(q, "_text_decode_execution_plan", lambda *a, **k: None)
    monkeypatch.setattr(q, "_run_causal_decoder_forward", runner)
    monkeypatch.setattr(q, "make_zero_visual_embeds", lambda *a, **k: [])
    monkeypatch.setattr(q, "_qwen3_vl_text_graph_signature", lambda *a, **k: object())
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_get_resolved_chunk_size", lambda h: state["chunk"], raising=False)
    text = SimpleNamespace(_rpu_decoder_handle=7,
        _rpu_execution={"prefill": {"padding_budget": 0}},
        _rpu_text_graph_cache=SimpleNamespace(capture=lambda sig: nullcontext()))
    model = SimpleNamespace(model=SimpleNamespace(language_model=text, visual=SimpleNamespace(), rope_deltas=None),
        config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=2, vocab_size=4),
                               vision_config=SimpleNamespace(spatial_merge_size=2)),
        lm_head=lambda hidden: hidden[..., :1].expand(*hidden.shape[:-1], 4))
    model.lm_head.weight = torch.empty(4, 2, dtype=torch.float16)
    def head_projection(hidden, weight, bias, partition, num_cores, acc32, scale):
        assert weight is model.lm_head.weight and bias is None and scale is None
        assert partition == 1 and num_cores == 8 and acc32 is False
        return model.lm_head(hidden)
    monkeypatch.setattr(torch.ops.rpu, "linear_with_accumulation", head_projection, raising=False)
    cache = object.__new__(RPUCache)
    cache.position, cache.max_seq_len = 128, 256
    embeddings = torch.arange(33).half().reshape(1, 33, 1).expand(1, 33, 2).contiguous()
    arguments = dict(input_ids=torch.zeros(1, 33, dtype=torch.long),
        inputs_embeds=embeddings, past_key_values=cache, attention_mask=torch.ones(1, 33),
        position_ids=torch.arange(128, 161).reshape(1, 1, 33).expand(3, 1, 33),
        cache_position=torch.arange(128, 161), logits_to_keep=0)
    if invalid == "mask":
        arguments["attention_mask"][0, -1] = 0
    if invalid == "positions":
        arguments["position_ids"] = arguments["position_ids"][..., :-1]
    if invalid == "cache_positions":
        arguments["cache_position"] = torch.arange(33)
    if invalid:
        with pytest.raises((ValueError, NotImplementedError, RuntimeError)):
            q._rpu_qwen3vl_forward.__wrapped__(model, **arguments)
        assert cache.position == 128
        assert not any(event[0] == "run" for event in events)
    else:
        output = q._rpu_qwen3vl_forward.__wrapped__(model, **arguments)
        assert events[0] == ("plan", 128, 32)
        assert [event for event in events if event[0] == "run"] == [("run", 128, 32), ("run", 160, 1)]
        torch.testing.assert_close(output.logits[..., 0], embeddings[..., 0])
        assert cache.position == 161
        assert text._rpu_last_execution_plan["execution_kind"] == "segmented_append"
