"""Cold migration admission and the real direct-prefix Tensor owner boundary."""
from types import SimpleNamespace as NS
import ast
import inspect

import pytest
import torch

from rpu_backend.adapters.lingbot_vla_v2 import runtime


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for key in tuple(runtime.os.environ):
        if key.startswith(("RPU_LINGBOT2_", "RPU_L2_")):
            monkeypatch.delenv(key)


def test_runtime_graph_is_cold_and_routed_gemm_requires_dense(monkeypatch):
    assert not runtime._Lingbot2MoeColdConfig.from_env().runtime_graph
    monkeypatch.setenv("RPU_LINGBOT2_RUNTIME_GRAPH", "1")
    snapshot = runtime._Lingbot2MoeColdConfig.from_env()
    monkeypatch.setenv("RPU_LINGBOT2_RUNTIME_GRAPH", "0")
    assert snapshot.runtime_graph
    monkeypatch.setenv("RPU_LINGBOT2_ROUTED_GEMM", "1")
    with pytest.raises(ValueError, match="dense grouped"):
        runtime._Lingbot2MoeColdConfig.from_env()
    monkeypatch.setenv("RPU_LINGBOT2_GROUPED_EXPERTS", "1")
    assert runtime._Lingbot2MoeColdConfig.from_env().routed_gemm
    monkeypatch.setenv("RPU_LINGBOT2_SPARSE_EXPERTS", "1")
    with pytest.raises(ValueError, match="dense grouped"):
        runtime._Lingbot2MoeColdConfig.from_env()


def test_builder_preserves_runtime_graph_flush_choice_and_legacy_default():
    # Execute the actual builder statement containing the setting, without
    # loading checkpoint weights or importing a device runtime.
    builder = ast.parse(inspect.getsource(runtime._build_lingbot_vla_v2_impl)).body[0]
    statements = [statement for statement in builder.body if any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "set_ddr_flush" for node in ast.walk(statement))]
    assert len(statements) == 1
    code = compile(ast.Module(body=statements, type_ignores=[]), "<actual-builder-flush>", "exec")
    for runtime_graph in (False, True):
        for initial in (False, True):
            state, calls = {"normal": initial, "force": True}, []
            def set_flush(value):
                calls.append(value)
                state["normal"] = value
            exec(code, {"native_cold": NS(runtime_graph=runtime_graph),
                        "torch": NS(rpu=NS(set_ddr_flush=set_flush))})
            assert calls == ([] if runtime_graph else [True])
            assert state == {"normal": initial if runtime_graph else True, "force": True}


@pytest.mark.parametrize("conflict", ["RPU_LINGBOT2_MULTIVIEW_SPM_Z2", "RPU_LINGBOT2_PREFILL_W4A16"])
def test_runtime_graph_rejects_incompatible_owner_before_loading(monkeypatch, conflict):
    monkeypatch.setenv("RPU_LINGBOT2_RUNTIME_GRAPH", "1")
    monkeypatch.setenv(conflict, "1")
    with pytest.raises(ValueError, match=conflict):
        runtime._Lingbot2MoeColdConfig.from_env()


@pytest.mark.parametrize("setter_fails", [False, True])
def test_direct_prefix_passes_owned_tensors_and_publishes_after_binding(monkeypatch, setter_fails):
    original_to = torch.Tensor.to
    def cpu_to(tensor, *args, **kwargs):
        args = tuple("cpu" if isinstance(a, str) and a == "rpu" else a for a in args)
        if kwargs.get("device") == "rpu":
            kwargs["device"] = "cpu"
        return original_to(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    monkeypatch.setattr(runtime, "_validate_direct_prefix_group", lambda *a, **kw: None)
    monkeypatch.setattr(runtime, "_validate_prefix_scatter_runs", lambda *a, **kw: None)
    layout = NS(S=4, spans={})
    execution = NS(input_ids=torch.zeros((1, 4), dtype=torch.long),
        pad_masks=torch.ones((1, 4), dtype=torch.bool), mm_token_type_ids=None,
        visual_pos_masks=torch.tensor([[True, True, False, False]]), execution_len=4)
    monkeypatch.setattr(runtime, "build_prefix_layout", lambda *a: layout)
    monkeypatch.setattr(runtime, "pad_prefix_layout_for_execution", lambda *a, **kw: execution)
    events = []
    class Vision:
        _rpu_vision_batch_cap = 3
        _rpu_vision_kv_cache = NS(max_seq_len=1024)
        _rpu_vision_handle = 9
        _rpu_vision_graph_cache = object()
        def __call__(self, *args, **kwargs):
            assert policy._pfx_memo is not None
            events.append("vision")
    def bind(handle, embeds, dense, starts, lengths):
        assert handle == 9 and isinstance(embeds, torch.Tensor)
        assert len(dense) == 3 and all(isinstance(x, torch.Tensor) for x in dense)
        assert policy._pfx_memo is None
        events.append((embeds, dense))
        if setter_fails:
            raise RuntimeError("binding failed")
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_set_prefix_scatter", bind, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "spm_alloc_reset_temporary", lambda: None, raising=False)
    policy = NS(_visual=Vision(), _align_cfg=None, _ids=NS(eos=0), _align_tokens={},
        _pfx_memo=None, _vision_direct_prefix_no_output=True, _rope_index_takes_mm_types=False,
        _vgc=object(),
        _mrope_head_dim=128, _rope_theta=5_000_000, _mrope_section=[24, 20, 20],
        _vlm=NS(get_rope_index=lambda **kw: (torch.arange(4).repeat(3, 1, 1), None),
            get_input_embeddings=lambda: lambda ids: ids.unsqueeze(-1).expand(
                1, 4, runtime.VLM_HID).half().clone()))
    args = (torch.zeros((1, 4, 6)), [0], torch.tensor([[1]]), None,
            torch.tensor([[1, 2, 2]]), 2, NS(real_len=4, execution_len=4, a6_plan=object()))
    call = runtime.LingbotVlaV2Policy._build_prefix_direct
    if setter_fails:
        with pytest.raises(RuntimeError, match="binding failed"):
            call(policy, *args)
        assert policy._pfx_memo is None and len(events) == 1
    else:
        result = call(policy, *args)
        assert result[0] is events[0][0] and result[3] is events[0][1]
        assert call(policy, *args)[0] is result[0]
        assert events[1:] == ["vision", "vision"]
        assert result[2].shape == (4, 3) and result[2].dtype == torch.int32
        assert result[4].shape == result[5].shape == (4, 64)
        owners = [result[0], result[2], *result[3], result[4], result[5]]
        addresses = [x.data_ptr() for x in owners]
        execution.input_ids.fill_(7)
        changed_args = (*args[:2], torch.tensor([[2]]), *args[3:])
        refreshed = call(policy, *changed_args)
        assert refreshed[0] is result[0] and bool(refreshed[0].eq(7).all())
        assert [x.data_ptr() for x in [refreshed[0], refreshed[2], *refreshed[3], refreshed[4], refreshed[5]]] == addresses
        assert len([x for x in events if isinstance(x, tuple)]) == 1

        def reject_frozen(caches):
            assert len(caches) == 2
            raise RuntimeError("frozen targets")
        monkeypatch.setattr(runtime, "_clear_graph_caches_for_rebind", reject_frozen)
        changed_layout = (*changed_args[:4], torch.tensor([[1, 1, 4]]), *args[5:])
        with pytest.raises(RuntimeError, match="frozen targets"):
            call(policy, *changed_layout)
        assert bool(result[0].eq(7).all())


@pytest.mark.parametrize("prime_fails", [False, True])
def test_resident_banks_reserve_all_before_prime_and_never_publish_partial(monkeypatch, prime_fails):
    events = []
    prefixes = ("qwen3vl_vision", "causal_decoder", "lingbot_v2_moe")
    for prefix in prefixes:
        def enable(handle, prefix=prefix):
            events.append(("enable", prefix))
        def reserve(*args, prefix=prefix):
            assert sum(event[0] == "enable" for event in events) == 3
            events.append(("reserve", prefix))
        def prime(handle, descriptor, prefix=prefix):
            assert sum(event[0] == "reserve" for event in events) == 3
            assert descriptor == [11, 22]
            events.append(("prime", prefix))
            if prime_fails and prefix == "causal_decoder":
                raise RuntimeError("injected preload failure")
        monkeypatch.setattr(torch.ops.rpu, prefix + "_enable_resident_preloads", enable, raising=False)
        monkeypatch.setattr(torch.ops.rpu, prefix + ("_prepare_persistent_spm" if prefix == "causal_decoder"
                                                   else "_prepare_resident_spm"), reserve, raising=False)
        monkeypatch.setattr(torch.ops.rpu, prefix + "_prime_resident_preloads", prime, raising=False)
    policy = runtime.LingbotVlaV2Policy.__new__(runtime.LingbotVlaV2Policy)
    policy._lingbot2_moe_cold_config = NS(runtime_graph=True)
    policy._visual = NS(_rpu_vision_handle=1, _rpu_vision_batch_cap=3,
                        _rpu_vision_kv_cache=NS(max_seq_len=1024))
    policy._text = NS(_rpu_decoder_handle=2)
    policy._exp = 3
    plan = NS(selected=NS(stage_tuple=NS(physical_descriptor=(11, 22))))
    policy._action_execution_plan = lambda *args: plan
    prefill = NS(execution_len=224, a6_plan=plan)
    metadata = dict(grid_key=(1, 16, 16) * 3, keep_img=(0, 1, 2), image_shape=(3, 256, 1536))
    assert policy._reserve_runtime_parameters(metadata, prefill)
    assert not policy._reserve_runtime_parameters(metadata, prefill)
    assert policy._runtime_resident_state["phase"] == "RESERVED"
    if prime_fails:
        with pytest.raises(RuntimeError, match="injected preload"):
            policy._prime_runtime_parameters((plan,), prefill)
        assert policy._runtime_resident_state["phase"] == "FAILED"
        count = len(events)
        with pytest.raises(RuntimeError, match="recreate"):
            policy._reserve_runtime_parameters(metadata, prefill)
        assert len(events) == count
    else:
        policy._prime_runtime_parameters((plan,), prefill)
        assert policy._runtime_resident_state["phase"] == "PRIMED"
        policy._prime_runtime_parameters((plan,), prefill)
        assert len(events) == 9
        with pytest.raises(ValueError, match="geometry changed"):
            policy._reserve_runtime_parameters(metadata, NS(execution_len=240))
        assert len(events) == 9
