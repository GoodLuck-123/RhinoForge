"""Board-free contracts for Gemma-family native planner authority."""

from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_gemma2_offset_norm_maps_once_without_mutating_source():
    import torch
    from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm
    from rpu_backend.adapters.paligemma.gemma2 import _gemma2_norm_gamma
    from rpu_backend.api.errors import RPUBackendError

    norm = Gemma2RMSNorm(16)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(-0.5, 0.5, 16))
    before, pointer, version = norm.weight.detach().clone(), norm.weight.data_ptr(), norm.weight._version
    first, second = _gemma2_norm_gamma(norm), _gemma2_norm_gamma(norm)
    assert torch.equal(first, (1.0 + before).half())
    assert torch.equal(first, second)
    assert first.data_ptr() != norm.weight.data_ptr()
    assert norm.weight.data_ptr() == pointer and norm.weight._version == version
    assert torch.equal(norm.weight, before)
    gamma_norm = torch.nn.RMSNorm(16)
    assert torch.equal(_gemma2_norm_gamma(gamma_norm), torch.ones(16, dtype=torch.float16))
    with pytest.raises(RPUBackendError, match="actual Gemma2RMSNorm"):
        _gemma2_norm_gamma(torch.nn.LayerNorm(16))
    class ReinterpretedNorm(Gemma2RMSNorm):
        def forward(self, value):
            return value * self.weight
    with pytest.raises(RPUBackendError, match="actual Gemma2RMSNorm"):
        _gemma2_norm_gamma(ReinterpretedNorm(16))
    norm.weight.data[0] = float("nan")
    with pytest.raises(RPUBackendError, match="finite"):
        _gemma2_norm_gamma(norm)
    norm.weight.data[0] = 1e10
    with pytest.raises(RPUBackendError, match="finite"):
        _gemma2_norm_gamma(norm)


@pytest.mark.parametrize("family", ["gemma2", "qwenpi05"])
@pytest.mark.parametrize("failure", [False, True])
def test_explicit_component_close_is_strict_but_gc_stays_best_effort(monkeypatch, family, failure):
    import weakref
    import torch
    from rpu_backend.api._execution import bind_execution_session
    from rpu_backend.adapters.paligemma import gemma2, _cleanup_gemma2_handle
    from rpu_backend.adapters.qwenpi05 import fused

    class Owner:
        pass

    owner, events = Owner(), []
    cache = SimpleNamespace(begin_warmup=lambda: None, clear=lambda: events.append("clear"))
    handle_attr = f"_rpu_{family}_handle"
    finalizer_attr = f"_rpu_{family}_handle_finalizer"
    cache_attr = f"_rpu_{family}_graph_cache"
    setattr(owner, handle_attr, 17)
    setattr(owner, cache_attr, cache)
    owner._rpu_execution = {}
    if family == "gemma2":
        gc_destroy = gemma2._gemma2_destroy_handle
        close = _cleanup_gemma2_handle
        components = gemma2._GEMMA2_EXECUTION_COMPONENTS
    else:
        gc_destroy = fused._qwenpi05_destroy_handle
        close = fused._close_qwenpi05_runtime
        components = fused._QWENPI05_EXECUTION_COMPONENTS

    def destroy(handle):
        assert handle == 17
        events.append("destroy")
        if failure:
            raise RuntimeError("injected native destroy failure")

    monkeypatch.setattr(torch.ops.rpu, f"{family}_destroy", destroy, raising=False)
    finalizer = weakref.finalize(owner, gc_destroy, 17)
    setattr(owner, finalizer_attr, finalizer)
    session = bind_execution_session(owner, {}, entry_point=family,
        supported_components=components, graph_mode="COMPOSITE_CHILD")
    try:
        if failure:
            with pytest.raises(RuntimeError, match="injected native destroy failure"):
                close(owner)
            assert session.stats()["state"] == "POISONED"
            assert getattr(owner, handle_attr) == 17
            assert getattr(owner, cache_attr) is cache
            assert getattr(owner, finalizer_attr) is finalizer and finalizer.alive
            with pytest.raises(RuntimeError, match="shutdown failed"):
                close(owner)
            assert events == ["clear", "destroy"]
            # The still-owned GC fallback may be retried and must remain best effort.
            finalizer()
            assert events == ["clear", "destroy", "destroy"]
        else:
            close(owner)
            assert session.stats()["state"] == "CLOSED" and not finalizer.alive
            close(owner)
            assert events == ["clear", "destroy"]
    finally:
        finalizer.detach()


def test_gemma2_diagnostic_owner_uses_real_session_for_rebuild_and_close(monkeypatch):
    import torch
    from rpu_backend.api._execution import bind_execution_session
    from rpu_backend.adapters.paligemma import gemma2, _cleanup_gemma2_handle

    events = []
    class Finalizer:
        alive = True
        def detach(self):
            self.alive = False
    monkeypatch.setattr(torch.ops.rpu, "gemma2_destroy", lambda _: events.append("destroy"), raising=False)
    owner = SimpleNamespace(
        _rpu_execution={}, _fmb_execution_generation=0,
        _rpu_gemma2_graph_cache=SimpleNamespace(
            begin_warmup=lambda: events.append("warmup"),
            clear=lambda: events.append("clear"),
        ),
        _rpu_cache=SimpleNamespace(reset_to_position=lambda pos: events.append(("kv", pos))),
        _rpu_gemma2_handle_finalizer=Finalizer(), _rpu_gemma2_handle=17,
    )
    session = bind_execution_session(
        owner, {}, entry_point="Gemma2 diagnostic",
        supported_components=gemma2._GEMMA2_EXECUTION_COMPONENTS,
        validate=gemma2._normalize_gemma2_execution,
        apply=lambda old, new, gen, **kw: gemma2._apply_gemma2_execution(owner, old, new, gen, **kw),
        rollback=lambda old, new, gen: gemma2._apply_gemma2_execution(owner, new, old, gen),
        graph_mode="COMPOSITE_CHILD",
    )
    session._bind_planner_owner(owner, "language_model")
    with session.execute():
        with pytest.raises(RuntimeError, match="during a forward"):
            _cleanup_gemma2_handle(owner)
    assert events == []
    session._reconfigure({}, force_rebuild=True)
    assert events == ["warmup", "clear", ("kv", 0)]
    assert owner._fmb_execution_generation == 1
    session.reconfigure({"prefill": {"chunk_size": 32}})
    assert owner._fmb_execution_generation == 2
    assert owner._rpu_execution["prefill"]["chunk_size"] == 32
    _cleanup_gemma2_handle(owner)
    assert events[-3:] == ["warmup", "clear", "destroy"]
    assert session.stats()["state"] == "CLOSED"
    _cleanup_gemma2_handle(owner)
    assert events.count("destroy") == 1


def test_gemma2_cold_planner_binds_actual_component_handle_and_generation(monkeypatch):
    from rpu_backend.adapters.paligemma import gemma2
    from rpu_backend.runtime import decoder

    owner = SimpleNamespace(
        _rpu_execution={"components": {"language_model": {"prefill": {"chunk_size": 32}}}},
        _rpu_gemma2_handle=73, _fmb_execution_generation=4,
        _rpu_gemma2_graph_cache=object(),
    )
    result = SimpleNamespace(selected=SimpleNamespace(stage_tuple=SimpleNamespace(physical_descriptor=(3, 1))))
    calls = []
    def plan(*args, **kwargs):
        calls.append((args, kwargs))
        kwargs["plan_result_sink"](result)
    monkeypatch.setattr(decoder, "plan_bounded_prefill_execution", plan)
    assert gemma2._plan_gemma2_execution(owner, seq_len=64, position=16, max_seq_len=128) is result
    args, kwargs = calls[0]
    assert args == (64, 112, 0)
    assert kwargs["execution_owner"] is owner
    assert kwargs["execution_component"] == "language_model"
    assert kwargs["execution_native"] == ("gemma2", 73)
    assert kwargs["graph_cache"] is owner._rpu_gemma2_graph_cache
    assert kwargs["exact_chunk_size"] == 32
    assert kwargs["physical_metadata"] == (("execution_generation", 4),)


def test_gemma_and_gemma2_complete_descriptors_cover_direct_routes():
    gemma = _read("src/fused/rpu_gemma_model.cpp")
    gemma2 = _read("src/fused/rpu_gemma2_model.cpp")

    for native, prefix, sites in (
        (
            gemma,
            "GEMMA",
            (
                "5100611559110851625", "1917889831839164982",
                "6163682617036203929", "7810258305288658911",
                "7733874484478142313", "3784284474306460535",
                "5049433632783468136", "5307885341939952186",
                "8164586374905126555", "8495030819554234104",
            ),
        ),
        (
            gemma2,
            "GEMMA2",
            (
                "1924793764763452535", "7642448968429412903",
                "4180536231153114197", "628747048398910094",
                "5903063903249551868", "6268593493954406303",
                "922579874619257911", "2738429165746498932",
                "208812815686546690", "3022200895558016477",
                "4029039243110208410", "6288278170806612905",
                "3373592289648928740", "6726693963906483239",
            ),
        ),
    ):
        assert "physical_manifest_for_candidate(" in native
        assert f"{prefix}_FIXED_KERNEL_BASIS" in native
        assert "FmbPhysicalManifestState::COMPLETE" in native
        assert "FmbGraphLifecycle::COMPOSITE_CHILD" in native
        assert "chunk.idx" in native
        assert "KV_INSERT_CAP_V2" in native
        assert "rpu_launch_insert_kvcache_spm_unified_with_plan(" in native
        assert "rpu_launch_insert_kvcache_spm_unified(" not in native
        assert (
            "consume_physical_attention_route(" in native
            or "ctx().consume_physical_route(" in native
        )
        assert "append_fmb_shared_runtime_routes(" in native
        for site in sites:
            assert site in native, f"{prefix} route site {site} disappeared"

    assert "FMB_SHARED_MLP_AUTO_TILE" in gemma
    assert "FMB_SHARED_MLP_RING_REDUCE" in gemma
    assert "int64_t best_cs = 320" not in gemma2


def test_gemma4_descriptor_is_retained_and_mixed_geometry_is_ddr_required():
    native = _read("src/fused/rpu_gemma4_model.cpp")
    sites = (
        "8725399426442661120", "6916943715144097592",
        "3870327977122590662", "723594843514692113",
        "5337083984893514264", "6562867400688913245",
        "6581941412312031012", "5286999247906518060",
        "8703745226169185634", "2468806415651175396",
        "728529537003975745", "668941020522880346",
        "146067009784919675", "4396400591359725542",
        "6582489299560935010", "7599842320145438531",
        "7972588270602457603", "7105886685527167396",
    )

    assert "resolve_stage_domain(" in native
    assert "physical_manifest_for_candidate(" in native
    assert "GEMMA4_FIXED_KERNEL_BASIS" in native
    assert "FmbPhysicalManifestState::COMPLETE" in native
    assert "FmbGraphLifecycle::RETAINED_CACHE" in native
    assert "gemma4_kv_invocation(chunk.idx, global)" in native
    assert "KV_INSERT_ROUTE_FLAG_DYNAMIC_POSITION" in native
    assert "rpu_rebase_kvinsert_segment_plan_position" in native
    assert "DDR_KV is required" in native
    assert "AttentionExecutionPolicy::DDR_KV" in native
    assert "rpu_launch_insert_kvcache_spm_unified_with_plan(" in native
    assert "rpu_launch_insert_kvcache_spm_unified(" not in native
    assert "consume_physical_attention_route(" in native
    assert "append_fmb_shared_runtime_routes(" in native
    for site in sites:
        assert site in native


def test_gemma_composite_adapters_forward_the_full_native_descriptor():
    gemma2 = _read("python/rpu_backend/adapters/paligemma/gemma2.py")
    gemma4 = _read("python/rpu_backend/adapters/gemma4/__init__.py")
    pi05 = _read("python/rpu_backend/adapters/pi05/runtime.py")
    registrations = _read("src/core/rpu_dispatch_registrations.inc")
    declarations = _read("src/core/rpu_kernel_decls.h")

    assert "plan_bounded_prefill_execution(" in gemma2
    assert "gemma2_resolve_prefill_stage_domain" in gemma2
    assert "selected.stage_tuple.physical_descriptor" in gemma2
    assert "model._rpu_execution = execution_config" in gemma2

    assert "@execution_serialized" in gemma4
    assert "bind_execution_session(" in gemma4
    assert "gemma4_resolve_stage_domain" in gemma4
    assert "position=int(canonical_position)" in gemma4
    assert "selected.stage_tuple.physical_descriptor" in gemma4
    assert "*planned.graph_key_words()" in gemma4
    assert "def reconfigure_rpu_execution" in gemma4

    pi05_language_plan = pi05.split(
        "def _plan_pi05_prefix_execution", 1
    )[1].split("\ndef ", 1)[0]
    assert "gemma_resolve_prefill_stage_domain" in pi05_language_plan
    assert "graph_mode=GRAPH_COMPOSITE_CHILD" in pi05_language_plan

    for symbol in (
        "rpu_gemma2_resolve_prefill_stage_domain",
        "rpu_gemma4_resolve_stage_domain",
    ):
        assert symbol in declarations
        assert symbol in registrations
    assert registrations.count("int[] planned_stage_descriptor=[]") >= 3
