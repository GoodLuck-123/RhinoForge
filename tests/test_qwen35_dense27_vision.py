"""Official 27B Vision cold ownership; host checks do not execute the model."""
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
from rpu_backend.adapters.qwen3_5 import cores, vision
from rpu_backend.graph import GraphRuntimePolicy
from test_qwen35_core_profiles import graph_factories, vision_install_boundary


def _policy(root=8, **kwargs):
    return GraphRuntimePolicy(execution_core_count=root, graph_arena_count=4,
        qwen35_dense_27b_sdk_budget=True, lkn_max_batch_entries=999,
        lkn_kd_buf_mb=12, lkn_instr_buf_mb=48, **kwargs)


def _state(root=8):
    policy = _policy(root)
    return NS(vision_arena_admitted=True, graph_cache=NS(runtime_policy=policy),
        prefill_graph=NS(runtime_policy=policy),
        execution_topology=cores.core_topology({"model": {"num_cores": root}}))


def _config():
    return NS(depth=27, num_heads=16, hidden_size=1152, intermediate_size=4304,
        out_hidden_size=5120, hidden_act="gelu_pytorch_tanh", in_channels=3,
        patch_size=16, temporal_patch_size=2, spatial_merge_size=2,
        num_position_embeddings=2304, deepstack_visual_indexes=[])


@pytest.mark.parametrize("root,expected", [(4, 4), (6, 4), (8, 8)])
def test_modern27_lazy_vision_reuses_the_frozen_root_budget(root, expected, monkeypatch):
    state = _state(root)
    root_policy = state.graph_cache.runtime_policy
    monkeypatch.setenv("RPU_FASTREPLAY_SKIP_SYNC", "invalid-after-cold-snapshot")
    monkeypatch.setenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", "0")
    options = vision._lazy_vision_graph_options(state)
    policy = options["_graph_runtime_policy"]
    assert options["_cold_numeric_opt_in"] is True
    assert policy == replace(root_policy, execution_core_count=expected)
    assert (policy is root_policy) is (root != 6)
    assert policy.qwen35_dense_27b_sdk_budget and not policy.qwen35_legacy_27b_sdk_budget
    assert vision._lazy_vision_graph_options(state)["_graph_runtime_policy"] is policy


@pytest.mark.parametrize("fault", ["untyped", "prefill_missing", "prefill_copy",
    "topology_missing", "topology_roles", "topology_root", "owner_core"])
def test_modern27_lazy_vision_rejects_inconsistent_text_owner_before_caching(fault):
    state = _state(6)
    policy = state.graph_cache.runtime_policy
    if fault == "untyped":
        state.graph_cache.runtime_policy = NS(**vars(policy))
    elif fault == "prefill_missing":
        del state.prefill_graph
    elif fault == "prefill_copy":
        state.prefill_graph.runtime_policy = replace(policy)
    elif fault == "topology_missing":
        del state.execution_topology
    elif fault == "topology_roles":
        state.execution_topology = NS(num_cores=6, attn_tp=8, mlp_tp=6,
            lm_head_tp=4, physical_kv_cores=8)
    elif fault == "topology_root":
        state.execution_topology = NS(num_cores=5, attn_tp=4, mlp_tp=5,
            lm_head_tp=4, physical_kv_cores=8)
        state.graph_cache.runtime_policy = _policy(5)
    else:
        state.graph_cache.runtime_policy = _policy(8)
    with pytest.raises(RuntimeError, match="cold.*owner|arena ownership"):
        vision._lazy_vision_graph_options(state)
    assert not hasattr(state, "vision_graph_policy")


@pytest.mark.parametrize("root", [4, 6, 8])
@pytest.mark.parametrize("fault", ["budget", "sdk", "core", "identity"])
def test_modern27_lazy_vision_rejects_stale_vision_policy(root, fault):
    state = _state(root)
    good = vision._lazy_vision_graph_options(state)["_graph_runtime_policy"]
    if fault == "budget":
        bad = replace(good, qwen35_dense_27b_sdk_budget=False)
    elif fault == "sdk":
        bad = replace(good, lkn_kd_buf_mb=13)
    elif fault == "core":
        bad = replace(good, execution_core_count=6)
    else:
        bad = replace(good)
    state.vision_graph_policy = bad
    if fault == "identity" and root == 6:
        # Root6 necessarily has a separate Vision4 policy with the same frozen fields.
        assert vision._lazy_vision_graph_options(state)["_graph_runtime_policy"] is bad
    else:
        with pytest.raises(RuntimeError, match="differs from its cold owner"):
            vision._lazy_vision_graph_options(state)


@pytest.mark.parametrize("root,vision_cores", [(4, 4), (6, 4), (8, 8)])
def test_modern27_actual_vision_install_uses_one_shared_fp16_policy(
    monkeypatch, graph_factories, vision_install_boundary, root, vision_cores,
):
    _, visual, events = vision_install_boundary
    visual.config = _config()
    visual.blocks += visual.blocks[:3]
    # Production raw shape/storage checks have separate tests. This boundary
    # fixture executes the real installer with small CPU/native doubles.
    checks = []
    monkeypatch.setattr(cores, "validate_dense_large_vision_weights",
        lambda model, config: checks.append(("geometry", config.out_hidden_size)))
    monkeypatch.setattr(cores, "validate_cold_weight_dtype",
        lambda model, **kwargs: checks.append(("storage", kwargs)))
    monkeypatch.setattr(GraphRuntimePolicy, "prepare_arenas",
        lambda _: pytest.fail("lazy Vision must not prepare another arena pool"))
    state = _state(root)
    options = vision._lazy_vision_graph_options(state)
    assert vision.install_qwen3_5_vision_for_rpu(visual,
        execution_config={"model": {"num_cores": root}}, **options) == 23
    policy = options["_graph_runtime_policy"]
    assert visual._rpu_vision_graph_cache.runtime_policy is policy
    assert visual._rpu_vision_debug_graph.runtime_policy is policy
    assert policy.execution_core_count == vision_cores
    assert policy.graph_arena_count == 4 and policy.qwen35_dense_27b_sdk_budget
    assert visual._rpu_vision_w8a16 is False
    assert graph_factories[0][1]["max_entries"] == 1
    assert checks == [("geometry", 5120), ("storage", {"fp16": True})]
    assert "weight-transform" in events and "set_weights" in events
    for block in visual.blocks:
        block._rpu_qwen3_5_vision_weights_converted = True
    assert vision.install_qwen3_5_vision_for_rpu(visual,
        execution_config={"model": {"num_cores": root}}, **options) == 23
    with pytest.raises(RuntimeError, match="policy is immutable"):
        vision.install_qwen3_5_vision_for_rpu(visual,
            execution_config={"model": {"num_cores": root}},
            _graph_runtime_policy=replace(policy), _cold_numeric_opt_in=True)
    assert events.count("create") == 1


@pytest.mark.parametrize("fault", ["missing_policy", "ordinary_policy", "no_opt_in",
    "legacy_policy", "9b_with_modern_budget", "nearby_width", "depth", "quantized_vision"])
def test_modern27_vision_rejects_profile_or_policy_mismatch_before_mutation(monkeypatch, fault):
    config = _config()
    visual = NS(config=config, blocks=[object()] * 27, merger=object())
    policy, opt_in = _policy(), True
    if fault == "missing_policy":
        policy = None
    elif fault == "ordinary_policy":
        policy = replace(policy, qwen35_dense_27b_sdk_budget=False)
    elif fault == "no_opt_in":
        opt_in = False
    elif fault == "legacy_policy":
        policy = GraphRuntimePolicy(qwen35_legacy_27b_sdk_budget=True)
    elif fault == "9b_with_modern_budget":
        config.out_hidden_size = 4096
    elif fault == "nearby_width":
        config.out_hidden_size = 5136
    elif fault == "depth":
        config.depth = 26
    else:
        config.quant_config = {"method": "w8a16"}
    monkeypatch.setattr(vision, "_install_qwen3_5_vision_for_rpu_impl",
        lambda *a, **kw: pytest.fail("invalid cold profile reached weight mutation"))
    before = dict(vars(visual))
    with pytest.raises(ValueError):
        vision.install_qwen3_5_vision_for_rpu(visual,
            _graph_runtime_policy=policy, _cold_numeric_opt_in=opt_in)
    assert vars(visual) == before
