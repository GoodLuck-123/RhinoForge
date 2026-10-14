"""A registered fixed physical stage is not a new public execution parser."""
from types import SimpleNamespace

import pytest

from rpu_backend.api import _execution as api
from rpu_backend.runtime import execution_planner as planner
from test_execution_session_costs import binding


@pytest.mark.parametrize("fields", [(), [], set()])
def test_fixed_decode_declaration_is_frozen_but_absent_from_public_union(fields):
    declared = {"decoder": {"decode": fields}, "vision": {"vision": ("chunk_size",)}}
    capabilities, union = api._component_capabilities(declared, entry_point="fixed-test")
    assert capabilities["decoder"]["decode"] == ()
    assert union == {"vision": {"chunk_size"}}
    if isinstance(fields, list):
        fields.append("chunk_size")
        assert capabilities["decoder"]["decode"] == ()
    assert api.resolve_component_rpu_execution({}, "decoder", entry_point="fixed-test",
        supported_components=capabilities, profile_auto={}) == {}


@pytest.mark.parametrize("stage,fields", [
    ("decode", ("chunk_size",)), ("decode", ("padding_rows",)),
    ("decode", ("fast_replay",)), ("decode", ""), ("other", ()),
])
def test_fixed_decode_cannot_declare_configurable_or_unknown_fields(stage, fields):
    with pytest.raises((TypeError, ValueError)):
        api._component_capabilities({"decoder": {stage: fields}}, entry_point="fixed-test")


@pytest.mark.parametrize("value", [
    {"decode": {}}, {"components": {"decoder": {"decode": {}}}},
    {"components": {"decoder": {"decode": {"chunk_size": "auto"}}}},
    {"components": {"decoder": {"decode": {"chunk_size": 16}}}},
])
def test_registered_decode_is_still_rejected_by_the_user_configuration_parser(value):
    with pytest.raises(ValueError, match="unknown stage"):
        api.normalize_rpu_execution(value, entry_point="fixed-test",
            supported_components={"decoder": {"decode": ()}})


def test_real_session_admits_only_the_declared_fixed_component_cost_binding():
    owner, child = SimpleNamespace(), SimpleNamespace()
    session = api.bind_execution_session(owner, {}, entry_point="fixed-test",
        supported_components={"decoder": {"decode": ()}, "other": {}},
        graph_mode="RETAINED_CACHE")
    session._bind_planner_owner(child, "decoder")
    api.record_planner_native(child, "decode", "decoder", ("causal_decoder", 7))
    expected = binding(planner)
    session._install_planner_costs("decoder", "decode", lambda: expected,
        native_prefix="causal_decoder")
    assert api.planner_cost_binding(child, "decode", "decoder", native_prefix="causal_decoder") == expected
    for component, stage in (("other", "decode"), ("decoder", "prefill"), ("missing", "decode")):
        with pytest.raises(ValueError):
            api.planner_cost_binding(child, stage, component)
        with pytest.raises(ValueError):
            session._install_planner_costs(component, stage, lambda: pytest.fail("foreign install"))
    before = session.stats()
    with pytest.raises(ValueError, match="unknown stage"):
        session.reconfigure({"components": {"decoder": {"decode": {}}}})
    assert session.stats()["generation"] == before["generation"]
    assert session.config == {}
