"""Board-free contracts for Pi0.5 AdaRMS native planner authority."""

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_adarms_complete_descriptor_controls_every_configurable_site():
    native = (ROOT / "src/fused/rpu_adarms_model.cpp").read_text(
        encoding="utf-8"
    )
    adapter = (
        ROOT / "python/rpu_backend/adapters/pi05/adarms.py"
    ).read_text(encoding="utf-8")
    registrations = (
        ROOT / "src/core/rpu_dispatch_registrations.inc"
    ).read_text(encoding="utf-8")

    assert "resolve_action_stage_domain(" in native
    assert "physical_manifest_for_candidate(" in native
    assert "FmbGraphLifecycle::COMPOSITE_CHILD" in native
    for site_id in (
        "1365898338277373764",  # mutable cond DMA
        "3720663967042490210",  # Q linear
        "737407077005117954",   # K linear
        "2757305400171169493",  # V linear
        "5408876450787035953",  # Q RoPE
        "7158713397924778966",  # K RoPE
        "1489108772795312297",  # typed KV insert
        "892830053731625482",   # DDR-required attention
        "2713753424009463107",  # ring prepare
        "8920591606190302727",  # O linear
        "1378336082325261091",  # attention all-reduce
        "73920063418775617",    # AdaRMS GEMV linear
        "3363421386255464359",  # AdaRMS GEMV all-reduce
    ):
        assert site_id in native
    assert "consume_physical_attention_route(" in native
    assert "restore_kvinsert_plan(" in native
    assert "ADARMS_KV_INSERT_SITE, manifest.graph_lifecycle," in native
    assert "rpu_kvinsert_segment_plan_from_route_arguments(" not in native
    assert "rpu_launch_insert_kvcache_spm_unified_with_plan(" in native
    assert "rpu_launch_insert_kvcache_spm_unified(" not in native

    assert "_plan_adarms_action_execution(" in adapter
    assert "_selected_component_plan.stage_tuple.physical_descriptor" in adapter
    assert '"authority": "NATIVE_A6_STAGE_DESCRIPTOR"' in adapter
    assert '"attention_policy": "DDR_REQUIRED"' in adapter
    assert "*_component_plan.graph_key_words()" in adapter
    assert "adarms_resolve_action_stage_domain" in registrations
    assert "int[] planned_stage_descriptor=[]" in registrations



def test_adarms_passthrough_escape_hatch_is_diagnostic_and_drift_checked():
    adapter = (
        ROOT / "python/rpu_backend/adapters/pi05/adarms.py"
    ).read_text(encoding="utf-8")
    assert "DIAGNOSTIC_ONLY" in adapter
    assert "dry/forward chunk plan drift" in adapter
    assert "adarms_get_resolved_chunk_size" in adapter
