import ast
import re
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_HEADER = _ROOT / "src/fused/rpu_qwen3_model.h"
_CPP = _ROOT / "src/fused/rpu_qwen3_model.cpp"


def _qualified_name(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))




def test_causal_decoder_requires_complete_descriptor_and_typed_kv_routes():
    header = _HEADER.read_text(encoding="utf-8")
    cpp = _CPP.read_text(encoding="utf-8")
    assert "planned_chunk_size == 0 && !planned_stage_descriptor.empty()" in header
    assert "manifest.state = FmbPhysicalManifestState::COMPLETE" in header
    assert "physical_manifest_forward_capability" in header
    assert "return {count == 0 || count == 2, manifest.graph_lifecycle};" in header
    assert "return {false, manifest.graph_lifecycle};" in header
    assert "seen[route.invocation]" in header
    assert "rpu_launch_insert_kvcache_spm_unified(" not in header
    assert "rpu_launch_insert_kvcache_spm_unified_with_plan(" in header
    assert "KV_INSERT_ROUTE_FLAG_DYNAMIC_POSITION" in header
    assert "rpu_rebase_kvinsert_segment_plan_position(" in header
    assert "resolve_decode_stage_descriptor" in header
    assert "rpu_causal_decoder_resolve_decode_stage_descriptor" in cpp


def test_internal_composite_decoder_forwards_pass_complete_descriptors():
    text = _CPP.read_text(encoding="utf-8")
    for member in (
        "wall_z1_stage_descriptor_",
        "qwen3vl_pooler_z1_stage_descriptor_",
        "qwen3vl_multiview_text_stage_descriptor_",
    ):
        assert f"{member} =" in text or f"{member} =\n" in text
        assert f"{member}.empty()" in text
        assert re.search(rf"/\*planned_chunk_size=\*/0,\s*{member}", text)
    assert text.count("FmbGraphLifecycle::COMPOSITE_CHILD") >= 3


def test_every_production_python_causal_forward_passes_a_descriptor():
    calls = []
    for path in (_ROOT / "python/rpu_backend").rglob("*.py"):
        if "tests" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _qualified_name(node.func) != "torch.ops.rpu.causal_decoder_forward":
                continue
            calls.append((path, node))
            keywords = {item.arg: item.value for item in node.keywords}
            descriptor = keywords.get("planned_stage_descriptor")
            if descriptor is None and len(node.args) >= 16:
                descriptor = node.args[15]
            assert descriptor is not None, f"{path}:{node.lineno}"
            assert not (
                isinstance(descriptor, (ast.List, ast.Tuple))
                and not descriptor.elts
            ), f"{path}:{node.lineno} passes an empty descriptor"
    assert calls


def test_shared_runtime_installs_keys_and_requires_decode_descriptor():
    text = (_ROOT / "python/rpu_backend/runtime/decoder.py").read_text(
        encoding="utf-8"
    )
    assert '"_rpu_decode_stage_descriptor"' in text
    assert "causal_decoder_resolve_decode_stage_descriptor(" in text
    assert "causal decoder decode has no installed physical descriptor" in text
    assert 'getattr(model, "_rpu_decode_stage_descriptor", ())' in text
    assert "if seq_len == 1 and not planned_stage_descriptor:" in text
    assert "planned_graph_mode = GRAPH_RETAINED_CACHE" in text
    assert '"graph_mode": planned_graph_mode' in text


def test_causal_descriptor_ops_have_explicit_registered_schemas():
    registrations = (
        _ROOT / "src/core/rpu_dispatch_registrations.inc"
    ).read_text(encoding="utf-8")
    assert 'm.def("causal_decoder_resolve_prefill_stage_domain(' in registrations
    assert "int rope_mode=0, int graph_lifecycle=1" in registrations
    assert ('m.def("causal_decoder_resolve_decode_stage_descriptor('
            'int handle, int position=0) -> int[]"') in registrations
