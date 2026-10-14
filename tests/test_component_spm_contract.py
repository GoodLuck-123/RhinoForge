"""Check the real Pi0.5 wrapper/helper chain and reject broken reset credit."""

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "component_spm_contract", ROOT / "scripts/check_component_spm_contract.py"
)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)
GEMMA = "python/rpu_backend/adapters/pi05/gemma.py"


def test_current_component_spm_contracts():
    failures = []
    for path, (patch, native) in checker.TARGET_FILES.items():
        checker._check_file(path, patch, native, failures)
    assert not failures


@pytest.mark.parametrize("fault", [
    "wrong_helper", "rebound_helper", "wrong_forward_mode", "wrong_kv_mode",
    "wrong_native_op", "wrong_native_receiver", "rebound_native_alias",
    "uncalled_native_alias", "wrong_graph_callee", "escaped_native_alias", "rebound_mode",
    "wrong_reset_receiver", "reset_before_native", "reset_in_uncalled_function",
    "reset_inside_capture",
])
def test_pi05_helper_chain_cannot_borrow_unrelated_calls(tmp_path, monkeypatch, fault):
    source = (ROOT / GEMMA).read_text()
    if fault == "wrong_helper":
        source = source.replace("return _run_gemma_prefill(", "return unrelated_prefill(", 1)
    elif fault == "rebound_helper":
        source = source.replace("    def rpu_gemma_model_forward(",
                                "    _run_gemma_prefill = unrelated_prefill\n\n    def rpu_gemma_model_forward(")
    elif fault == "wrong_forward_mode":
        source = source.replace("_kv_only=False, **kwargs", "_kv_only=True, **kwargs")
    elif fault == "wrong_kv_mode":
        source = source.replace("            _kv_only=True,", "            _kv_only=False,")
    elif fault == "wrong_native_op":
        source = source.replace("if _kv_only else torch.ops.rpu.gemma_forward",
                                "if _kv_only else torch.ops.rpu.adarms_forward")
    elif fault == "wrong_native_receiver":
        source = source.replace("if _kv_only else torch.ops.rpu.gemma_forward",
                                "if _kv_only else other.ops.rpu.gemma_forward")
    elif fault == "rebound_native_alias":
        source = source.replace("        if graph_enabled:",
                                "        native_forward = unrelated_prefill\n        if graph_enabled:")
    elif fault == "uncalled_native_alias":
        source = source.replace("output = native_forward(", "output = unrelated_prefill(")
    elif fault == "wrong_graph_callee":
        source = source.replace("output = native_forward(", "output = unrelated_prefill(", 1)
    elif fault == "escaped_native_alias":
        source = source.replace("        if graph_enabled:",
                                "        opaque(native_forward)\n        if graph_enabled:")
    elif fault == "rebound_mode":
        source = source.replace("        native_forward = (",
                                "        _kv_only = True\n        native_forward = (")
    elif fault == "wrong_reset_receiver":
        source = source.replace("torch.ops.rpu.spm_alloc_reset_temporary()",
                                "other.ops.rpu.spm_alloc_reset_temporary()")
    elif fault == "reset_before_native":
        source = source.replace("        torch.ops.rpu.spm_alloc_reset_temporary()", "        pass")
        source = source.replace("        native_forward = (",
                                "        torch.ops.rpu.spm_alloc_reset_temporary()\n        native_forward = (")
    elif fault == "reset_in_uncalled_function":
        source = source.replace("        torch.ops.rpu.spm_alloc_reset_temporary()",
                                "        def unused():\n            torch.ops.rpu.spm_alloc_reset_temporary()")
    elif fault == "reset_inside_capture":
        source = source.replace("        torch.ops.rpu.spm_alloc_reset_temporary()",
                                "        with self._rpu_gemma_graph_cache.capture(_sig):\n            torch.ops.rpu.spm_alloc_reset_temporary()")
    assert source != (ROOT / GEMMA).read_text(), fault
    target = tmp_path / GEMMA
    target.parent.mkdir(parents=True)
    target.write_text(source)
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    failures = []
    checker._check_file(GEMMA, *checker.TARGET_FILES[GEMMA], failures)
    assert failures, fault
