"""Host-only performance summaries preserve the measured report semantics."""
import copy
import json
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def common(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from examples.common import performance, results
    return performance, results


def timing(metric, value, *, statistic="mean", samples=None, unit="ms", boundary="actual host call"):
    return {"metric": metric, "value": value, "statistic": statistic, "unit": unit,
            "samples": samples or [], "sample_count": len(samples or []), "boundary": boundary}


def report(*timings, **extra):
    return {"execution": {"status": "COMPLETED"}, "timing": list(timings), **extra}


def test_vla_e2e_median_uses_formal_samples_without_mutating_mean_or_boundaries(common):
    formatter, _ = common
    config = {"pi05": {"precision": "w8a16"}, "run": {"warmup": 2, "runs": 3}}
    value = report(timing("total_latency", 40, samples=[10, 10, 100],
                          boundary="predict_action_chunk including CPU output copy"),
                   timing("vision_latency", 2, samples=[1, 2, 3]),
                   model="pi05", mean_ms={"total_ms": 40})
    original = copy.deepcopy((config, value))
    lines = formatter.summary(config, value)
    assert "VLA E2E: median=10.000 ms; n=3" in lines[0]
    assert "load/warmup excluded" in lines[0]
    assert "including CPU output copy" in lines[0]
    assert "vision component: mean=2.000 ms" in lines[1]
    assert (config, value) == original


def test_diagnostic_components_are_not_added_to_e2e(common):
    formatter, _ = common
    value = report(timing("total_latency", 10, statistic="median", samples=[10]),
                   timing("diagnostic_vision_latency", 80, samples=[80],
                          boundary="separate instrumented replay"), model="g05")
    lines = formatter.summary({"g05": {}}, value)
    assert "median=10.000 ms" in lines[0]
    assert "diagnostic vision component: mean=80.000 ms" in lines[1]
    assert "separate instrumented replay" in lines[1]


@pytest.mark.parametrize("target,component,label", [
    ("halo", None, "Action expert component (single step)"),
    ("dinov3", None, "Vision component"),
    ("qwen3_vl", "vision", "Vision component"),
])
def test_component_examples_are_never_called_vla_e2e(common, target, component, label):
    formatter, _ = common
    config = {"example": {"target": target}, "input": {"component": component}}
    lines = formatter.summary(config, report(timing("total_latency", 5, samples=[5])))
    assert lines[0].startswith(label + ":")
    assert "E2E" not in "\n".join(lines)


def test_early_eos_uses_actual_counts_and_original_aggregate_rate(common):
    formatter, _ = common
    config = {"example": {"target": "llama"}, "input": {"decode_steps": 999}}
    rows = [{"prefill_tokens": 7, "decode_calls": count, "output_tokens": count + 1}
            for count in (1, 3)]
    value = report(timing("prefill_throughput", 700, unit="tokens/s",
                          statistic="aggregate_tokens_over_total_time"),
                   timing("decode_throughput", 123.25, unit="tokens/s",
                          statistic="aggregate_tokens_over_total_time"),
                   generation_runs=rows)
    lines = "\n".join(formatter.summary(config, value))
    assert "prompt_tokens=7; decode_calls=1..3 (varies); output_tokens=2..4 (varies)" in lines
    assert "123.250 tok/s (aggregate_tokens_over_total_time)" in lines
    assert "999" not in lines


def test_median_vlm_rates_are_not_recomputed_and_no_vision_time_is_invented(common):
    formatter, _ = common
    config = {"example": {"target": "qwen3_5_vision"}, "run": {"summary_statistic": "median"}}
    rows = [{"prefill_tokens": 80, "decode_calls": 4, "output_tokens": 5}]
    value = report(generation_runs=rows, generation_summary={
        "summary_statistic": "median", "prefill_tokens_per_second": 789.25,
        "decode_tokens_per_second": 12.5, "prefill_ms_median": 100,
        "decode_ms_median": 300})
    lines = "\n".join(formatter.summary(config, value))
    assert "VLM prefill: 789.250 tok/s (median_per_run_rate); median=100.000 ms" in lines
    assert "VLM decode: 12.500 tok/s (median_per_run_rate); median=300.000 ms" in lines
    assert "Vision not separately timed" in lines
    assert "vision component" not in lines


def test_zero_decode_calls_and_missing_values_do_not_fabricate_throughput(common):
    formatter, _ = common
    value = report(generation_runs=[{"prefill_tokens": 3, "decode_calls": 0, "output_tokens": 1}],
                   generation_summary={"decode_tokens_per_second": None})
    lines = "\n".join(formatter.summary({"example": {"target": "gemma4"}}, value))
    assert "LLM decode: tok/s=n/a (no decode calls)" in lines
    assert "LLM prefill:" not in lines
    assert "0.000 tok/s" not in lines


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1, True])
def test_unavailable_or_invalid_measurements_are_not_presented_as_performance(common, value):
    formatter, _ = common
    assert formatter.summary({}, report(timing("total_latency", value))) == []


def test_failed_and_running_reports_do_not_claim_measured_success(common):
    formatter, _ = common
    value = report(timing("total_latency", 5, samples=[5]))
    value["execution"]["status"] = "FAILED"
    assert formatter.summary({"pi05": {}}, value) == []
    assert formatter.summary({}, {"status": "RUNNING"}) == []


def test_results_hook_gets_final_normalized_fields_without_changing_saved_json(common, monkeypatch, tmp_path):
    _, results = common
    sink = ModuleType("examples.common.logging")
    sink.emit_performance = Mock()
    monkeypatch.setitem(sys.modules, "examples.common.logging", sink)
    config = {"example": {"target": "gr00t"}}
    value = {"config": config}
    measurements = [timing("total_latency", 5, samples=[5])]
    results.add_result_fields(value, tmp_path, completed=True, timing=measurements)
    sink.emit_performance.assert_not_called()
    assert value["execution"]["status"] == "COMPLETED"
    assert value["timing"] == measurements
    observed = []
    sink.emit_performance.side_effect = lambda *_: observed.append(
        json.loads((tmp_path / "report.json").read_text()))
    results.write_report(value, tmp_path)
    assert observed == [value]
    sink.emit_performance.assert_called_once_with(config, value)


def test_results_hook_skips_failures_and_input_preparation_only(common, monkeypatch, tmp_path):
    _, results = common
    sink = ModuleType("examples.common.logging")
    sink.emit_performance = Mock()
    monkeypatch.setitem(sys.modules, "examples.common.logging", sink)
    prepared = results.add_result_fields({}, tmp_path, completed=True)
    results.write_report(prepared, tmp_path)
    failed = results.add_result_fields({}, tmp_path, completed=False,
                                      timing=[timing("total_latency", 5, samples=[5])])
    results.write_report(failed, tmp_path)
    sink.emit_performance.assert_not_called()


def test_report_write_failure_never_emits_performance(common, monkeypatch, tmp_path):
    _, results = common
    sink = ModuleType("examples.common.logging")
    sink.emit_performance = Mock()
    monkeypatch.setitem(sys.modules, "examples.common.logging", sink)
    value = results.add_result_fields({}, tmp_path, completed=True,
                                      timing=[timing("total_latency", 5, samples=[5])])
    monkeypatch.setattr(Path, "write_text", Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        results.write_report(value, tmp_path)
    sink.emit_performance.assert_not_called()


def test_record_result_without_output_directory_emits_only_after_completion(common, monkeypatch):
    _, results = common
    sink = ModuleType("examples.common.logging")
    sink.emit_performance = Mock()
    monkeypatch.setitem(sys.modules, "examples.common.logging", sink)
    value = {}
    with results.record_result(value, None, timing=lambda: [timing("total_latency", 5, samples=[5])]):
        sink.emit_performance.assert_not_called()
    sink.emit_performance.assert_called_once_with(None, value)
    assert value["execution"]["status"] == "COMPLETED"
    assert value["artifacts"]["report"] is None


def test_record_result_without_output_directory_skips_failures_and_empty_timings(common, monkeypatch):
    _, results = common
    sink = ModuleType("examples.common.logging")
    sink.emit_performance = Mock()
    monkeypatch.setitem(sys.modules, "examples.common.logging", sink)
    with results.record_result({}, None, timing=lambda: []):
        pass
    with pytest.raises(ValueError, match="inference failed"):
        with results.record_result({}, None, timing=lambda: [timing("total_latency", 5, samples=[5])]):
            raise ValueError("inference failed")
    sink.emit_performance.assert_not_called()
