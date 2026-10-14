"""Host-only synthetic launch-writer traces; never evidence of hardware timing."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[1]
PERFORMANCE = runpy.run_path(str(ROOT / "python/rpu_backend/runtime/performance.py"))
sample_ns = PERFORMANCE["_hwperf_device_sample_ns"]


def pair(index=0, *, category="Compute", stream=0, start=100, end=900, base=100, mhz=800):
    """Mirror r1_queue.cc's %.3f microseconds and asymmetric B/E args."""
    name = "same_kernel_name"
    args = {"batch_index": index, "stream": stream}
    duration_cycles = (end - start) % (1 << 32)
    duration_us = duration_cycles / mhz
    end_args = {"duration_cycles": duration_cycles, "duration_us": round(duration_us, 3)}
    if category == "DMA":
        name = f"dma_{index}_ch{stream // 2}"
        args.update(channel=stream // 2, size_bytes=128, direction="DDR->SPM")
        end_args["bandwidth_GBps"] = round(128 / 1000 / duration_us, 3)
    else:
        args["cores"] = [0, 1, 2, 3, 4, 5, 6, 7]
    common = {"name": name, "cat": category, "pid": 0, "tid": f"stream_{stream}"}
    return [
        {**common, "ph": "B", "ts": round(((start - base) % (1 << 32)) / mhz, 3), "args": args},
        {**common, "ph": "E", "ts": round(((end - base) % (1 << 32)) / mhz, 3), "args": end_args},
    ]


def trace(*pairs, barriers=0, base=100, mhz=800):
    pairs = pairs or (pair(base=base, start=base, end=base + 800, mhz=mhz),)
    return {
        "otherData": {
            "source": "rhino-launch-kernel HW perf trace", "frequency_hz": str(mhz * 1_000_000),
            "base_cycle": str(base), "batch_events": str(len(pairs) + barriers),
            "records_captured": str(2 * (len(pairs) + 1)),
            "record_validation": "work-completion-v1",
        },
        "traceEvents": [event for work in pairs for event in work],
    }


def test_device_segment_elapsed_includes_dma_overlap_and_gaps_not_kernel_sum():
    doc = trace(pair(end=1700), pair(1, category="DMA", stream=2, start=900, end=3300))
    assert sample_ns(doc) == 4000  # Compute 2us + DMA 3us overlap by 1us.
    doc = trace(pair(), pair(1, start=2500, end=3300))
    assert sample_ns(doc) == 4000  # Includes the 2us idle gap; same names are legal.


def test_dma_only_segment_is_valid_without_inventing_compute_work():
    assert sample_ns(trace(pair(category="DMA", stream=14))) == 1000


def test_legacy_completion_substitution_is_diagnostic_only(tmp_path):
    # The old writer's lkn_find_pair ignored unit_id. With a lost work pair,
    # completion START/END (raw unit=16; packet unit=0xff, cycles 81000..81008) get labeled as
    # the expected Compute. Neither these JSON bytes nor the work census expose
    # that substitution. Strict cost ingestion must reject the legacy writer.
    doc = trace(pair(start=81000, end=81008, base=81000), base=81000)
    del doc["otherData"]["record_validation"]
    path = tmp_path / "rpu_hwperf_legacy_completion.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    report = PERFORMANCE["_hwperf_aggregate_file"](str(path))
    assert report["error"] is None
    assert report["latency_ms"] == 0.00001  # Diagnostic, not certification.
    with pytest.raises(ValueError, match="record_validation"):
        sample_ns(doc)


def test_zero_base_cycle_is_valid_and_other_metadata_stays_positive():
    assert sample_ns(trace(base=0)) == 1000
    for field in ("frequency_hz", "batch_events", "records_captured"):
        doc = trace(base=0)
        doc["otherData"][field] = "0"
        with pytest.raises(ValueError):
            sample_ns(doc)


def test_short_capture_across_counter_wrap_preserves_chronological_origin():
    # The raw minimum is 0, but the first work starts before the uint32 wrap.
    # The validated writer normalizes against completion, not the raw minimum.
    base = (1 << 32) - 800
    doc = trace(pair(start=base, end=0, base=base),
                pair(1, category="DMA", stream=2, start=0, end=800, base=base), base=base)
    assert [event["ts"] for event in doc["traceEvents"]] == [0., 1., 1., 2.]
    assert sample_ns(doc) == 2000


@pytest.mark.parametrize("value", [None, True, 1, "", "work-completion-v0",
                                  "work-completion-v1 ", ["work-completion-v1"]])
def test_cost_ingestion_requires_exact_writer_validation_contract(value):
    doc = trace()
    doc["otherData"]["record_validation"] = value
    with pytest.raises(ValueError, match="record_validation"):
        sample_ns(doc)


def test_rounding_matches_writer_to_three_microsecond_decimals():
    doc = trace(pair(end=101), pair(1, start=101, end=103))
    assert sample_ns(doc) == 4
    assert sample_ns(trace(mhz=1000)) == 800


def test_flow_pairs_follow_all_work_and_bind_actual_endpoints():
    doc = trace(pair(end=1700), pair(1, category="DMA", stream=2, start=900, end=2500),
                pair(2, start=2500, end=3300), barriers=1)
    doc["traceEvents"] += [
        {"name": "barrier_0", "cat": "SYNC", "ph": "s", "pid": 0, "tid": "stream_2", "ts": 3., "id": "0"},
        {"name": "barrier_0", "cat": "SYNC", "ph": "f", "pid": 0, "tid": "stream_0", "ts": 3., "id": "0", "bp": "e"},
    ]
    assert sample_ns(doc) == 4000
    for field, value in (("id", "1"), ("ts", 2.999), ("name", "barrier_1"), ("bp", "s")):
        changed = copy.deepcopy(doc)
        changed["traceEvents"][-1][field] = value
        with pytest.raises(ValueError, match="barrier flow"):
            sample_ns(changed)
    doc["otherData"]["batch_events"] = "3"
    with pytest.raises(ValueError, match="metadata disagrees"):
        sample_ns(doc)


def test_unexported_barriers_are_not_falsely_treated_as_missing_work():
    # The writer omits flows without both neighboring work endpoints. JSON
    # alone is not proof of the expected Graph work counts or invocation scope.
    assert sample_ns(trace(barriers=3)) == 1000


@pytest.mark.parametrize("field", ["source", "frequency_hz", "base_cycle", "batch_events", "records_captured",
                                  "record_validation"])
def test_metadata_is_mandatory(field):
    doc = trace()
    del doc["otherData"][field]
    with pytest.raises(ValueError):
        sample_ns(doc)


@pytest.mark.parametrize("value", [None, True, 4, 4., "", "-1", "+4", "04", " 4", "4.0", "nan", "1" * 100])
@pytest.mark.parametrize("field", ["frequency_hz", "base_cycle", "batch_events", "records_captured"])
def test_metadata_uses_canonical_writer_decimal_strings(field, value):
    doc = trace()
    doc["otherData"][field] = value
    with pytest.raises(ValueError):
        sample_ns(doc)


@pytest.mark.parametrize("doc", [None, [], {}, {"otherData": None}, {"otherData": []}])
def test_invalid_document_is_a_value_error(doc):
    with pytest.raises(ValueError):
        sample_ns(doc)


@pytest.mark.parametrize("change", ["empty", "not_list", "not_object", "orphan_begin", "orphan_end", "missing_pair",
                                   "duplicate_pair", "full_buffer", "too_many_records", "too_few_events"])
def test_missing_duplicate_or_clipped_work_cannot_supply_a_cost(change):
    doc = trace(pair(), pair(1, start=900, end=1700))
    events = doc["traceEvents"]
    if change == "empty":
        events.clear()
    elif change == "not_list":
        doc["traceEvents"] = {}
    elif change == "not_object":
        events[0] = None
    elif change == "orphan_begin":
        events.pop()
    elif change == "orphan_end":
        events.pop(0)
    elif change == "missing_pair":
        del events[2:]
    elif change == "duplicate_pair":
        events[2:] = copy.deepcopy(events[:2])
    elif change == "full_buffer":
        doc["otherData"]["records_captured"] = "4"
    elif change == "too_many_records":
        doc["otherData"]["records_captured"] = "8"
    elif change == "too_few_events":
        doc["otherData"]["batch_events"] = "1"
    with pytest.raises(ValueError):
        sample_ns(doc)


@pytest.mark.parametrize("field,value", [
    ("name", "other"), ("cat", "DMA"), ("ph", "B"), ("ph", "X"),
    ("pid", 1), ("pid", False), ("tid", "stream_1"), ("tid", "stream_16"),
    ("args", []), ("args", None),
])
def test_compute_end_must_pair_the_actual_begin(field, value):
    doc = trace()
    doc["traceEvents"][1][field] = value
    with pytest.raises(ValueError):
        sample_ns(doc)


@pytest.mark.parametrize("value", [None, True, "1", -1, float("nan"), float("inf"), float("-inf"), 10**1000])
@pytest.mark.parametrize("where", ["begin", "end", "duration"])
def test_nonfinite_or_non_numeric_times_fail_closed(where, value):
    doc = trace()
    if where == "duration":
        doc["traceEvents"][1]["args"]["duration_us"] = value
    else:
        doc["traceEvents"][where == "end"]["ts"] = value
    with pytest.raises(ValueError):
        sample_ns(doc)


@pytest.mark.parametrize("field,value", [
    ("duration_cycles", None), ("duration_cycles", False), ("duration_cycles", 0),
    ("duration_cycles", -1), ("duration_cycles", 1 << 32), ("duration_cycles", 2),
    ("duration_us", 0), ("duration_us", 2), ("duration_us", 1.0006),
])
@pytest.mark.parametrize("category", ["Compute", "DMA"])
def test_duration_must_agree_with_elapsed_and_frequency(category, field, value):
    doc = trace(pair(category=category))
    doc["traceEvents"][1]["args"][field] = value
    with pytest.raises(ValueError, match="duration"):
        sample_ns(doc)


@pytest.mark.parametrize("change", ["negative_elapsed", "zero_elapsed", "inflated_elapsed", "wrong_frequency", "nonzero_origin", "oversized_base"])
def test_elapsed_and_clock_metadata_cannot_disagree(change):
    doc = trace()
    if change == "negative_elapsed":
        doc["traceEvents"][0]["ts"] = 2
    elif change == "zero_elapsed":
        doc["traceEvents"][1]["ts"] = 0
    elif change == "inflated_elapsed":
        doc["traceEvents"][1]["ts"] = 2
    elif change == "wrong_frequency":
        doc["otherData"]["frequency_hz"] = "1000000000"
    elif change == "nonzero_origin":
        for event in doc["traceEvents"]:
            event["ts"] += 1
    elif change == "oversized_base":
        doc["otherData"]["base_cycle"] = str(1 << 32)
    with pytest.raises(ValueError):
        sample_ns(doc)


@pytest.mark.parametrize("field,value", [
    ("batch_index", 1), ("batch_index", False), ("stream", False), ("stream", 2),
    ("cores", None), ("cores", [8]), ("cores", [True]), ("cores", [0, 0]),
])
def test_compute_input_metadata_must_match_writer(field, value):
    doc = trace()
    doc["traceEvents"][0]["args"][field] = value
    with pytest.raises(ValueError):
        sample_ns(doc)


@pytest.mark.parametrize("field,value", [
    ("channel", False), ("channel", 8), ("channel", 1), ("size_bytes", 0),
    ("size_bytes", -16), ("size_bytes", 17), ("size_bytes", 1 << 32), ("direction", "unknown"),
])
def test_dma_input_metadata_must_match_writer(field, value):
    doc = trace(pair(category="DMA"))
    doc["traceEvents"][0]["args"][field] = value
    with pytest.raises(ValueError, match="DMA"):
        sample_ns(doc)


def test_ordinary_report_stays_permissive_for_legacy_diagnostic_traces(tmp_path):
    doc = {"traceEvents": [
        {"ph": "B", "ts": 10},
        {"ph": "E", "ts": 30, "cat": "Compute", "name": "linear_1", "args": {"duration_us": 20}},
        {"ph": "E", "ts": 60, "cat": "Compute", "name": "linear_2", "args": {"duration_us": 30}},
        {"ph": "E", "ts": 70, "cat": "DMA", "name": "copy_3", "args": {"duration_us": 10}},
    ]}
    path = tmp_path / "rpu_hwperf_legacy.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    report = PERFORMANCE["_hwperf_aggregate_file"](str(path))
    assert report["latency_ms"] == 0.06
    assert report["rows"] == [{"name": "linear", "count": 2, "total_us": 50., "avg_us": 25., "pct": 100.}]
    with pytest.raises(ValueError, match="otherData"):
        sample_ns(doc)
