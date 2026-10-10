"""Small, stdlib-only result metadata shared by the existing example runners.

The original report fields remain authoritative. This module does not run or
infer numerical, Graph or task checks from an exit code or an old status label.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import statistics
import sys


def latency_timing(values, boundary, *, metric="total_latency", statistic="median"):
    """Describe existing host measurements without changing their timer."""
    values = list(values)
    if not values:
        return []
    summarize = {"mean": statistics.mean, "median": statistics.median}[statistic]
    return [{"metric": metric, "unit": "ms", "boundary": boundary,
             "statistic": statistic, "value": summarize(values),
             "samples": values, "sample_count": len(values)}]


def phase_timings(rows, boundary, *, statistic="median", phase_boundary=None, metric_prefix=""):
    rows = list(rows)
    if not rows:
        return []
    return [item for key in rows[0] for item in latency_timing(
        (row[key] for row in rows), boundary if key == "total_ms" else phase_boundary or boundary,
        metric=metric_prefix + key.removesuffix("_ms") + "_latency", statistic=statistic)]


def add_result_fields(report, output_dir, *, completed, timing=(),
                      runtime_checks=(), reference_keys=(), output_names=()):
    """Add discoverable output/timing fields; measured errors are not a pass."""
    report["result_schema_version"] = 1
    report["execution"] = {"status": "COMPLETED" if completed else "FAILED"}
    for key in ("error", "cleanup_error"):
        if report.get(key):
            report["execution"][key] = report[key]
    # Wall base already records library artifacts here. Keep those entries.
    artifacts = report.setdefault("artifacts", {})
    directory = Path(output_dir).expanduser().resolve() if output_dir is not None else None
    artifacts["directory"] = str(directory) if directory is not None else None
    artifacts["report"] = str(directory / "report.json") if directory is not None else None
    # Reusable output directories may contain stale files from an earlier run.
    artifacts["outputs"] = [str(directory / name) for name in output_names
        if completed and directory is not None and (directory / name).is_file()]
    hwperf = report.get("hwperf", [])
    if completed and directory is not None:
        for item in hwperf:
            if isinstance(item, dict) and item.get("file"):
                path = Path(item["file"]).expanduser().resolve()
                if path.is_file() and path.is_relative_to(directory):
                    artifacts["outputs"].append(str(path))
    requested_hwperf = bool(hwperf or report.get("hwperf_enabled") or report.get("hwperf_replays"))
    artifacts["hwperf_directory"] = (
        str(directory / "hwperf") if completed and directory is not None
        and requested_hwperf and (directory / "hwperf").is_dir() else None)
    compared = [key for key in reference_keys
                if isinstance(report.get(key), dict) and report[key].get("metrics")]
    report["validation"] = {
        "runtime": {"status": "CHECKED" if completed and runtime_checks else
                               "NOT_COMPLETED" if not completed else "NOT_EVALUATED",
                    "checks": list(runtime_checks) if completed else []},
        "reference": {"status": "MEASURED" if compared else "NOT_RUN",
                      "report_fields": compared},
        "task": {"status": "NOT_EVALUATED"},
    }
    report["timing"] = list(timing)
    return report


def close_with_report(owner, report):
    """Retire once, recording cleanup failure without replacing an active error."""
    if owner is None:
        return
    primary = sys.exc_info()[1]
    try:
        owner.close()
    except BaseException as error:
        report["cleanup_error"] = f"{type(error).__name__}: {error}"
        if primary is None:
            raise
        if hasattr(primary, "add_note"):
            primary.add_note(f"model cleanup failed: {error!r}")


def _emit_performance(report):
    if report.get("execution", {}).get("status") == "COMPLETED" and report.get("timing"):
        from examples.common.logging import emit_performance
        emit_performance(report.get("config"), report)


def write_report(report, output_dir, **json_options):
    """A recording failure must not replace the active execution exception."""
    primary = sys.exc_info()[1]
    try:
        (Path(output_dir) / "report.json").write_text(
            json.dumps(report, indent=2, **json_options) + "\n", encoding="utf-8")
    except BaseException as error:
        if primary is None:
            raise
        if hasattr(primary, "add_note"):
            primary.add_note(f"result recording failed: {error!r}")
    else:
        _emit_performance(report)


@contextmanager
def record_result(report, output_dir, *, timing, runtime_checks=(), reference_keys=(), output_names=()):
    """Persist success and failure from the existing execution block."""
    primary = None
    try:
        yield
    except BaseException as error:
        primary = error
        report.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            add_result_fields(report, output_dir, completed=primary is None,
                              timing=timing(), runtime_checks=runtime_checks,
                              reference_keys=reference_keys, output_names=output_names)
            if primary is None:
                report.setdefault("status", "completed")
            if output_dir is not None:
                write_report(report, output_dir)
            else:
                _emit_performance(report)
        except BaseException as error:
            if primary is None:
                raise
            if hasattr(primary, "add_note"):
                primary.add_note(f"result recording failed: {error!r}")
