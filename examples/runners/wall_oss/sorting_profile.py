"""Summarize separate Wall sorting replay traces, including cycle-counter wrap."""

from collections import defaultdict
import json
from pathlib import Path
import re


def summarize_profile(output_dir):
    root = Path(output_dir)
    records = defaultdict(list)
    paths = sorted(
        (root / "hwperf").glob("*replay*.json"),
        key=lambda path: int(re.search(r"_seq(\d+)_", path.name)[1]),
    )
    for path in paths:
        component = next(
            name for name, tag in (("vision", "vision"), ("prefill", "llm_embeds"),
                                  ("denoise", "denoise")) if tag in path.name
        )
        nodes = [json.loads(line) for line in
                 (root / f"{component}_0_nodes.jsonl").read_text().splitlines()]
        names = [node["kernel"] for node in nodes if node["kind"] == "Kernel"]
        trace = json.loads(path.read_text())
        mhz = float(trace["otherData"]["frequency_hz"]) / 1e6
        events = [event for event in trace["traceEvents"]
                  if event.get("cat") in ("Compute", "DMA")]
        if mhz <= 0 or not events or len(events) % 2:
            raise ValueError(f"incomplete hardware trace: {path}")
        starts, durations, kernels = [], [], []
        for begin, end in zip(events[::2], events[1::2]):
            if begin["ph"] != "B" or end["ph"] != "E" or begin["name"] != end["name"]:
                raise ValueError(f"unpaired trace event: {path}")
            start, finish = round(begin["ts"] * mhz), round(end["ts"] * mhz)
            duration = int(end["args"]["duration_cycles"])
            if (finish - start) & 0xffffffff != duration:
                raise ValueError(f"trace timestamp/cycle mismatch: {path}")
            starts.append(start)
            durations.append(duration)
            if begin["cat"] == "Compute":
                if len(kernels) >= len(names):
                    raise ValueError(f"trace exceeds captured graph: {path}")
                kernels.append({"name": names[len(kernels)], "cycles": duration,
                                "us": duration / mhz})
        # This checks trace completeness, not any expected model kernel quantity.
        if len(kernels) != len(names):
            raise ValueError(f"trace does not cover its captured graph: {path}")
        period = 1 << 32
        points = sorted(set(starts))
        gap = max(range(len(points)),
                  key=lambda i: (points[(i + 1) % len(points)] - points[i]) % period)
        origin = points[(gap + 1) % len(points)]
        span = max((start - origin) % period + duration
                   for start, duration in zip(starts, durations))
        if span >= period // 2:
            raise ValueError(f"ambiguous graph counter interval: {path}")
        records[component].append({
            "trace": str(path), "span_ms": span / (mhz * 1000),
            "frequency_hz": mhz * 1e6,
            "counter_wrap": max(starts) - min(starts) > period // 2,
            "kernels": kernels,
        })
    if set(records) != {"vision", "prefill", "denoise"}:
        raise ValueError(f"missing component replay traces: {sorted(records)}")
    output = {}
    for component, traces in records.items():
        selected = traces[-1]  # Each named startup bundle packs all three images.
        grouped = defaultdict(list)
        for kernel in selected["kernels"]:
            grouped[kernel["name"]].append(kernel["us"])
        output[component] = {
            **selected,
            "kernel_count": len(selected["kernels"]),
            "groups": [
                {"name": name, "count": len(values), "sum_ms": sum(values) / 1000}
                for name, values in sorted(grouped.items(), key=lambda item: -sum(item[1]))
            ],
        }
    (root / "PROFILE.json").write_text(json.dumps(output, indent=2) + "\n")
    return output
