"""Shared CPU execution settings; validation never imports torch/backend."""
from __future__ import annotations

import os


def positive_threads(value, label="[cpu].threads"):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def resolve_cpu_threads(config, *, default=None):
    """Resolve the public setting and legacy aliases without mutating config."""
    cpu = config.get("cpu", {})
    if not isinstance(cpu, dict) or set(cpu) - {"threads"}:
        raise ValueError("[cpu] only accepts threads and must be a table")
    run = config.get("run", {})
    if not isinstance(run, dict):
        raise ValueError("[run] must be a table")
    supplied = []
    for table, key, label in (
        (cpu, "threads", "[cpu].threads"),
        (run, "torch_num_threads", "[run].torch_num_threads"),
        (run, "cpu_threads", "[run].cpu_threads"),
    ):
        if key in table:
            supplied.append((label, positive_threads(table[key], label)))
    if len({value for _, value in supplied}) > 1:
        fields = ", ".join(f"{label}={value}" for label, value in supplied)
        raise ValueError(f"conflicting CPU thread settings: {fields}")
    if supplied:
        return supplied[0][1]
    return None if default is None else positive_threads(default)


def configure_cpu_threads(threads):
    """Apply before model/backend imports and return the effective CPU settings.

    None leaves inherited/default settings intact. This controls intra-op CPU
    threads, not inter-op threads, affinity, tokenizer workers or device cores.
    """
    if threads is not None:
        positive_threads(threads)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[name] = str(threads)
    import torch
    if threads is not None:
        torch.set_num_threads(threads)
    return {
        "requested_threads": threads,
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
    }
