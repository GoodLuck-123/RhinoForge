"""Process-environment setup used by the package composition root.

This module intentionally imports only the standard library at module load so
the file-descriptor limit can be adjusted before PyTorch is imported.
"""
from __future__ import annotations

import resource
import sys
import warnings


_TARGET_NOFILE = 1_048_576


def ensure_nofile_limit() -> None:
    """Raise the soft fd limit when possible and report changes or failures."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft >= _TARGET_NOFILE:
        return
    new_soft = min(_TARGET_NOFILE, hard)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        print(
            f"[RhinoForge] RLIMIT_NOFILE soft raised {soft} -> {new_soft} "
            f"(hard={hard}) to avoid bad_alloc from fd exhaustion on large "
            "device models.",
            file=sys.stderr,
        )
    except (ValueError, OSError) as exc:
        warnings.warn(
            f"RhinoForge: could not raise RLIMIT_NOFILE soft from {soft} "
            f"to {new_soft}: {exc}. Large models may fail with std::bad_alloc "
            "on device alloc."
        )


def configure_hf_cache() -> None:
    """Apply the project model-cache policy after the PyTorch import boundary."""
    from . import model_registry

    model_registry.set_hf_home_env()


__all__ = ["configure_hf_cache", "ensure_nofile_limit"]
