"""Scoped `sys.modules` stubbing for the host-only tests in this directory.

Several tests here exec ONE adapter source file on a plain CPU python by putting
stub modules in `sys.modules` first (importing the real `rpu_backend` would pull
the device extension). `unittest.mock.patch.dict` looks like the right tool for
that and is not: its `__exit__` CLEARS `sys.modules` and restores the snapshot
taken at `__enter__`, so every module first imported INSIDE the scope is
evicted — for these files that means `transformers` and, through it,
`torchvision` and `safetensors`.

Re-importing an evicted extension in the same process then hard-fails, because
the C-level side effects of the first import are still there:

  * `torchvision/_meta_registrations.py` → ``RuntimeError: This is not allowed
    since there's already a kernel registered from python overriding roi_align's
    behavior for Meta dispatch key and torchvision namespace.``
  * `safetensors` → ``ImportError: PyO3 modules compiled for CPython 3.8 or
    older may only be initialized once per interpreter process``

So one stub load silently poisons every later test in the same process that
loads an adapter source. That is an ordering bug, not a flake: on commit
4311852d `pytest -q tests/` was green (alphabetical order happens to import the
heavy chain before the first stub scope) while the `host_contracts` argument
list was red on the same tree.

`stub_modules` installs exactly the given keys and removes exactly those keys
again, leaving whatever was imported inside the scope in place.
"""
from __future__ import annotations

import contextlib
import importlib.util
import pathlib
import sys
import types

_ABSENT = object()
_REPO = pathlib.Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def stub_modules(stubs):
    """Install `stubs` in `sys.modules` for the duration of the block."""
    saved = {name: sys.modules.get(name, _ABSENT) for name in stubs}
    sys.modules.update(stubs)
    try:
        yield
    finally:
        for name, previous in saved.items():
            if previous is _ABSENT:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def real_rpu_env_bool():
    """`rpu_backend.runtime.rpu_env_bool`, exec'd out of control.py by path.

    Whatever stands in for `rpu_backend.runtime` MUST carry this function rather
    than a `MagicMock` or `None`: every adapter switch goes through it, a Mock is
    truthy for all of them, and the adapters then silently run a configuration no
    test asked for (measured on this tree: `_prefill_chunk_size_cap(8)` returns
    384 instead of 256). Nor may a caller hand-roll a copy — this repo has
    exactly ONE boolean env parser by policy (docs/runtime_config.md convention
    D). control.py's only package-level import is `rpu_backend._native_loader`
    (stdlib-only, and unused by the parser), so stubbing that one name is enough
    to exec it off the device.
    """
    loader = types.ModuleType("rpu_backend._native_loader")
    loader.device_nodes_accessible = None
    spec = importlib.util.spec_from_file_location(
        "rpu_runtime_control_cpu", _REPO / "python/rpu_backend/runtime/control.py"
    )
    control = importlib.util.module_from_spec(spec)
    with stub_modules(
        {"rpu_backend": types.ModuleType("rpu_backend"), loader.__name__: loader}
    ):
        spec.loader.exec_module(control)
    return control.rpu_env_bool
