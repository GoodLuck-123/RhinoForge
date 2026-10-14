"""Canonical error identity and actual planner loading without native bootstrap."""
import builtins
import importlib.util
from pathlib import Path
import pickle
import sys
from types import ModuleType

import pytest

from _module_stubs import stub_modules


ROOT = Path(__file__).resolve().parents[1] / "python/rpu_backend"


def _load(name, path, modules):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    modules.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_normal_package_error_reexports_keep_the_same_base_without_loading_device(monkeypatch):
    parent = ModuleType("rpu_backend")
    parent.__path__ = [str(ROOT)]
    api = ModuleType("rpu_backend.api")
    api.__path__ = [str(ROOT / "api")]
    blocked = {"rpu_backend": parent, "rpu_backend.api": api}
    for leaf in ("control", "debug", "decoder", "hw_attrs", "registry", "weights"):
        blocked["rpu_backend.runtime." + leaf] = ModuleType("rpu_backend.runtime." + leaf)
    blocked["rpu_backend.runtime.control"].rpu_env_bool = lambda *args: False
    original = builtins.__import__

    def no_torch(name, *args, **kwargs):
        assert name != "torch" and not name.startswith("torch."), name
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    # These two real-source modules retain the fake leaves they import. Restore
    # their registrations too, without evicting unrelated imports/extensions.
    with stub_modules(blocked), monkeypatch.context() as modules:
        runtime = _load("rpu_backend.runtime", ROOT / "runtime/__init__.py", modules)
        public = _load("rpu_backend.api.errors", ROOT / "api/errors.py", modules)
        assert public.RPUBackendError is runtime.RPUBackendError
        assert public.PlannerRejectError is runtime.PlannerRejectError
        assert issubclass(public.PlannerRejectError, (runtime.RPUBackendError, RuntimeError))
        assert isinstance(public.PlannerRejectError("CAPABILITY"), runtime.RPUBackendError)
        assert type(pickle.loads(pickle.dumps(runtime.RPUBackendError("test")))) is runtime.RPUBackendError


@pytest.mark.parametrize("preexisting", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_error_reexport_probe_restores_loaded_parents(monkeypatch, preexisting, failure):
    """Running this probe before real adapter tests must not retain fake leaves."""
    names = ("rpu_backend.runtime", "rpu_backend.api.errors")
    previous = {name: ModuleType(name) for name in names}
    for name in names:
        if preexisting:
            monkeypatch.setitem(sys.modules, name, previous[name])
        else:
            monkeypatch.delitem(sys.modules, name, raising=False)

    def interrupted(_value):
        raise RuntimeError("probe interrupted after loading both parents")

    if failure:
        monkeypatch.setattr(pickle, "dumps", interrupted)
    with pytest.MonkeyPatch.context() as probe:
        if failure:
            with pytest.raises(RuntimeError, match="probe interrupted"):
                test_normal_package_error_reexports_keep_the_same_base_without_loading_device(probe)
        else:
            test_normal_package_error_reexports_keep_the_same_base_without_loading_device(probe)
    for name in names:
        if preexisting:
            assert sys.modules[name] is previous[name]
        else:
            assert name not in sys.modules
