from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tomllib
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_host_only_import_reports_missing_native_backend() -> None:
    site_packages = [path for path in sys.path if "site-packages" in path]
    script = (
        "import sys; "
        f"sys.path.insert(0, {str(ROOT / 'python')!r}); "
        f"sys.path.extend({site_packages!r}); "
        "import rpu_backend, torch; "
        "assert rpu_backend.__version__; "
        "assert hasattr(torch, 'rpu')"
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("RPU_BACKEND_SO", None)
    result = subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "[RhinoForge]" in result.stdout
    assert "native backend not loaded" in result.stdout


def test_nofile_preflight_reports_best_effort_changes(monkeypatch, capsys) -> None:
    from rpu_backend import _bootstrap_env

    calls = []
    monkeypatch.setattr(
        _bootstrap_env.resource,
        "getrlimit",
        lambda _kind: (1024, 4096),
    )
    monkeypatch.setattr(
        _bootstrap_env.resource,
        "setrlimit",
        lambda kind, value: calls.append((kind, value)),
    )
    _bootstrap_env.ensure_nofile_limit()
    assert calls == [
        (_bootstrap_env.resource.RLIMIT_NOFILE, (4096, 4096))
    ]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "RLIMIT_NOFILE soft raised 1024 -> 4096" in captured.err

    def deny_limit_change(*_args):
        raise OSError("denied")

    monkeypatch.setattr(
        _bootstrap_env.resource,
        "setrlimit",
        deny_limit_change,
    )
    with pytest.warns(UserWarning, match="could not raise RLIMIT_NOFILE"):
        _bootstrap_env.ensure_nofile_limit()
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_public_api_and_host_graph_lifecycle() -> None:
    import rpu_backend
    from rpu_backend.graph import GraphCache, GraphSignature

    assert rpu_backend.__all__ == [
        "__version__",
        "RPUCache",
        "RPUModelForCausalLM",
        "reset_graph_cache",
    ]

    cache = GraphCache()
    signature = GraphSignature(op_id="host-contract", shapes=(1, 16))
    assert cache.phase == "CONFIGURING"
    assert cache.size() == 0
    assert cache.cache_invariant_ok()
    cache.begin_warmup()
    with cache.capture(signature):
        pass
    assert cache.phase == "WARMING"
    with pytest.raises(RuntimeError, match=r"requires the C\+\+ backend"):
        cache.freeze()


def test_pi05_remap_cache_tracks_source_modification_time(
    monkeypatch, tmp_path,
) -> None:
    from rpu_backend.adapters.pi05 import loader

    source = tmp_path / "model.safetensors"
    cache = tmp_path / "model_remapped.safetensors"
    source.write_bytes(b"make")
    cache.write_bytes(b"remapped-make")
    stats = {
        source: SimpleNamespace(st_mtime_ns=10, st_ctime_ns=10),
        cache: SimpleNamespace(st_mtime_ns=20, st_ctime_ns=20),
    }
    monkeypatch.setattr(
        loader,
        "os",
        SimpleNamespace(
            path=SimpleNamespace(exists=os.path.exists),
            stat=lambda path: stats[Path(path)],
        ),
    )
    assert not loader._cache_is_stale([source], cache)

    source.write_bytes(b"pour")
    stats[source] = SimpleNamespace(st_mtime_ns=9, st_ctime_ns=30)
    assert source.read_bytes() == b"pour"
    assert not loader._cache_is_stale([source], cache)
    stats[source] = SimpleNamespace(st_mtime_ns=21, st_ctime_ns=30)
    assert loader._cache_is_stale([source], cache)


def test_graph_structure_diagnostics_are_safe_and_public() -> None:
    from rpu_backend.graph import Graph

    graph = Graph()

    class FakeGraph:
        @staticmethod
        def dump_replay_plan() -> str:
            return "GraphReplayPlan segments=1"

        @staticmethod
        def dump_tree() -> str:
            return "GraphTree segments=1"

    graph._impl = FakeGraph()
    assert graph.dump_replay_plan() == "GraphReplayPlan segments=1"
    assert graph.dump_tree() == "GraphTree segments=1"

    header = (ROOT / "src/graph/graph_runtime.h").read_text(encoding="utf-8")
    bindings = (ROOT / "src/graph/graph_pybind.cpp").read_text(encoding="utf-8")
    diagnostics = (ROOT / "src/graph/graph_runtime_dump.cpp").read_text(encoding="utf-8")

    for name in ("dump_replay_plan", "dump_tree"):
        assert f"std::string {name}() const;" in header
        assert f'.def("{name}", &RpuKernelGraph::{name}' in bindings
    for forbidden in (
        "get_kernel_" + "args_instr",
        "instr_" + "data",
        "args_" + "data",
        ".src_" + "addr",
        ".dst_" + "addr",
        ".live_" + "base",
        ".src_dev_" + "addr",
        ".dst_dev_" + "addr",
        "reinterpret_cast",
    ):
        assert forbidden not in diagnostics
    assert "boundary_flush_count=" in diagnostics


def test_external_adapter_registration() -> None:
    from rpu_backend.runtime import registry

    class PublicTestAdapter:
        pass

    architecture = "RhinoForgePublicTestArchitecture"
    try:
        registry.register_adapter(architecture, PublicTestAdapter)
        assert registry.get_adapter(architecture) is PublicTestAdapter
        assert architecture in registry.list_adapters()
        registry.register_adapter(architecture, PublicTestAdapter)
        with pytest.raises(RuntimeError, match="already registered"):
            registry.register_adapter(architecture, type("OtherAdapter", (), {}))
    finally:
        registry.ADAPTERS.pop(architecture, None)


def test_registry_keeps_public_example_paths_local() -> None:
    from rpu_backend import model_registry

    required = {
        "qwen3-0.6b",
        "llama-3.2-1b",
        "qwen3-vl-2b",
        "qwen3-vl-4b",
        "qwen3-vl-8b",
        "qwen3_5-0.8b",
        "dinov3-vit-b",
        "pi05-libero-finetuned",
        "qwen3-14b-w8a16-lmhead-int8",
        # Public component and quantization-recipe inputs. Registry presence
        # is not model admission; loaders still enforce their exact profiles.
        "lingbot-vla-4b",
        "qwen3-14b",
        "qwen3-0.6b-w8a16-lmhead-int8",
        "qwen3-1.7b-w8a16-lmhead-int8",
        "qwen3-4b-w8a16-lmhead-int8",
        "qwen3-0.6b-instruct",
        "qwen3-4b-instruct",
    }
    # Asset lookup also serves diagnostic/source-only profiles. Admission
    # remains the loader's responsibility.
    required.update({"dinov3-vit-s", "dinov3-vit-l", "paligemma2-3b-pt-224",
                     "paligemma2-3b-mix-224", "qwen3_5-2b-w4-mlp-w8-attn-gdn-g32"})
    excluded = {"gemma2", "paligemma2"}
    assert required <= model_registry.MODELS.keys()
    assert excluded.isdisjoint(model_registry.MODELS)
    for relative in model_registry.MODELS.values():
        path = Path(relative)
        assert not path.is_absolute()
        assert ".." not in path.parts


def test_qwen3_vl_deepstack_targets_first_text_layers() -> None:
    from types import SimpleNamespace

    from rpu_backend.adapters.qwen3_vl import (
        _deepstack_text_layer_indices,
    )

    config = SimpleNamespace(deepstack_visual_indexes=[5, 11, 17])
    assert _deepstack_text_layer_indices(config) == [0, 1, 2]


def test_shutdown_invalidates_registered_graphs_before_allocator_teardown() -> None:
    source = (ROOT / "src" / "core" / "rpu_backend.cpp").read_text(
        encoding="utf-8"
    )
    shutdown = source.split("void rpu_shutdown()", 1)[1]
    assert shutdown.index("g_rpu_backend_lifecycle = RpuBackendLifecycleState::kShutdown") < shutdown.index(
        "invalidate_registered_rpu_kernel_graphs()"
    )
    assert shutdown.index("invalidate_registered_rpu_kernel_graphs()") < shutdown.index(
        "rpu::RPUCachingAllocator::get().mark_shutdown()"
    )


def test_native_loader_explicit_path_precedes_installed_extension(tmp_path: Path) -> None:
    from rpu_backend._native_loader import find_shared_object

    package_dir = tmp_path / "python" / "rpu_backend"
    package_dir.mkdir(parents=True)
    external = tmp_path / "custom" / "rpu_backend.so"
    external.parent.mkdir()
    external.write_bytes(b"external")
    packaged = package_dir / "rpu_backend.so"
    packaged.write_bytes(b"packaged")
    assert find_shared_object(package_dir=str(package_dir), environ={}) == str(packaged)
    assert find_shared_object(package_dir=str(package_dir),
                              environ={"RPU_BACKEND_SO": str(external)}) == str(external)
    with pytest.raises(FileNotFoundError, match="RPU_BACKEND_SO"):
        find_shared_object(package_dir=str(package_dir),
                           environ={"RPU_BACKEND_SO": str(tmp_path / "missing.so")})


def test_native_loader_source_build_scope_and_priority(tmp_path: Path) -> None:
    from rpu_backend._native_loader import find_shared_object

    package_dir = tmp_path / "python" / "rpu_backend"
    package_dir.mkdir(parents=True)
    standard = tmp_path / "build" / "nested" / "rpu_backend.so"
    current = tmp_path / "build_current" / "nested" / "rpu_backend.so"
    unrelated = tmp_path / "build_other" / "rpu_backend.so"
    for path in (standard, current, unrelated):
        path.parent.mkdir(parents=True)
        path.write_bytes(b"extension")
    kwargs = dict(package_dir=str(package_dir), executable="/envs/current/bin/python", environ={})
    assert find_shared_object(**kwargs) == str(current)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='rhinoforge'\n")
    assert find_shared_object(**kwargs) == str(current)
    current.unlink()
    assert find_shared_object(**kwargs) == str(standard)
    standard.unlink()
    assert find_shared_object(**kwargs) is None


def test_graph_structured_diagnostics_delegate_without_backend_requirement():
    from types import SimpleNamespace
    from rpu_backend.graph import Graph

    graph = Graph.__new__(Graph)
    graph._impl = None
    assert graph.debug_nodes() == []
    assert graph.debug_segments() == []
    assert graph.debug_segments_jsonl() == ""
    nodes = [{"kind": "kernel", "node_id": 0}]
    segments = [{"segment_id": 0}]
    graph._impl = SimpleNamespace(debug_nodes=lambda: nodes,
                                 debug_segments=lambda: segments,
                                 debug_segments_jsonl=lambda: '{"segment_id":0}\n')
    assert graph.debug_nodes() == nodes
    assert graph.debug_segments() == segments
    assert graph.debug_segments_jsonl() == '{"segment_id":0}\n'


@pytest.mark.parametrize(
    ("setting", "profiler_enabled", "expected"),
    [(None, False, True), (None, True, False), ("0", False, False)],
)
def test_idle_record_scope_default_is_profiler_safe(
    setting, profiler_enabled, expected, monkeypatch
) -> None:
    import torch

    from rpu_backend.graph import _runtime

    if setting is None:
        monkeypatch.delenv("RPU_SKIP_IDLE_RECORD_FUNCTION", raising=False)
    else:
        monkeypatch.setenv("RPU_SKIP_IDLE_RECORD_FUNCTION", setting)
    monkeypatch.setattr(_runtime, "_SKIP_IDLE_RF_ENV", None)
    monkeypatch.setattr(
        torch.autograd, "_profiler_enabled", lambda: profiler_enabled
    )
    assert _runtime._skip_idle_record_function() is expected
