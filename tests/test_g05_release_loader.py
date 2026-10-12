"""Host checks for the source-distributed G05 K1 loader boundary."""
from __future__ import annotations

import hashlib
import importlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "model.pt", "config.yaml", "dataset_stats.json", "tokenizer.pt",
    "hf_processor.pt", "action_tokenizer.pt",
)


@pytest.fixture
def release(monkeypatch, tmp_path):
    loader = importlib.import_module("rpu_backend.adapters.g05.release_replay")
    root = tmp_path / "release"
    package = root / "vla_rtc"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "loader.py").write_text(
        "def load_for_eval(checkpoint, device):\n"
        "    raise AssertionError('unexpected weight load')\n"
    )
    checkpoint = root / "checkpoint"
    checkpoint.mkdir()
    expected = {}
    for name in FILES:
        data = f"fixture {name}".encode()
        (checkpoint / name).write_bytes(data)
        expected[name] = hashlib.sha256(data).hexdigest()
    # Small files exercise the real validator without loading model weights.
    monkeypatch.setattr(loader, "CERTIFIED_CHECKPOINT_SHA256", expected)
    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.setattr(sys, "path", sys.path.copy())
    for name in tuple(sys.modules):
        if name == "vla_rtc" or name.startswith("vla_rtc."):
            monkeypatch.delitem(sys.modules, name)
    yield SimpleNamespace(loader=loader, root=root, checkpoint=checkpoint)
    for name in tuple(sys.modules):
        if name == "vla_rtc" or name.startswith("vla_rtc."):
            sys.modules.pop(name, None)


def test_source_package_imports_real_loader_without_release_installed(tmp_path):
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0",
               PYTHONPATH=str(ROOT / "python"))
    script = (
        "import pathlib, sys\n"
        "from rpu_backend.adapters.g05 import release_replay as m\n"
        "assert pathlib.Path(m.__file__).resolve() == pathlib.Path(sys.argv[1]).resolve()\n"
        "assert callable(m._resolve_release_paths)\n"
        "assert callable(m._checkpoint_manifest)\n"
        "assert callable(m._release_import_transaction)\n"
        "assert 'vla_rtc' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script,
         str(ROOT / "python/rpu_backend/adapters/g05/release_replay.py")],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_resolve_default_and_explicit_source_root(release, tmp_path):
    loader, root, checkpoint = release.loader, release.root, release.checkpoint
    assert loader._resolve_release_paths(checkpoint, None) == (root, checkpoint)
    link = tmp_path / "checkpoint-link"
    link.symlink_to(checkpoint, target_is_directory=True)
    assert loader._resolve_release_paths(link, root) == (root, checkpoint)
    with pytest.raises(FileNotFoundError, match="vla_rtc/loader.py"):
        loader._resolve_release_paths(checkpoint, tmp_path)


def test_manifest_records_all_verified_files(release):
    manifest = release.loader._checkpoint_manifest(release.checkpoint)
    assert set(manifest) == set(FILES)
    for name, item in manifest.items():
        path = release.checkpoint / name
        assert item == {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}


@pytest.mark.parametrize("name", FILES)
@pytest.mark.parametrize("failure", ["missing", "changed"])
def test_runner_rejects_every_checkpoint_file_before_release_import(
        monkeypatch, release, name, failure):
    import rpu_backend.adapters.g05 as adapter
    from examples.runners.g05 import runner as entry

    path = release.checkpoint / name
    path.unlink() if failure == "missing" else path.write_bytes(b"wrong identity")
    patch = Mock()
    monkeypatch.setattr(adapter, "patch_g05_policy_for_rpu", patch)
    monkeypatch.setattr(entry, "_load_mapping", lambda *args: {"samples": [], "pixel_values": {}})
    owner = SimpleNamespace(source_config={"example": {"target": "g05"}, "input": {
        "checkpoint": str(release.checkpoint), "release_root": str(release.root)}}, owner=None)
    expected = FileNotFoundError if failure == "missing" else ValueError
    with pytest.raises(expected, match=name):
        entry.prepare(owner)
    assert "vla_rtc.loader" not in sys.modules
    assert str(release.root) not in sys.path
    assert owner.owner is None
    patch.assert_not_called()


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("finish", ["commit", "return", "raise"])
def test_import_path_commit_and_rollback(monkeypatch, release, existing, finish):
    value = str(release.root)
    if existing:
        monkeypatch.syspath_prepend(value)
    before = sys.path.copy()
    try:
        with release.loader._release_import_transaction(release.root) as commit:
            assert value in sys.path
            if finish == "raise":
                raise RuntimeError("failed load")
            if finish == "commit":
                commit()
    except RuntimeError:
        assert finish == "raise"
    if finish == "commit" and not existing:
        assert sys.path == [value, *before]
    else:
        assert sys.path == before


def test_runner_loads_verified_release_on_cpu_before_patch(monkeypatch, release):
    import rpu_backend.adapters.g05 as adapter
    from examples.runners.g05 import runner as entry

    (release.root / "vla_rtc/loader.py").write_text(
        "from types import SimpleNamespace\n"
        "calls = []\n"
        "policy = SimpleNamespace(_rpu_g05_fm_only_customer_release=True)\n"
        "policy.eval = lambda: policy\n"
        "policy.predict_action = lambda batch: {'action': batch['samples']}\n"
        "def load_for_eval(checkpoint, device):\n"
        "    calls.append((checkpoint, device))\n"
        "    return policy, object()\n"
    )
    def patch(policy, **kwargs):
        module = importlib.import_module("vla_rtc.loader")
        assert module.calls == [(release.checkpoint, "cpu")]
        assert policy is module.policy
    install = Mock(side_effect=patch)
    monkeypatch.setattr(adapter, "patch_g05_policy_for_rpu", install)
    monkeypatch.setattr(entry, "_load_mapping", lambda *args: {"samples": ["sample"], "pixel_values": {}})
    owner = SimpleNamespace(source_config={"example": {"target": "g05"}, "input": {
        "checkpoint": str(release.checkpoint), "release_root": str(release.root)}}, owner=None)
    infer = entry.prepare(owner)
    assert infer() == ["sample"]
    assert str(release.root) in sys.path
    install.assert_called_once_with(owner.owner, rpu_execution=None)


def test_runner_rejects_loader_cached_from_another_release(monkeypatch, release, tmp_path):
    import rpu_backend.adapters.g05 as adapter
    from examples.runners.g05 import runner as entry

    other = tmp_path / "other"
    package = other / "vla_rtc"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "loader.py").write_text(
        "def load_for_eval(*args, **kwargs):\n"
        "    raise AssertionError('foreign loader must not run')\n"
    )
    monkeypatch.syspath_prepend(str(other))
    importlib.import_module("vla_rtc.loader")
    patch = Mock()
    monkeypatch.setattr(adapter, "patch_g05_policy_for_rpu", patch)
    monkeypatch.setattr(entry, "_load_mapping", lambda *args: {"samples": [], "pixel_values": {}})
    owner = SimpleNamespace(source_config={"example": {"target": "g05"}, "input": {
        "checkpoint": str(release.checkpoint), "release_root": str(release.root)}}, owner=None)
    with pytest.raises(RuntimeError, match="different release root"):
        entry.prepare(owner)
    assert str(release.root) not in sys.path
    assert owner.owner is None
    patch.assert_not_called()
