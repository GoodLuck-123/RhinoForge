"""CPU transport tests: mirror equivalence is required before any publication."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/_model_download.py"
spec = importlib.util.spec_from_file_location("verified_weights", PATH)
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
SHA = "a" * 40
DATA = b"official checkpoint bytes"
DIGEST = hashlib.sha256(DATA).hexdigest()


def configure(monkeypatch, name="model.safetensors", mirror_digest=DIGEST):
    monkeypatch.setattr(transport.shutil, "which", lambda _: None)
    file = SimpleNamespace(rfilename=name, size=len(DATA), lfs=SimpleNamespace(sha256=DIGEST))
    hub = ModuleType("huggingface_hub")
    hub.HfApi = lambda: SimpleNamespace(model_info=lambda *a, **kw:
        SimpleNamespace(sha=SHA, siblings=[file]))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(transport, "urlopen", lambda *a, **kw: io.BytesIO(json.dumps({
        "Data": {"Files": [{"Path": name, "Sha256": mirror_digest, "Size": len(DATA)}]}
    }).encode()))


def test_mismatched_mirror_is_rejected_before_download(monkeypatch, tmp_path):
    configure(monkeypatch, mirror_digest="b" * 64)
    monkeypatch.setattr(transport.subprocess, "run", lambda *a, **kw:
                        pytest.fail("mismatched mirror must not be downloaded"))
    with pytest.raises(ValueError, match="mirror differs"):
        transport.fetch_weights("owner/repo", SHA, "c" * 40, tmp_path)
    assert not list(tmp_path.iterdir())


def test_existing_tampered_asset_is_not_accepted(monkeypatch, tmp_path):
    configure(monkeypatch)
    (tmp_path / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="download differs"):
        transport.fetch_weights("owner/repo", SHA, "c" * 40, tmp_path)


def test_explicit_ckpt_is_verified_before_atomic_publication(monkeypatch, tmp_path):
    configure(monkeypatch, name="model.ckpt")
    calls = []

    def download(command, **kwargs):
        calls.append(command)
        Path(command[command.index("--output") + 1]).write_bytes(DATA)

    monkeypatch.setattr(transport.subprocess, "run", download)
    transport.fetch_weights("Owner/Repo", SHA, "c" * 40, tmp_path,
                            filenames=("model.ckpt",), mirror_repo_id="owner/Repo")
    assert (tmp_path / "model.ckpt").read_bytes() == DATA
    assert not list(tmp_path.glob("*.download"))
    assert "models/owner/Repo/" in calls[0][-1]


def test_existing_platform_parallel_downloader_keeps_the_same_hash_gate(monkeypatch, tmp_path):
    configure(monkeypatch)
    monkeypatch.setattr(transport.shutil, "which", lambda _: "/usr/bin/aria2c")
    calls = []

    def download(command, **kwargs):
        calls.append(command)
        folder = Path(command[command.index("--dir") + 1])
        (folder / command[command.index("--out") + 1]).write_bytes(DATA)

    monkeypatch.setattr(transport.subprocess, "run", download)
    transport.fetch_weights("Owner/Repo", SHA, "c" * 40, tmp_path)
    assert calls[0][0] == "aria2c"
    assert "--file-allocation=none" in calls[0]
    assert (tmp_path / "model.safetensors").read_bytes() == DATA


def test_incomplete_explicit_selection_is_rejected(monkeypatch, tmp_path):
    configure(monkeypatch, name="model.ckpt")
    with pytest.raises(ValueError, match="absent"):
        transport.fetch_weights("owner/repo", SHA, "c" * 40, tmp_path,
                                filenames=("model.ckpt", "missing.ckpt"))


def test_destination_symlink_cannot_escape_task_root(monkeypatch, tmp_path):
    configure(monkeypatch)
    root = tmp_path / "download"
    root.mkdir()
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(DATA)
    (root / "model.safetensors").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink escapes"):
        transport.fetch_weights("owner/repo", SHA, "c" * 40, root)


@pytest.mark.parametrize("name", ["../model.ckpt", "nested/../model.ckpt", "/model.ckpt"])
def test_source_weight_paths_are_canonical_relative(monkeypatch, tmp_path, name):
    configure(monkeypatch, name=name)
    with pytest.raises(ValueError, match="filename must stay"):
        transport.fetch_weights("owner/repo", SHA, "c" * 40, tmp_path, filenames=(name,))
