"""Board-free checks for reviewed-source admission and CPU-loader ordering."""
import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture
def package(monkeypatch):
    root = Path(__file__).parents[1] / "python/rpu_backend/adapters/qwen3_tts"
    name = "_test_qwen3_tts_source"
    spec = importlib.util.spec_from_file_location(name, root / "__init__.py", submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    yield module
    for key in list(sys.modules):
        if key.startswith(name + "."):
            sys.modules.pop(key, None)


@pytest.fixture
def reviewed_source(tmp_path, package, monkeypatch):
    source_module = sys.modules[package.__name__ + ".source"]
    upstream = tmp_path / "upstream"
    path = upstream / "qwen_tts/__init__.py"
    path.parent.mkdir(parents=True)
    # Preparation must never execute the candidate's Python code.
    content = b"raise RuntimeError('must not be imported during preparation')\n"
    path.write_bytes(content)
    hashes = {"__init__.py": hashlib.sha256(content).hexdigest()}
    monkeypatch.setattr(source_module, "SOURCE_HASHES", hashes)
    monkeypatch.setattr(source_module, "PREPARED_HASHES", hashes)
    return upstream, path, content


def test_prepare_preserves_upstream_and_rejects_changes_before_writing(tmp_path, package, reviewed_source):
    upstream, path, original = reviewed_source
    prepared = package.prepare_source(upstream, tmp_path / "prepared")
    assert path.read_bytes() == original == (prepared / "__init__.py").read_bytes()
    with pytest.raises(ValueError, match="new directory"):
        package.prepare_source(upstream, upstream / "nested")
    path.write_bytes(original + b"# changed\n")
    with pytest.raises(ValueError, match="identity mismatch"):
        package.prepare_source(upstream, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("extra", ["unreviewed.py", "__pycache__/__init__.pyc", "extension.so"])
def test_prepared_package_rejects_unreviewed_import_code(tmp_path, package, reviewed_source, extra):
    upstream, _, _ = reviewed_source
    prepared = package.prepare_source(upstream, tmp_path / "prepared")
    unexpected = prepared / extra
    unexpected.parent.mkdir(exist_ok=True)
    unexpected.write_bytes(b"unreviewed")
    with pytest.raises(ValueError, match="unreviewed code"):
        package.preflight(tmp_path / "checkpoint", prepared)


def test_bad_metadata_fails_before_weight_read_or_source_import(tmp_path, package, reviewed_source, monkeypatch):
    upstream, _, _ = reviewed_source
    prepared = package.prepare_source(upstream, tmp_path / "prepared")
    loader = sys.modules[package.__name__ + ".loader"]
    monkeypatch.setattr(loader.importlib.metadata, "version", lambda _: "5.3.0")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    monkeypatch.setattr(loader, "CHECKPOINT_HASHES", {"config.json": "invalid", "model.safetensors": "never read"})
    with pytest.raises(ValueError, match="config.json"):
        package.load_cpu_model(checkpoint, prepared)
    assert not any(key.startswith("_rhinoforge_reviewed_qwen3_tts_") for key in sys.modules)


def test_wrong_transformers_rejected_before_loading(tmp_path, package, reviewed_source, monkeypatch):
    upstream, _, _ = reviewed_source
    prepared = package.prepare_source(upstream, tmp_path / "prepared")
    loader = sys.modules[package.__name__ + ".loader"]
    monkeypatch.setattr(loader.importlib.metadata, "version", lambda _: "4.57.3")
    with pytest.raises(ValueError, match="transformers==5.3.0"):
        package.load_cpu_model(tmp_path / "absent", prepared)


def test_prepared_source_hash_is_checked_without_import(tmp_path, package, reviewed_source):
    upstream, _, _ = reviewed_source
    prepared = package.prepare_source(upstream, tmp_path / "prepared")
    (prepared / "__init__.py").write_text("raise RuntimeError('unreviewed')")
    with pytest.raises(ValueError, match="prepared source identity mismatch"):
        package.load_cpu_model(tmp_path / "absent", prepared)
