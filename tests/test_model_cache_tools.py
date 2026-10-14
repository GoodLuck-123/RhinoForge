from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import install_hf_env_pth
import link_flat_to_hf_cache


def test_cache_assignments_quote_paths_and_replace_environment(tmp_path, monkeypatch):
    cache = tmp_path / 'weights "quoted" \\ path'
    monkeypatch.setenv("RPU_MODEL_CACHE", str(cache))
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.setattr(install_hf_env_pth.site, "getsitepackages", lambda: [str(site)])
    assert install_hf_env_pth.main() == 0
    pth = site / "rpu_backend_hf_env.pth"
    env = dict(os.environ, HF_HOME="caller-home", HF_HUB_CACHE="caller-hub",
               HUGGINGFACE_HUB_CACHE="caller-other-hub")
    checked = subprocess.run([sys.executable, "-S", "-c", pth.read_text() +
        "import json; print(json.dumps([os.environ['HF_HOME'], os.environ['HF_HUB_CACHE'], os.environ['HUGGINGFACE_HUB_CACHE']]))"],
        env=env, check=True, capture_output=True, text=True)
    home = cache / ".hf"
    assert json.loads(checked.stdout) == [str(home), str(home / "hub"), str(home / "hub")]


def test_link_snapshot_updates_refs_and_replaces_existing_links(tmp_path, monkeypatch):
    root = tmp_path / "models"
    flat = root / "model"
    flat.mkdir(parents=True)
    (flat / "config.json").write_text("{}")
    (flat / ".hidden").write_text("hidden")
    monkeypatch.setenv("RPU_MODEL_CACHE", str(root))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "other-home"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "other-hub"))
    assert link_flat_to_hf_cache.cache_root() == root
    assert link_flat_to_hf_cache.hf_cache_root() == root / ".hf"
    result = link_flat_to_hf_cache.link("Example/Model", "model", "revision")
    hub = root / ".hf" / "hub" / "models--Example--Model"
    assert result == hub / "snapshots" / "revision"
    assert (result / "config.json").resolve() == flat / "config.json"
    assert not (result / ".hidden").exists()
    assert link_flat_to_hf_cache.link("Example/Model", "model", "revision") == result
    changed = link_flat_to_hf_cache.link("Example/Model", "model", "changed")
    assert (hub / "refs" / "main").read_text() == "changed"
    assert (changed / "config.json").resolve() == flat / "config.json"
    (result / "config.json").unlink()
    (result / "config.json").write_text("old snapshot copy")
    assert link_flat_to_hf_cache.link("Example/Model", "model", "revision") == result
    assert (result / "config.json").is_symlink()
    assert (result / "config.json").resolve() == flat / "config.json"
    assert (hub / "refs" / "main").read_text() == "revision"
    assert not (tmp_path / "other-hub").exists()
