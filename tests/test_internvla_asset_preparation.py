"""CPU preparation preserves model component and example input contracts."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import types

import pytest
import torch
from safetensors.torch import load_file, save_file


@pytest.fixture
def prep():
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_internvla_assets.py"
    spec = importlib.util.spec_from_file_location("internvla_asset_preparation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def checkpoint(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    config = dict(hidden_size=3584, num_hidden_layers=28, vocab_size=151668, n_query=4)
    (snapshot / "config.json").write_text(json.dumps(config))
    weights = {
        "model.navdp.layer.weight": torch.ones(2, 3, dtype=torch.bfloat16),
        "model.embed_tokens.weight": torch.ones(4, 3, dtype=torch.bfloat16),
        "model.latent_queries": torch.ones(1, 4, 3, dtype=torch.bfloat16),
        "visual.proj.weight": torch.ones(2, 3, dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_proj.weight": torch.tensor(
            [[1.0, -0.5, 0.0], [0.0, 0.0, 0.0]], dtype=torch.bfloat16),
        "model.norm.weight": torch.ones(3, dtype=torch.bfloat16),
    }
    save_file(weights, str(snapshot / "weights.safetensors"))
    (snapshot / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {key: "weights.safetensors" for key in weights},
    }))
    source = tmp_path / "source"
    code = source / "internnav/model/encoder/depth_anything/depth_anything_v2"
    (code / "dinov2_layers").mkdir(parents=True)
    (code / "dinov2.py").write_text("class Model: pass\n")
    (code / "dinov2_layers/__init__.py").write_text("")
    (source / "LICENSE").write_text("model source license\n")
    return snapshot, source


def test_extract_paired_components_and_quantize_zero_row(prep, checkpoint, tmp_path):
    snapshot, source = checkpoint
    bundle = tmp_path / "bundle"
    prep.extract_assets(snapshot, bundle, source)
    decoder = load_file(str(bundle / "s2_w8a16/model.safetensors"))
    key = "model.layers.0.self_attn.q_proj.weight"
    assert decoder[key].dtype == torch.int8
    assert decoder[key].tolist() == [[127, -64, 0], [0, 0, 0]]
    assert decoder[key + "_scale"].dtype == torch.float16
    assert bool((decoder[key + "_scale"] > 0).all())
    assert decoder["model.norm.weight"].dtype == torch.float16
    assert load_file(str(bundle / "navdp_head.safetensors"))["layer.weight"].dtype == torch.bfloat16
    assert load_file(str(bundle / "s2_vit/model.safetensors"))["visual.proj.weight"].dtype == torch.float16
    assert (bundle / "rgbd_src/LICENSE").read_bytes() == (source / "LICENSE").read_bytes()
    manifest = prep.write_manifest(bundle)
    assert "asset_manifest.json" not in manifest
    assert manifest["s2_w8a16/model.safetensors"] == prep.sha256(bundle / "s2_w8a16/model.safetensors")
    assert manifest["rgbd_src/dinov2_layers/__init__.py"] == prep.sha256(bundle / "rgbd_src/dinov2_layers/__init__.py")


def test_make_fixture_matches_runner_input_schema_without_reference_outputs(prep):
    class Inputs(dict):
        __getattr__ = dict.__getitem__

    class Processor:
        def apply_chat_template(self, conversation, **kwargs):
            assert conversation[0]["content"][1]["type"] == "image"
            return "prepared prompt"

        def __call__(self, **kwargs):
            return Inputs(input_ids=torch.zeros(1, 200, dtype=torch.int64),
                          pixel_values=torch.ones(784, 4),
                          image_grid_thw=torch.tensor([[1, 28, 28]]))

    class Model:
        def generate(self, **kwargs):
            assert kwargs["do_sample"] is False
            ids = torch.ones(1, 202, dtype=torch.int64)
            ids[:, :196] = 151655
            return types.SimpleNamespace(sequences=ids)

        def get_n_query(self):
            return 4

        def get_rope_index(self, ids, grid):
            assert ids.shape == (1, 206)
            return torch.arange(206).expand(3, 1, -1).clone(), None

    implementation = types.SimpleNamespace(TRAJ_TOKEN_INDEX=151667, IMAGE_TOKEN_INDEX=151655)
    result = prep.make_fixture(Model(), Processor(), implementation, object(), "walk forward")
    assert set(result) == {"pixel_values", "grid_thw", "output_ids", "image_idx", "position_ids",
                           "rgb", "depth", "init_noise"}
    assert result["init_noise"].shape == (4, 32, 3)
    bad = dict(result, rgb=torch.full_like(result["rgb"], float("nan")))
    with pytest.raises(ValueError, match="non-finite input"):
        prep.validate_fixture(bad)
    bad = dict(result, position_ids=torch.zeros(3, 1, 205, dtype=torch.int64))
    with pytest.raises(ValueError, match="four latent rows"):
        prep.validate_fixture(bad)


def test_preparation_failure_never_publishes_partial_bundle(prep, monkeypatch, tmp_path):
    import huggingface_hub

    monkeypatch.setattr(prep, "validate_source", lambda source: None)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **kw: tmp_path)
    def fail(snapshot, bundle, source):
        bundle.mkdir(parents=True)
        (bundle / "incomplete").write_text("partial")
        raise ValueError("incomplete checkpoint")
    monkeypatch.setattr(prep, "extract_assets", fail)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="incomplete checkpoint"):
        prep.main(["--root", str(output), "--hf-cache", str(tmp_path),
                   "--internnav-root", str(tmp_path), "--prepare-only"])
    assert list(output.iterdir()) == []


def test_existing_bundle_is_not_overwritten(prep, tmp_path):
    bundle = tmp_path / prep.BUNDLE_NAME
    bundle.mkdir()
    original = bundle / "asset_manifest.json"
    original.write_text("existing")
    with pytest.raises(ValueError, match="already exists"):
        prep.main(["--root", str(tmp_path), "--hf-cache", str(tmp_path),
                   "--internnav-root", str(tmp_path), "--prepare-only"])
    assert original.read_text() == "existing"


def test_source_requires_exact_clean_revision(prep, monkeypatch, tmp_path):
    def git(args, **kwargs):
        return prep.INTERNNAV_REVISION if "rev-parse" in args else " M model.py"
    monkeypatch.setattr(prep.subprocess, "check_output", git)
    with pytest.raises(ValueError, match="clean, pinned"):
        prep.validate_source(tmp_path)
