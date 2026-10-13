"""Public asset preparation preserves observation bytes and emits inputs only."""
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_rhinovla_assets", ROOT / "scripts/prepare_rhinovla_assets.py")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def image_bytes():
    from PIL import Image

    stream = io.BytesIO()
    Image.new("RGB", (256, 256), "red").save(stream, format="PNG")
    return stream.getvalue()


def parquet_fixture(monkeypatch, tmp_path, **changes):
    data = image_bytes()
    row = {"episode_index": 0, "frame_index": 0, "task_index": 0,
           "observation.images.image": {"bytes": data},
           "observation.images.wrist_image": {"bytes": data}}
    row.update(changes)
    parquet = tmp_path / "episode.parquet"
    parquet.write_bytes(b"pinned parquet bytes")
    monkeypatch.setattr(prepare, "PARQUET_SHA256", prepare.sha256(parquet))
    module = ModuleType("pyarrow.parquet")
    module.ParquetFile = lambda path: SimpleNamespace(iter_batches=lambda batch_size: iter([
        SimpleNamespace(to_pylist=lambda: [row])]))
    arrow = ModuleType("pyarrow")
    arrow.parquet = module
    monkeypatch.setitem(sys.modules, "pyarrow", arrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", module)
    return parquet, data


def test_inputs_only_preserves_images_and_never_loads_model(monkeypatch, tmp_path):
    parquet, original = parquet_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(prepare, "generate_cpu_fixture", lambda *args: pytest.fail("fixture generation called"))
    output = tmp_path / "inputs"
    assert prepare.main(["--libero-parquet", str(parquet), "--output-dir", str(output), "--inputs-only"]) == 0
    assert (output / "image.png").read_bytes() == original
    assert (output / "wrist_image.png").read_bytes() == original
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["view_roles"] == ["side", "hand"]
    assert set(manifest["files_sha256"]) == {"image.png", "wrist_image.png"}
    with pytest.raises(FileExistsError):
        prepare.main(["--libero-parquet", str(parquet), "--output-dir", str(output), "--inputs-only"])


def test_parquet_pin_and_observation_identity_are_required(monkeypatch, tmp_path):
    parquet, _ = parquet_fixture(monkeypatch, tmp_path, frame_index=1)
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError, match="frame 0"):
        prepare.export_inputs(parquet, output)
    parquet.write_bytes(b"different parquet")
    with pytest.raises(ValueError, match="SHA-256"):
        prepare.export_inputs(parquet, output)
    assert not list(output.iterdir())


@pytest.mark.parametrize("changed,value", [("commit", "0" * 40), ("dirty", "?? extra.py"), ("origin", "https://example.org/model")])
def test_source_identity_rejects_unreviewed_code(monkeypatch, tmp_path, changed, value):
    identity = {"commit": prepare.SOURCE_SHA, "dirty": "", "origin": prepare.SOURCE_URL + ".git"}
    identity[changed] = value

    def git(args, **kwargs):
        return identity[{"rev-parse": "commit", "status": "dirty", "remote": "origin"}[args[3]]]

    monkeypatch.setattr(prepare.subprocess, "check_output", git)
    with pytest.raises(ValueError):
        prepare._source_identity(tmp_path)


def test_fixture_matches_runner_input_contract_without_reference_outputs(monkeypatch, tmp_path):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch

    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint is hashed, never loaded")
    monkeypatch.setattr(prepare, "CHECKPOINT_SHA256", prepare.sha256(checkpoint))
    qin = {"input_ids": torch.ones((1, 171), dtype=torch.long),
           "attention_mask": torch.ones((1, 171), dtype=torch.long),
           "pixel_values": torch.zeros((392, 1536)),
           "image_grid_thw": torch.tensor([[1, 14, 14], [1, 14, 14]])}
    calls = []

    def build(processor, images, instructions, **options):
        calls.append((images, instructions, options))
        return qin

    monkeypatch.setattr(prepare, "_load_preprocessor", lambda source: (build, object(), {"cot_prompt_present": False}))
    inputs = {"view_roles": ["side", "hand"], "view_modalities": ["rgb", "rgb"]}
    for name in ("image", "wrist_image"):
        (tmp_path / f"{name}.png").write_bytes(image_bytes())
    prepare.generate_cpu_fixture(tmp_path, checkpoint, tmp_path, inputs)
    actual = torch.load(tmp_path / "cpu_fixture.pt", weights_only=True)
    assert set(actual) == {"metadata", "qwen_inputs", "noise", "state", "state_mask", "action_mask"}
    assert actual["metadata"]["contents"] == "inputs-only"
    assert actual["noise"].shape == (1, 30, 72)
    assert actual["state"].shape == (72,)
    assert actual["action_mask"].tolist() == [1] * 64 + [0] * 8
    assert all(t.device.type == "cpu" for t in actual["qwen_inputs"].values())
    assert calls[0][1] == [prepare.INSTRUCTION]
    assert calls[0][2]["view_roles"] == [["side", "hand"]]
    (tmp_path / "cpu_fixture.pt").unlink()
    qin["input_ids"] = torch.ones((1, 170), dtype=torch.long)
    with pytest.raises(ValueError, match="P171"):
        prepare.generate_cpu_fixture(tmp_path, checkpoint, tmp_path, inputs)
    assert not (tmp_path / "cpu_fixture.pt").exists()
