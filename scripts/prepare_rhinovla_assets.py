#!/usr/bin/env python3
"""Prepare the official non-RTC RhinoVLA v1 inputs on CPU.

Use the pinned model source, checkpoint and LIBERO parquet. The source's own
processor prepares the two images; state and initial noise use a fixed seed.
No model weights are loaded and no inference, reference output or timing is run.
``--inputs-only`` exports just the original PNG images and their manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


SOURCE_URL = "https://github.com/HUIXI-AI/RhinoVLA"
SOURCE_SHA = "a3edf0c1addf4bf849192642bff3ad15559db3f4"
HF_REPO = "HuixiAI/RhinoVLA"
HF_REVISION = "7f3708650e302f395e7c63607b48b56fb96b8076"
CHECKPOINT_SHA256 = "93be25953542bf59e1b036bcb3be6c161b9a20dc070aaceae768ddb1c6d2b145"
DATASET = "lerobot/libero_10_image"
DATASET_REVISION = "7e324b526699f444044952c82ce3f438e8d300d0"
PARQUET_SHA256 = "76a1ff332cbb0c3ae2aa4d930a4308a9c5aaf1330c8ea7c6a1e5093a32062108"
INSTRUCTION = (
    "put the white mug on the left plate and put the yellow and white mug "
    "on the right plate"
)
STEPS = 10


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _require_digest(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected}")


def _source_identity(source: Path) -> None:
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(source), *args], text=True,
        ).strip()

    if git("rev-parse", "HEAD") != SOURCE_SHA:
        raise ValueError(f"RhinoVLA source must be pinned to {SOURCE_SHA}")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("RhinoVLA source must be clean before executing model code")
    if git("remote", "get-url", "origin").removesuffix(".git") != SOURCE_URL:
        raise ValueError("RhinoVLA source must use the reviewed official origin")


def export_inputs(parquet: Path, output: Path) -> dict:
    """Reuse exact episode-0/frame-0 bytes, rather than re-rendering LIBERO."""
    import pyarrow.parquet as pq
    from PIL import Image

    _require_digest(parquet, PARQUET_SHA256)
    row = next(pq.ParquetFile(parquet).iter_batches(batch_size=1)).to_pylist()[0]
    if any(row.get(key) != 0 for key in ("episode_index", "frame_index", "task_index")):
        raise ValueError("LIBERO fixture must be episode 0 / frame 0 / task 0")
    images = {}
    for name in ("image", "wrist_image"):
        data = row[f"observation.images.{name}"]["bytes"]
        with Image.open(io.BytesIO(data)) as opened:
            if opened.format != "PNG" or opened.size != (256, 256):
                raise ValueError(f"unexpected LIBERO {name} image format/geometry")
            opened.load()
        path = output / f"{name}.png"
        with path.open("xb") as stream:
            stream.write(data)
        images[name] = sha256(path)
    return {
        "dataset": DATASET, "dataset_revision": DATASET_REVISION,
        "parquet_sha256": PARQUET_SHA256,
        "episode": 0, "frame": 0, "task": 0,
        "instruction": INSTRUCTION, "images_sha256": images,
        "view_roles": ["side", "hand"], "view_modalities": ["rgb", "rgb"],
        "state_source": "synthetic_seed0",
    }


def _load_preprocessor(source: Path):
    """Use the same CPU helper and processor options as the pinned model."""
    _source_identity(source)
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    for name, module in tuple(sys.modules.items()):
        if name == "rhinovla" or name.startswith("rhinovla."):
            location = getattr(module, "__file__", None)
            if location is None or not Path(location).resolve().is_relative_to(source):
                raise RuntimeError("a different RhinoVLA source is imported; use a fresh process")
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    try:
        from omegaconf import OmegaConf
        from transformers import AutoProcessor

        qwen = importlib.import_module("rhinovla.model.modules.qwen")
        cfg = OmegaConf.load(source / "configs/training/demo_full_finetune.yaml")
        minimum, maximum = qwen._resolve_pixel_budget(cfg)
        processor = AutoProcessor.from_pretrained(
            source / "rhinovla/assets/qwen3_vl_processor",
            min_pixels=minimum, max_pixels=maximum, local_files_only=True,
        )
        processor.tokenizer.padding_side = "left"
        data = cfg.datasets.vla_data
        options = {
            "allowed_roles": data.get("view_role_vocab", qwen._DEFAULT_VIEW_ROLE_VOCAB),
            "allowed_modalities": data.get("view_modality_vocab", qwen._DEFAULT_VIEW_MODALITY_VOCAB),
            "cot_prompt": data.get("CoT_prompt", "") if "CoT_prompt" in data else None,
            "cot_prompt_present": "CoT_prompt" in data,
        }
        return qwen.build_qwenvl_cpu_inputs, processor, options
    finally:
        sys.dont_write_bytecode = previous_bytecode
        sys.path.remove(str(source))


def generate_cpu_fixture(source: Path, checkpoint: Path, output: Path, inputs: dict) -> dict:
    """Save only the input fields consumed by the official example runner."""
    _require_digest(checkpoint, CHECKPOINT_SHA256)
    build_inputs, processor, options = _load_preprocessor(source)
    import torch
    from PIL import Image

    images = []
    for name in ("image", "wrist_image"):
        with Image.open(output / f"{name}.png") as image:
            images.append(image.convert("RGB"))
    qin = build_inputs(
        processor, [images], [INSTRUCTION],
        view_roles=[inputs["view_roles"]], view_modalities=[inputs["view_modalities"]],
        **options,
    )
    required = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
    if not required.issubset(qin) or set(qin) - required - {"mm_token_type_ids"}:
        raise ValueError("official processor returned unexpected input fields")
    qin = {name: value.detach().cpu().clone() for name, value in qin.items()}
    if qin["input_ids"].dtype != torch.long or qin["input_ids"].shape != (1, 171):
        raise ValueError("official input must retain the two-image P171 profile")
    if any(not bool(torch.isfinite(value).all()) for value in qin.values()):
        raise ValueError("official processor returned a non-finite input")
    state = torch.randn(72, generator=torch.Generator().manual_seed(0)) * 0.1
    mask = torch.ones(72)
    mask[-8:] = 0
    noise = torch.randn((1, 30, 72), generator=torch.Generator().manual_seed(0))
    metadata = {
        "schema": 1,
        # Existing readers use this identity for the FP32 input representation.
        "kind": "rhinovla_official_cpu_fp32_nonrtc",
        "contents": "inputs-only",
        "source_url": SOURCE_URL, "source_commit": SOURCE_SHA,
        "hf_repo": HF_REPO, "hf_revision": HF_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "input_dtype": "float32", "steps": STEPS,
        "flow_time_start": 1.0, "flow_dt": -1.0 / STEPS,
        "action_horizon": 30, "rtc": False, "inputs": inputs,
    }
    with (output / "cpu_fixture.pt").open("xb") as stream:
        torch.save({"metadata": metadata, "qwen_inputs": qin, "noise": noise,
                    "state": state, "state_mask": mask, "action_mask": mask.clone()}, stream)
    return metadata


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--libero-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inputs-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.inputs_only and (args.source is None or args.checkpoint is None):
        parser.error("--source and --checkpoint are required unless --inputs-only")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    inputs = export_inputs(args.libero_parquet, args.output_dir)
    metadata = inputs if args.inputs_only else generate_cpu_fixture(
        args.source.resolve(), args.checkpoint.resolve(), args.output_dir, inputs,
    )
    metadata["files_sha256"] = {
        path.name: sha256(path) for path in sorted(args.output_dir.iterdir())
    }
    with (args.output_dir / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"[write] {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
