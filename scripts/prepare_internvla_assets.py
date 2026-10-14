#!/usr/bin/env python3
"""Prepare InternVLA-N1 model components and full-chain example inputs on CPU."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types

REPO_ID = "InternRobotics/InternVLA-N1-w-NavDP"
REVISION = "6d48df24da45e8b31fdcef9572bbcbf68c0baf64"
MODELSCOPE_REVISION = "f0458b17d5cf475728c73d3dea8fc5da5a66c9ea"
INTERNNAV_REVISION = "7a5c62400ac45b313d9b709c740b64191556a242"
BUNDLE_NAME = "internvla_n1-controlled"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_source(source: Path) -> None:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()
    if git("rev-parse", "HEAD") != INTERNNAV_REVISION or git("status", "--porcelain"):
        raise ValueError("InternNav source must be the clean, pinned source revision")


def extract_assets(snapshot: Path, bundle: Path, source: Path) -> None:
    """Extract paired model components and quantize decoder rows to W8."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    config = json.loads((snapshot / "config.json").read_text())
    if (config.get("hidden_size"), config.get("num_hidden_layers"),
            config.get("vocab_size"), config.get("n_query")) != (3584, 28, 151668, 4):
        raise ValueError("checkpoint does not match the exact InternVLA-N1 profile")
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    handles = {name: safe_open(str(snapshot / name), framework="pt")
               for name in sorted(set(index.values()))}

    def get(key):
        return handles[index[key]].get_tensor(key)

    bundle.mkdir(parents=True, exist_ok=True)
    nav = {key.removeprefix("model.navdp."): get(key).contiguous()
           for key in index if key.startswith("model.navdp.")}
    if not nav:
        raise ValueError("checkpoint is missing the NavDP head")
    save_file(nav, str(bundle / "navdp_head.safetensors"))
    del nav
    save_file({key: get(key).contiguous() for key in
               ("model.embed_tokens.weight", "model.latent_queries")},
              str(bundle / "s2_frontend.safetensors"))
    vit = bundle / "s2_vit"
    vit.mkdir(exist_ok=True)
    save_file({key: get(key).to(torch.float16).contiguous()
               for key in index if key.startswith("visual.")},
              str(vit / "model.safetensors"))
    (vit / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    decoder = {}
    for key in sorted(index):
        if not (key.startswith("model.layers.") or key == "model.norm.weight"):
            continue
        weight = get(key).to(torch.float16)
        if weight.ndim == 2 and key.endswith("_proj.weight"):
            scale = (weight.float().abs().amax(dim=1) / 127).to(torch.float16)
            scale = scale.clamp_min(torch.finfo(torch.float16).tiny)
            decoder[key] = torch.round(weight.float() / scale.float()[:, None]).clamp(
                -128, 127).to(torch.int8).contiguous()
            decoder[key + "_scale"] = scale.contiguous()
        else:
            decoder[key] = weight.contiguous()
    target = bundle / "s2_w8a16"
    target.mkdir(exist_ok=True)
    save_file(decoder, str(target / "model.safetensors"))
    (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    del decoder, handles
    gc.collect()
    src = source / "internnav/model/encoder/depth_anything/depth_anything_v2"
    dst = bundle / "rgbd_src"
    dst.mkdir(exist_ok=True)
    # Only the pure-torch DINOv2 implementation is consumed by the adapter.
    shutil.copy2(src / "dinov2.py", dst / "dinov2.py")
    for path in sorted((src / "dinov2_layers").rglob("*.py")):
        target = dst / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    (dst / "__init__.py").write_text("")
    for name in ("LICENSE", "LICENSE.txt"):
        if (source / name).is_file():
            shutil.copy2(source / name, dst / name)


def load_system2(source: Path, snapshot: Path):
    """Load the pinned System-2 implementation without its unused NavDP head."""
    import torch
    from transformers import AutoProcessor, AutoTokenizer

    package = "internnav.model.basemodel.internvla_n1"
    for name in ("internnav", "internnav.model", "internnav.model.encoder",
                 "internnav.model.basemodel", package):
        module = types.ModuleType(name)
        module.__path__ = [str(source / name.replace(".", "/"))]
        sys.modules[name] = module
    for suffix, names in (
        ("navdp", ("NavDP_Policy_DPT_CriticSum_DAT",)),
        ("nextdit_crossattn_traj", ("NextDiTCrossAttn", "NextDiTCrossAttnConfig")),
        ("nextdit_traj", ("NextDiT", "NextDiTConfig")),
    ):
        module = types.ModuleType(f"{package}.{suffix}")
        for name in names:
            setattr(module, name, type(name, (), {
                "__init__": lambda self, *args, **kwargs: None,
                "load_model": lambda self: None,
            }))
        sys.modules[module.__name__] = module
    implementation = importlib.import_module(f"{package}.internvla_n1")
    model = implementation.InternVLAN1ForCausalLM.from_pretrained(
        str(snapshot), torch_dtype=torch.bfloat16,
        attn_implementation="eager", device_map={"": "cpu"},
        local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True)
    processor.tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), use_fast=True, local_files_only=True,
    )
    processor.tokenizer.padding_side = "left"
    return model, processor, implementation


def make_fixture(model, processor, implementation, image, instruction: str):
    """Prepare inputs consumed by the example; no reference outputs are saved."""
    import torch

    prompt = (
        f"You are an autonomous navigation assistant. Your task is to {instruction}. "
        "Where should you go next to stay on track? Please output the next "
        "waypoint's coordinates in the image. Please output STOP when you have "
        "successfully completed the task. you can see "
    )
    conversation = [{"role": "user", "content": [
        {"type": "text", "text": prompt}, {"type": "image", "image": image},
        {"type": "text", "text": "."},
    ]}]
    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=64, do_sample=False, use_cache=True,
            past_key_values=None, return_dict_in_generate=True,
        )
        ids = generated.sequences.cpu()
        extended = torch.cat([ids, torch.full(
            (1, model.get_n_query()), implementation.TRAJ_TOKEN_INDEX,
            dtype=ids.dtype,
        )], dim=1)
        positions, _ = model.get_rope_index(extended, inputs.image_grid_thw)
    generator = torch.Generator().manual_seed(20260707)
    fixture = {
        "pixel_values": inputs.pixel_values.cpu(),
        "grid_thw": inputs.image_grid_thw.cpu(),
        "output_ids": ids,
        "image_idx": ids == implementation.IMAGE_TOKEN_INDEX,
        "position_ids": positions.cpu(),
        "rgb": torch.rand((1, 2, 224, 224, 3), generator=generator),
        "depth": torch.rand((1, 2, 224, 224, 1), generator=generator),
        "init_noise": torch.randn((4, 32, 3), generator=torch.Generator().manual_seed(1)),
    }
    validate_fixture(fixture)
    return fixture


def validate_fixture(fixture: dict) -> None:
    import torch

    if fixture["grid_thw"].tolist() != [[1, 28, 28]]:
        raise ValueError("System-2 requires a 384px input with grid [[1,28,28]]")
    pixels = fixture["pixel_values"]
    ids, image_idx, positions = (fixture[key] for key in
                                 ("output_ids", "image_idx", "position_ids"))
    if (ids.dtype != torch.int64 or ids.ndim != 2 or ids.shape[0] != 1
            or ids.numel() == 0 or int(ids.min()) < 0 or int(ids.max()) >= 151668):
        raise ValueError("output_ids must be [1,S] int64 within the vocabulary")
    if (image_idx.dtype != torch.bool or image_idx.shape != ids.shape
            or not torch.equal(image_idx, ids == 151655)):
        raise ValueError("image_idx must identify the image tokens in output_ids")
    if pixels.shape[0] != 784 or int(image_idx.sum()) != 196:
        raise ValueError("System-2 requires 784 patches and 196 image tokens")
    if (positions.dtype != torch.int64 or tuple(positions.shape) not in
            ((3, 1, ids.shape[1] + 4), (ids.shape[1] + 4, 3))
            or int(positions.min()) < 0):
        raise ValueError("position_ids must cover text plus four latent rows")
    if not bool(pixels.isfinite().all()):
        raise ValueError("pixel_values must be finite")
    for key, shape in {
        "rgb": (1, 2, 224, 224, 3), "depth": (1, 2, 224, 224, 1),
        "init_noise": (4, 32, 3),
    }.items():
        tensor = fixture[key]
        if tuple(tensor.shape) != shape or not tensor.is_floating_point():
            raise ValueError(f"unexpected shape or dtype: {key}")
        if not bool(tensor.isfinite().all()):
            raise ValueError(f"non-finite input: {key}")


def write_manifest(bundle: Path) -> dict[str, str]:
    manifest = {str(path.relative_to(bundle)): sha256(path)
                for path in sorted(bundle.rglob("*"))
                if path.is_file() and path.name != "asset_manifest.json"}
    (bundle / "asset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path,
                        help="parent directory for the new model bundle")
    parser.add_argument("--hf-cache", required=True, type=Path,
                        help="download or reuse the pinned public checkpoint in this cache")
    parser.add_argument("--internnav-root", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true",
                        help="extract weights without generating input tensors")
    parser.add_argument("--modelscope-weights", action="store_true",
                        help="use the official mirror after checking Hugging Face hashes")
    parser.add_argument("--rgb-input", type=Path,
                        help="System-2 image; defaults to a seeded synthetic image")
    parser.add_argument("--instruction", default=(
        "walk forward through the doorway and turn left at the end of the hallway"))
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    source = args.internnav_root.expanduser().resolve()
    target = root / BUNDLE_NAME
    if target.exists():
        raise ValueError("output bundle already exists; select a fresh output root")
    validate_source(source)
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if not args.prepare_only and importlib.metadata.version("transformers") != "4.51.0":
        raise ValueError("input preparation requires transformers==4.51.0 in a separate CPU environment")
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(
        REPO_ID, revision=REVISION, cache_dir=args.hf_cache.expanduser().resolve(),
        allow_patterns=["*.json", "*.txt", "*.jinja"] + (
            [] if args.modelscope_weights else ["*.safetensors"]), max_workers=2,
    ))
    if args.modelscope_weights:
        from _model_download import fetch_weights
        fetch_weights(REPO_ID, REVISION, MODELSCOPE_REVISION, snapshot)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".internvla-", dir=root) as temporary:
        bundle = Path(temporary) / BUNDLE_NAME
        extract_assets(snapshot, bundle, source)
        if not args.prepare_only:
            import torch
            import numpy as np
            from PIL import Image

            torch.manual_seed(0)
            if args.rgb_input:
                with Image.open(args.rgb_input.expanduser()) as raw:
                    image = raw.convert("RGB").resize((384, 384))
            else:
                rng = np.random.default_rng(0)
                image = Image.fromarray(rng.integers(60, 200, (384, 384, 3)).astype(np.uint8))
            model, processor, implementation = load_system2(source, snapshot)
            fixture = make_fixture(model, processor, implementation, image, args.instruction)
            torch.save(fixture, bundle / "full_chain_fixture.pt")
        provenance = {
            "schema": 1, "repo_id": REPO_ID, "revision": REVISION,
            "internnav_revision": INTERNNAV_REVISION,
            "fixture_complete": not args.prepare_only,
            "input_kind": "image-file" if args.rgb_input else "synthetic",
            "rgbd_input_kind": "synthetic", "instruction": args.instruction,
            "producer_sha256": sha256(Path(__file__)),
            "weight_transport": "modelscope" if args.modelscope_weights else "huggingface",
        }
        (bundle / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        manifest = write_manifest(bundle)
        bundle.rename(target)
    print(f"Prepared {len(manifest)} files in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
