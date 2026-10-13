#!/usr/bin/env python3
"""Build real-token HyVLA text inputs using existing CPU image/state/noise tensors."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


DEFAULT_INSTRUCTION = (
    "Pick up the bottle from the table with the right gripper, keep it upright, and move it toward "
    "the empty target area. Avoid the cup and the other objects near the bottle. Use the left "
    "gripper to stabilize the target container if necessary. Lower the bottle gently into the "
    "target area, release the gripper when it is stable, and move both hands away from the objects. "
    "Keep the wrists level during the transfer and leave enough space between the arms. "
    "After placing the bottle, return the hands to a neutral position without moving the cup."
)
DEFAULT_INSTRUCTIONS = {
    16: "Pick up the bottle carefully with the right gripper and keep it upright.",
    32: "Pick up the bottle from the table with the right gripper. Keep it upright and move it gently toward the empty target area without touching the nearby cup.",
    48: "Pick up the bottle from the table with the right gripper. Keep it upright and move it gently toward the empty target area without touching the nearby cup. Lower it very gently and release the gripper once the bottle is fully stable.",
    64: "Pick up the bottle from the table with the right gripper. Keep it upright and move it gently toward the empty target area without touching the nearby cup. Use the left gripper to stabilize the container. Lower the bottle into the container, release it once stable, and move both hands back to a neutral position.",
}


def instruction_with_tokens(tokenizer, instruction, length):
    """Use a natural-text prefix that round-trips through the checkpoint tokenizer."""
    from rpu_backend.adapters.hy_vla.preprocess import prepare_language
    text = instruction.strip().replace("_", " ").replace("\n", " ")
    # Try the full instruction first, then complete-word prefixes. Never cut a
    # word in the middle, repeat IDs, or turn padding slots into valid tokens.
    candidates = [text] + [text[:match.end()] for match in re.finditer(r"\S+", text)]
    for candidate in candidates:
        for suffix in ("", "."):
            actual = candidate + suffix
            tokens, mask = prepare_language(actual, tokenizer, max_length=128)
            if int(mask.sum()) == length:
                checked, checked_mask = prepare_language(actual, tokenizer)
                if int(checked_mask.sum()) == length:
                    return actual, checked[0], checked_mask
    raise ValueError(f"instruction has no round-trippable prefix of exactly {length} tokens including assistant suffix")


def make_text_batch(source, tokenizer, instruction, length):
    import torch
    from rpu_backend.adapters.hy_vla import preprocess as prep
    if length not in (16, 32, 48, 64):
        raise ValueError("HyVLA text length must be 16,32,48,64")
    actual, tokens, lang_mask = instruction_with_tokens(tokenizer, instruction, length)
    obs = dict(source)
    pad, att, modality, ranges, full = prep.prefix_masks(3, 7, lang_mask)
    prefill = prep.make_att_2d_masks(pad, att)
    prep.apply_visual_segment_mask(prefill, ranges, full)
    suffix_pad, suffix_att, _ = prep.suffix_masks()
    obs.update(lang_tokens=tokens, prefill_mask=prefill[0], prefill_pos=(pad.cumsum(1)-1)[0],
        modality_mask=modality[0], n_valid=int(pad.sum()),
        denoise_mask=torch.cat((pad[:, None, :].expand(1, 51, 240), prep.make_att_2d_masks(suffix_pad, suffix_att)), 2)[0],
        suffix_pos=(pad.sum(-1)[:, None] + suffix_pad.cumsum(1)-1)[0])
    return {"observation": obs, "instruction": actual}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source-observation", type=Path,
                        help="CPU tensor dictionary exported from the original immutable observation")
    source_group.add_argument("--source-assets", type=Path,
                        help="Fixture directory with synth_images.pt, state.pt and initial_noise.pt")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--instruction-file", type=Path)
    args = parser.parse_args()
    import torch
    from transformers import AutoTokenizer
    from rpu_backend.adapters.hy_vla.preprocess import build_observation
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=False)
    custom_instruction = args.instruction_file.read_text() if args.instruction_file else None
    source_files = {}
    if args.source_assets is not None:
        files = {name: args.source_assets / filename for name, filename in
                 (("images", "synth_images.pt"), ("state", "state.pt"), ("noise", "initial_noise.pt"))}
        raw = {name: torch.load(path, map_location="cpu", weights_only=True) for name, path in files.items()}
        source = build_observation(raw["images"], "", tokenizer, state=raw["state"], noise=raw["noise"])
        image_source = "fixed synthetic images from synth_images.pt"
    else:
        files = {"observation": args.source_observation}
        source = torch.load(args.source_observation, map_location="cpu", weights_only=True)
        if "observation" in source:
            source = source["observation"]
        image_source = "unchanged source observation images; provenance supplied by caller"
    for name, path in files.items():
        with path.open("rb") as stream:
            source_files[name] = {"path": str(path), "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for length in (16, 32, 48, 64):
        instruction = custom_instruction if custom_instruction is not None else DEFAULT_INSTRUCTIONS[length]
        batch = make_text_batch(source, tokenizer, instruction, length)
        batch["source"] = {"files": source_files, "images": image_source, "text_tokens": length,
                           "instruction_source": instruction}
        path = args.output / f"hyvla_3cam_t{length}_batch.pt"
        torch.save(batch, path)
        records.append({"path": str(path), "text_tokens": length, "prefix_rows": 176+length,
                        "instruction": batch["instruction"]})
    (args.output / "inputs.json").write_text(json.dumps(records, indent=2)+"\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
