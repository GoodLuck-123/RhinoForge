#!/usr/bin/env python3
"""Link a local model directory into Hugging Face's snapshot cache."""
from __future__ import annotations

import argparse
from pathlib import Path

from install_hf_env_pth import registry


def cache_root() -> Path:
    return registry().cache_root()


def hf_cache_root() -> Path:
    return cache_root() / ".hf"


def link(repo_id: str, flat_subdir: str, revision: str = "local") -> Path:
    """Create ``hub/models--<org>--<name>/snapshots/<revision>/`` symlinks
    into ``cache_root() / flat_subdir``."""
    flat_dir = cache_root() / flat_subdir
    if not flat_dir.is_dir():
        raise FileNotFoundError(f"Flat dir does not exist: {flat_dir}")

    org, name = repo_id.split("/", 1)
    hub_root = hf_cache_root() / "hub" / f"models--{org}--{name}"
    snap = hub_root / "snapshots" / revision
    refs = hub_root / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    snap.mkdir(parents=True, exist_ok=True)

    (refs / "main").write_text(revision)

    for src in flat_dir.iterdir():
        if src.name.startswith("."):
            continue  # skip .cache, .gitattributes, etc.
        dst = snap / src.name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())

    return snap


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("repo_id", help='HF model id, e.g. "Qwen/Qwen3-0.6B"')
    ap.add_argument("flat_subdir", help='Flat-dir basename under cache root, e.g. "Qwen3-0.6B"')
    ap.add_argument("--revision", default="local", help='Snapshot revision (default: "local")')
    args = ap.parse_args()

    snap = link(args.repo_id, args.flat_subdir, args.revision)
    print(f"OK: bridged {args.repo_id} -> {snap}")
    print("Verify:")
    print(f"  HF_HUB_OFFLINE=1 python -c \"from transformers import AutoConfig; "
          f"print(AutoConfig.from_pretrained('{args.repo_id}'))\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
