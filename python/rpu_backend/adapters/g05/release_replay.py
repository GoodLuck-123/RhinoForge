"""Checkpoint identity and source loading for the G05 K1 release."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
import sys
from typing import Any


CERTIFIED_CHECKPOINT_SHA256 = {
    "model.pt": "c51b92b07818aaf760a9b585db4abea1c25f2b235ccf5c9b5398ceca7ed833d6",
    "config.yaml": "bf6cb0d9be0fb62df4a5ee45904d9b763e9cde04a3cfe97d71918c643c1f08a3",
    "dataset_stats.json": "2bf28a32fef98bdef4128e8e79f3c71e8cd83022a5b4411f26ea48bd020b030d",
    "tokenizer.pt": "e2455a0448edaf1110ab2a6c1d94d866351ffc2d800ccb3af942c19a9281bba7",
    "hf_processor.pt": "babc02e59938dd867e3c525992f7419dea7d476a301746155aac550f36447471",
    "action_tokenizer.pt": "55740ca66a24079bdf88f390c0f91749702486a93fa37facb9c0179517297e17",
}


@contextmanager
def _release_import_transaction(release_root: Path):
    """Keep the source import path only after the caller commits a load."""
    value = str(release_root)
    inserted = value not in sys.path
    committed = False
    if inserted:
        sys.path.insert(0, value)

    def commit() -> None:
        nonlocal committed
        committed = True

    try:
        yield commit
    finally:
        if inserted and not committed:
            sys.path.remove(value)


def _resolve_release_paths(
    checkpoint: Path, release_root: Path | None
) -> tuple[Path, Path]:
    checkpoint = checkpoint.resolve()
    root = (checkpoint.parent if release_root is None else release_root).resolve()
    if not (root / "vla_rtc/loader.py").is_file():
        raise FileNotFoundError(
            f"release root must contain vla_rtc/loader.py: {root}"
        )
    for name in ("model.pt", "config.yaml", "dataset_stats.json"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    return root, checkpoint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_manifest(checkpoint: Path) -> dict[str, Any]:
    """Verify all six checkpoint files before the release loader reads them."""
    result: dict[str, Any] = {}
    for name, expected_sha256 in CERTIFIED_CHECKPOINT_SHA256.items():
        path = checkpoint / name
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"G05 K1 checkpoint identity mismatch: {name} sha256")
        result[name] = {
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": actual_sha256,
        }
    return result
