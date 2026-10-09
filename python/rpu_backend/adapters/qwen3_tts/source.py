"""Prepare and verify the exact reviewed Qwen3-TTS source without importing it."""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil

from ._compat import patch_source
from ._manifest import PREPARED_HASHES, SOURCE_HASHES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(source: str | Path) -> Path:
    """Accept only the prepared package closure, without additional Python code."""
    source = Path(source).resolve()
    actual = {str(p.relative_to(source)) for p in source.rglob("*")
              if p.is_file() and p.suffix in {".py", ".pyc", ".so"}}
    if actual != set(PREPARED_HASHES):
        raise ValueError("Qwen3-TTS prepared source has missing or unreviewed code")
    for name, expected in PREPARED_HASHES.items():
        if sha256(source / name) != expected:
            raise ValueError(f"Qwen3-TTS prepared source identity mismatch: {name}")
    return source


def prepare_source(source: str | Path, output: str | Path) -> Path:
    """Copy the reviewed upstream repository closure to a new, patched directory.

    Returns the prepared ``output/qwen_tts`` package path used by the loader.
    The upstream repository is never imported or changed.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(source):
        raise ValueError("Qwen3-TTS output must be a new directory outside the upstream source")
    prepared = {}
    for name, expected in SOURCE_HASHES.items():
        raw = (source / "qwen_tts" / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Qwen3-TTS upstream source identity mismatch: {name}")
        value = patch_source(name, raw.decode("utf-8")).encode("utf-8")
        if hashlib.sha256(value).hexdigest() != PREPARED_HASHES[name]:
            raise ValueError(f"Qwen3-TTS compatibility patch identity mismatch: {name}")
        prepared[name] = value
    output.mkdir(parents=True, exist_ok=False)
    try:
        for name, value in prepared.items():
            path = output / "qwen_tts" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
        return verify_source(output / "qwen_tts")
    except BaseException:
        shutil.rmtree(output)
        raise
