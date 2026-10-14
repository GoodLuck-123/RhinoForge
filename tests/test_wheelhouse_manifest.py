"""Verify the wheelhouse manifest remains bound to the expected source."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts/ci/verify_wheelhouse.py"
_spec = importlib.util.spec_from_file_location("verify_wheelhouse", _PATH)
verify_wheelhouse = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify_wheelhouse)


def test_wheelhouse_manifest_is_bound_to_frozen_source(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    wheel = wheelhouse / "demo-1-py3-none-any.whl"
    wheel.write_bytes(b"original")
    manifest = wheelhouse / "SHA256SUMS"
    manifest.write_text(
        f"{hashlib.sha256(wheel.read_bytes()).hexdigest()}  {wheel.name}\n",
        encoding="utf-8",
    )
    frozen = tmp_path / "wheelhouse.SHA256SUMS"
    frozen.write_bytes(manifest.read_bytes())
    assert verify_wheelhouse.verify(wheelhouse, expected_manifest=frozen) == 1

    wheel.write_bytes(b"replacement")
    manifest.write_text(
        f"{hashlib.sha256(wheel.read_bytes()).hexdigest()}  {wheel.name}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="frozen manifest"):
        verify_wheelhouse.verify(wheelhouse, expected_manifest=frozen)
