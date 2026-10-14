#!/usr/bin/env python3
"""Verify complete SHA-256 coverage for a local CI wheelhouse."""

from __future__ import annotations

import argparse
import hashlib
import re
import tempfile
from pathlib import Path, PurePosixPath


_LINE = re.compile(r"([0-9a-f]{64}) [ *](.+)")


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify(root: Path, *, expected_manifest: Path | None = None) -> int:
    if root.is_symlink():
        raise ValueError(f"wheelhouse root symlink is forbidden: {root}")
    root = root.resolve(strict=True)
    manifest = root / "SHA256SUMS"
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError(f"missing regular SHA256SUMS: {manifest}")
    if expected_manifest is not None:
        if expected_manifest.is_symlink() or not expected_manifest.is_file():
            raise ValueError(
                f"missing regular frozen wheelhouse manifest: {expected_manifest}"
            )
        if manifest.read_bytes() != expected_manifest.read_bytes():
            raise ValueError(
                "wheelhouse SHA256SUMS differs from frozen manifest: "
                f"{expected_manifest}"
            )

    expected: dict[str, str] = {}
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        match = _LINE.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid SHA256SUMS line {number}")
        digest, name = match.groups()
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or len(relative.parts) != 1
            or name == "SHA256SUMS"
        ):
            raise ValueError(f"unsafe SHA256SUMS path on line {number}: {name}")
        if name in expected:
            raise ValueError(f"duplicate SHA256SUMS entry: {name}")
        expected[name] = digest

    actual: set[str] = set()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"wheelhouse symlink is forbidden: {path.relative_to(root)}")
        if path.is_file() and path != manifest:
            actual.add(path.relative_to(root).as_posix())
    if not actual or not any(name.endswith(".whl") for name in actual):
        raise ValueError("wheelhouse contains no wheels")
    if set(expected) != actual:
        raise ValueError(
            "SHA256SUMS coverage mismatch: "
            f"unlisted={sorted(actual - set(expected))}, "
            f"missing={sorted(set(expected) - actual)}"
        )
    for name, digest in expected.items():
        if _digest(root / name) != digest:
            raise ValueError(f"SHA-256 mismatch: {name}")
    return len(actual)


def _self_test() -> None:
    with tempfile.TemporaryDirectory() as raw:
        base = Path(raw)
        root = base / "wheelhouse"
        root.mkdir()
        wheel = root / "demo-1-py3-none-any.whl"
        wheel.write_bytes(b"wheel")
        manifest = root / "SHA256SUMS"
        manifest.write_text(f"{_digest(wheel)}  {wheel.name}\n", encoding="utf-8")
        frozen_manifest = base / "wheelhouse.SHA256SUMS"
        frozen_manifest.write_bytes(manifest.read_bytes())
        assert verify(root, expected_manifest=frozen_manifest) == 1
        wheel.write_bytes(b"tampered")
        try:
            verify(root, expected_manifest=frozen_manifest)
        except ValueError as error:
            assert "mismatch" in str(error)
        else:
            raise AssertionError("tampered wheelhouse passed")
        manifest.write_text(f"{_digest(wheel)}  {wheel.name}\n", encoding="utf-8")
        try:
            verify(root, expected_manifest=frozen_manifest)
        except ValueError as error:
            assert "frozen manifest" in str(error)
        else:
            raise AssertionError("self-consistent but unfrozen wheelhouse passed")
        manifest.write_text(f"{'0' * 64}  ../escape.whl\n", encoding="utf-8")
        try:
            verify(root)
        except ValueError as error:
            assert "unsafe" in str(error)
        else:
            raise AssertionError("path traversal passed")
        link = root / "linked.whl"
        link.symlink_to(wheel)
        manifest.write_text(
            f"{_digest(wheel)}  {wheel.name}\n{'0' * 64}  {link.name}\n",
            encoding="utf-8",
        )
        try:
            verify(root)
        except ValueError as error:
            assert "symlink" in str(error)
        else:
            raise AssertionError("symlinked wheel passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path)
    parser.add_argument("--expected-manifest", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print("verify_wheelhouse self-test: PASS")
        return 0
    if args.root is None:
        parser.error("root is required unless --self-test is used")
    try:
        count = verify(args.root, expected_manifest=args.expected_manifest)
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))
    print(f"wheelhouse verified: {count} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
