"""Capture native console writes without pipes or work in the inference loop.

The files intentionally survive normal exit and process abort.  They preserve
the original bytes; replay only controls what is shown on the console.  This is
process-wide CLI state, not a context for concurrent model sessions.
"""
from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path
import re
import sys
from typing import Callable
import uuid


_LEVELS = {
    "TRACE": logging.DEBUG, "DEBUG": logging.DEBUG, "INFO": logging.INFO,
    "WARN": logging.WARNING, "WARNING": logging.WARNING,
    "ERROR": logging.ERROR, "FATAL": logging.CRITICAL,
    "CRITICAL": logging.CRITICAL,
}
_LABEL = re.compile(
    r"\[\s*(TRACE|DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL|CRITICAL)\s*(?:\]|:)"
    r"|(?:^\s*|\]\s*)(TRACE|DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL|CRITICAL)(?:\s*:|\s+)",
    re.IGNORECASE,
)
_BANNER = re.compile(
    r"^RpuInit\s+\d+:\s+Rpu api version:"
    r"|^heap name:\s*[^,]+,\s*close fd:\s*\d+\s*$"
)


def _line_level(line: str, *, stderr: bool) -> int:
    if _BANNER.match(line):
        return logging.INFO
    match = _LABEL.search(line)
    if match:
        return _LEVELS[next(value for value in match.groups() if value).upper()]
    return logging.ERROR if stderr else logging.INFO


def _flush_output() -> None:
    """Flush Python and C stdio before changing their underlying descriptors."""
    for stream in (sys.stdout, sys.stderr):
        stream.flush()
    # RpuInit uses C stdio, whose stdout may remain buffered until process exit.
    # Keep backend imports and initialization outside this module.
    libc = ctypes.CDLL(None, use_errno=True)
    fflush = libc.fflush
    fflush.argtypes = [ctypes.c_void_p]
    fflush.restype = ctypes.c_int
    if fflush(None) != 0:
        number = ctypes.get_errno()
        raise OSError(number, "failed to flush native console output")


class NativeCapture:
    """Redirect fd 1/2 once, restore them, then replay retained native logs.

    Call ``start`` before native imports and ``close`` after model cleanup.
    The caller owns atexit ordering and writes its summaries through descriptors
    duplicated before this capture.  ``replay`` preserves order within each
    stream (stdout then stderr); it cannot reconstruct cross-stream chronology.
    """

    def __init__(self, directory: Path):
        directory = Path(directory).expanduser().resolve()
        name = f"native-{uuid.uuid4().hex}"
        self._paths = {
            "stdout": directory / f"{name}.stdout.log",
            "stderr": directory / f"{name}.stderr.log",
        }
        self._saved: dict[int, int] = {}
        self._started = False
        self._closed = False

    @property
    def paths(self) -> dict[str, Path]:
        return dict(self._paths)

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("native capture cannot restart after close")
        if self._started:
            return
        _flush_output()
        self._paths["stdout"].parent.mkdir(parents=True, exist_ok=True)
        targets = {}
        redirected = []
        try:
            # Prepare both files and both backups before redirecting either fd.
            for fd, stream in ((1, "stdout"), (2, "stderr")):
                targets[fd] = os.open(
                    self._paths[stream], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                self._saved[fd] = os.dup(fd)
            for fd in (1, 2):
                os.dup2(targets[fd], fd)
                redirected.append(fd)
            self._started = True
        except BaseException:
            for fd in reversed(redirected):
                os.dup2(self._saved[fd], fd)
            for backup in self._saved.values():
                os.close(backup)
            self._saved.clear()
            self._closed = True
            raise
        finally:
            for target in targets.values():
                os.close(target)

    def close(self) -> None:
        if not self._started or self._closed:
            return
        error = None
        try:
            _flush_output()
        except BaseException as exc:
            error = exc
        # Even a flush failure must not leave the console redirected.  Attempt
        # both descriptors, and retain any failed backup so a caller can retry.
        for fd, backup in tuple(self._saved.items()):
            try:
                os.dup2(backup, fd)
            except OSError as exc:
                if error is None:
                    error = exc
                elif hasattr(error, "add_note"):
                    error.add_note(f"could not restore console fd {fd}: {exc}")
            else:
                os.close(backup)
                del self._saved[fd]
        self._closed = not self._saved
        if error is not None:
            raise error

    def replay(self, callback: Callable[[int, str], None]) -> None:
        if not self._started or not self._closed:
            raise RuntimeError("native capture must be started and closed before replay")
        for stream, path in self._paths.items():
            with path.open(encoding="utf-8", errors="replace") as source:
                for line in source:
                    text = line.rstrip("\r\n")
                    callback(_line_level(text, stderr=stream == "stderr"), text)
