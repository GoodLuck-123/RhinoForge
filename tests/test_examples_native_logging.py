"""Native console capture uses real files; these children never import Torch."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap

from examples.common.native_logging import NativeCapture, _line_level


ROOT = Path(__file__).resolve().parents[1]


def _child(tmp_path, body):
    code = """
import ctypes, json, os, resource, sys
from pathlib import Path
from examples.common.native_logging import NativeCapture
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
directory = Path(sys.argv[1])
capture = NativeCapture(directory)
""" + textwrap.dedent(body)
    env = dict(os.environ, PYTHONPATH=str(ROOT), TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    return subprocess.run([sys.executable, "-c", code, str(tmp_path)], cwd=ROOT,
                          env=env, capture_output=True, text=True, timeout=15)


def test_native_gil_held_large_write_and_buffered_stdio_restore(tmp_path):
    result = _child(tmp_path, """
capture.start()
capture.start()
# PyDLL keeps the GIL throughout write(): a Python pipe-drain thread would
# block once the pipe filled. A seekable file has no such dependency.
libc = ctypes.PyDLL(None, use_errno=True)
libc.write.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
libc.write.restype = ctypes.c_ssize_t
payload = b'x' * (256 * 1024) + b'\\n'
assert libc.write(1, payload, len(payload)) == len(payload)
libc.printf.argtypes = [ctypes.c_char_p]
assert libc.printf(b'RpuInit 177: Rpu api version: test-sdk\\n') > 0
os.write(2, b'[WARN ] warning from native\\nunclassified native failure\\n')
capture.close()
capture.close()
lines = []
capture.replay(lambda level, line: lines.append([level, line]))
(directory / 'receipt.json').write_text(json.dumps(dict(
    paths={key: str(path) for key, path in capture.paths.items()}, lines=lines)))
assert 'torch' not in sys.modules and 'rpu_backend' not in sys.modules
os.write(1, b'original stdout restored\\n')
os.write(2, b'original stderr restored\\n')
""")
    assert result.returncode == 0, result.stderr
    assert result.stdout == "original stdout restored\n"
    assert result.stderr == "original stderr restored\n"
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["lines"] == [
        [logging.INFO, "x" * (256 * 1024)],
        [logging.INFO, "RpuInit 177: Rpu api version: test-sdk"],
        [logging.WARNING, "[WARN ] warning from native"],
        [logging.ERROR, "unclassified native failure"],
    ]
    assert Path(receipt["paths"]["stdout"]).read_bytes().endswith(b"test-sdk\n")
    assert Path(receipt["paths"]["stderr"]).read_bytes().endswith(b"native failure\n")


def test_native_error_classification_keeps_unlabelled_stderr_and_raw_bytes(tmp_path):
    result = _child(tmp_path, """
capture.start()
os.write(1, b'[INFO] ready\\n[DEBUG] detail\\n[ERROR] failed on stdout\\n')
os.write(2, b'2026-10-10 12:00:00 [WARN ] optional\\n'
            b'WARNING: optional too\\nERROR: real failure\\n[FATAL] allocation failed\\n'
            b'heap name: hx_mem_high_reserve, close fd: 7\\n'
            b'RpuInit 177: initialization failed\\n'
            b'bad utf8: \\xff\\nno newline')
capture.close()
lines = []
capture.replay(lambda level, line: lines.append([level, line]))
(directory / 'receipt.json').write_text(json.dumps(dict(
    paths={key: str(path) for key, path in capture.paths.items()}, lines=lines)))
""")
    assert result.returncode == 0, result.stderr
    assert not result.stdout and not result.stderr
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert [row[0] for row in receipt["lines"]] == [
        logging.INFO, logging.DEBUG, logging.ERROR, logging.WARNING,
        logging.WARNING, logging.ERROR, logging.CRITICAL, logging.INFO,
        logging.ERROR, logging.ERROR, logging.ERROR,
    ]
    assert receipt["lines"][-1][1] == "no newline"
    assert b"\xff" in Path(receipt["paths"]["stderr"]).read_bytes()


def test_native_exception_traceback_remains_on_restored_console(tmp_path):
    result = _child(tmp_path, """
try:
    capture.start()
    os.write(2, b'[ERROR] native context before failure\\n')
    raise RuntimeError('original inference failure')
finally:
    capture.close()
""")
    assert result.returncode != 0
    assert "RuntimeError: original inference failure" in result.stderr
    assert "native context before failure" not in result.stderr
    logs = list(tmp_path.glob("native-*.stderr.log"))
    assert len(logs) == 1
    assert logs[0].read_text() == "[ERROR] native context before failure\n"


def test_native_abort_keeps_error_without_a_reader_or_python_cleanup(tmp_path):
    result = _child(tmp_path, """
capture.start()
os.write(1, b'load checkpoint\\n')
os.write(2, b'[FATAL] allocation failed before abort\\n')
os.abort()
""")
    assert result.returncode == -signal.SIGABRT
    stdout, = tmp_path.glob("native-*.stdout.log")
    stderr, = tmp_path.glob("native-*.stderr.log")
    assert stdout.read_bytes() == b"load checkpoint\n"
    assert stderr.read_bytes() == b"[FATAL] allocation failed before abort\n"


def test_native_close_restores_descriptors_even_when_flush_fails(tmp_path):
    result = _child(tmp_path, """
import examples.common.native_logging as module
capture.start()
os.write(1, b'captured before flush failure\\n')
def fail():
    raise OSError('flush failed')
module._flush_output = fail
try:
    capture.close()
except OSError as error:
    assert str(error) == 'flush failed'
else:
    raise AssertionError('flush failure must remain visible')
capture.close()
os.write(1, b'stdout restored\\n')
os.write(2, b'stderr restored\\n')
""")
    assert result.returncode == 0, result.stderr
    assert result.stdout == "stdout restored\n"
    assert result.stderr == "stderr restored\n"


def test_native_partial_start_failure_restores_first_descriptor(tmp_path):
    result = _child(tmp_path, """
import examples.common.native_logging as module
original_dup2 = module.os.dup2
failed = False
def fail_second(source, destination):
    global failed
    if destination == 2 and not failed:
        failed = True
        raise OSError('stderr redirection failed')
    return original_dup2(source, destination)
module.os.dup2 = fail_second
try:
    capture.start()
except OSError as error:
    assert str(error) == 'stderr redirection failed'
else:
    raise AssertionError('start failure must remain visible')
capture.close()
os.write(1, b'stdout restored\\n')
os.write(2, b'stderr restored\\n')
""")
    assert result.returncode == 0, result.stderr
    assert result.stdout == "stdout restored\n"
    assert result.stderr == "stderr restored\n"


def test_native_paths_are_unique_and_only_known_startup_banners_are_info(tmp_path):
    first, second = NativeCapture(tmp_path), NativeCapture(tmp_path)
    assert set(first.paths.values()).isdisjoint(second.paths.values())
    assert not any(path.exists() for path in first.paths.values())
    assert _line_level("RpuInit 177: Rpu api version: release", stderr=True) == logging.INFO
    assert _line_level("RpuInit 177: failed", stderr=True) == logging.ERROR
    assert _line_level("heap name: xyz, close fd: 4", stderr=True) == logging.INFO
    assert _line_level("heap name: xyz, close fd: failed", stderr=True) == logging.ERROR


def test_native_capture_can_outlive_model_atexit_cleanup(tmp_path):
    result = _child(tmp_path, """
import atexit
capture.start()
def finish_logging():
    capture.close()
    os.write(1, b'logging finished after model cleanup\\n')
# Model imports register their cleanup after the logging finalizer.
atexit.register(finish_logging)
atexit.register(lambda: os.write(2, b'[INFO] model exit cleanup\\n'))
""")
    assert result.returncode == 0, result.stderr
    assert result.stdout == "logging finished after model cleanup\n"
    assert not result.stderr
    stderr, = tmp_path.glob("native-*.stderr.log")
    assert stderr.read_text() == "[INFO] model exit cleanup\n"
