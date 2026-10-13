import subprocess
import sys
import textwrap
from pathlib import Path
from rpu_backend.tests.acceptance import COLD_IMPORT_TIMEOUT_S, timeout_message
_PYTHON_ROOT = Path(__file__).resolve().parents[4] / 'python'
_PREAMBLE = textwrap.dedent(f'\n    import sys\n    sys.meta_path[:] = [\n        finder for finder in sys.meta_path\n        if not type(finder).__module__.startswith("_rpu_backend_editable")\n    ]\n    sys.path.insert(0, {str(_PYTHON_ROOT)!r})\n    sys.modules["lerobot"] = None  # block all downstream `import lerobot.*`\n').lstrip()

def test_import_rpu_backend_with_lerobot_missing_survives():
    code = _PREAMBLE + textwrap.dedent('\n        import sys\n        import rpu_backend  # must NOT raise\n        print("OK: rpu_backend imported cleanly with lerobot blocked")\n    ')
    try:
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=COLD_IMPORT_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(timeout_message('REG-06 leg 1', exc)) from exc
    assert result.returncode == 0, f'REG-06 leg 1: subprocess exited with code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}'
    assert 'OK: rpu_backend imported cleanly' in result.stdout, f'Success sentinel missing:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}'
    assert 'Traceback (most recent call last):' not in result.stderr, f'Uncaught traceback on stderr — leg 1 resilience contract failed:\n{result.stderr}'

def test_pi05_from_pretrained_without_lerobot_raises_actionable():
    code = _PREAMBLE + textwrap.dedent('\n        import sys\n        import rpu_backend\n        from rpu_backend.api import Pi05Policy   # codex R1 HIGH-1: post-CUT-04 canonical surface\n        try:\n            Pi05Policy.from_pretrained("/nonexistent/path/for/testing")\n        except ImportError as e:\n            msg = str(e)\n            assert "pip install lerobot" in msg, (\n                f"REG-06 leg 2: actionable install hint missing: {msg!r}"\n            )\n            print("OK: actionable ImportError raised:", msg[:120].replace(chr(10), \' | \'))\n            sys.exit(0)\n        except Exception as e:\n            print(f"FAIL: unexpected exception {type(e).__name__}: {e}")\n            sys.exit(1)\n        else:\n            print("FAIL: no exception raised from Pi05Policy.from_pretrained")\n            sys.exit(1)\n    ')
    try:
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=COLD_IMPORT_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(timeout_message('REG-06 leg 2', exc)) from exc
    assert result.returncode == 0, f'REG-06 leg 2: subprocess exited with code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}'
    assert 'OK: actionable ImportError raised' in result.stdout, f'Success sentinel missing:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}'

def main() -> int:
    test_import_rpu_backend_with_lerobot_missing_survives()
    test_pi05_from_pretrained_without_lerobot_raises_actionable()
    print('PASS: REG-06 — import-resilience + actionable Pi0.5 error (2 legs)')
    return 0
if __name__ == '__main__':
    sys.exit(main())
