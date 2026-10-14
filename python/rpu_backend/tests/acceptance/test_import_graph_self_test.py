import subprocess
import sys
import pathlib
REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / 'scripts' / 'check_import_graph_v5.py'

def test_self_test_mode_passes():
    result = subprocess.run([sys.executable, str(SCRIPT), '--self-test'], capture_output=True, text=True)
    assert result.returncode == 0, f'self-test FAILED:\nstdout: {result.stdout}\nstderr: {result.stderr}'

def test_full_mode_passes_on_current_tree():
    result = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, f'DAG check FAILED on current tree:\nstdout: {result.stdout}\nstderr: {result.stderr}'
