"""Generate the native provider binding for standalone host compilation."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def runtime_identity_include(tmp_path, source_root=ROOT):
    output = tmp_path / "runtime-identity"
    driver = tmp_path / "runtime-identity.cmake"
    driver.write_text(
        f'include("{source_root.as_posix()}/release/version.cmake")\n'
        f'rhinoforge_generate_runtime_identity("{output.as_posix()}")\n'
    )
    result = subprocess.run(["cmake", "-P", str(driver)], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return output
