#!/usr/bin/env python3
"""Initialize Hugging Face cache variables before third-party imports."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import site
import sys


def registry():
    path = Path(__file__).resolve().parents[1] / "python/rpu_backend/model_registry.py"
    spec = importlib.util.spec_from_file_location("_cache_registry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hf_root() -> str:
    return str(registry().cache_root() / ".hf")


def main() -> int:
    site_pkgs = Path(site.getsitepackages()[0])
    if not site_pkgs.is_dir():
        print(f"site-packages does not exist: {site_pkgs}", file=sys.stderr)
        return 1

    pth = site_pkgs / "rpu_backend_hf_env.pth"
    hf = _hf_root()
    hub = str(Path(hf) / "hub")
    pth.write_text(
        f"import os; os.environ['HF_HOME'] = {hf!r}; "
        f"os.environ['HF_HUB_CACHE'] = {hub!r}; "
        f"os.environ['HUGGINGFACE_HUB_CACHE'] = {hub!r}\n"
    )
    print(f"Installed {pth}")
    print(f"  HF_HOME       -> {hf}")
    print(f"  HF_HUB_CACHE  -> {hub}")
    print("Verify:")
    print(f"  {sys.executable} -c \"import os; print(os.environ.get('HF_HOME'))\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
