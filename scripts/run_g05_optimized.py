"""Historical tool import/CLI adapter; implementation: examples/runners/g05/optimized.py."""
from importlib import import_module
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))
_engine = import_module("examples.runners.g05.optimized")
globals().update({name: value for name, value in vars(_engine).items()
                  if not name.startswith("__")})
