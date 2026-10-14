#!/usr/bin/env python3
"""Run a model profile selected by its example configuration."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from examples.common.cli import main

if __name__ == "__main__":
    raise SystemExit(main(keep_logging_until_exit=True))
