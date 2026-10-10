#!/usr/bin/env python3
"""Compatibility entry; current examples also use examples/run.py."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.common.legacy import main

if __name__ == "__main__":
    raise SystemExit(main("qwen3_vl"))
