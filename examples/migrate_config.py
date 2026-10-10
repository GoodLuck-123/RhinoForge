#!/usr/bin/env python3
"""Migrate an old example configuration without importing the device runtime."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.common.migration import main

if __name__ == "__main__":
    raise SystemExit(main())
