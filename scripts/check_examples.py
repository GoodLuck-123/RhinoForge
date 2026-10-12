#!/usr/bin/env python3
"""Validate public templates, compatibility paths and their configuration index."""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.common.config import load_config
from examples.common.legacy import DEFAULTS, PATH_ALIASES, resolve_config_path

CONFIGS = ROOT / "examples/configs"


def load_configs() -> dict[Path, dict]:
    configs = {}
    for path in sorted(CONFIGS.rglob("*.toml")):
        try:
            configs[path] = load_config(path)
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid template {path.relative_to(CONFIGS)}: {exc}") from exc
    if not configs:
        raise ValueError("no example configurations found")
    return configs


def check_aliases(configs: dict[Path, dict]) -> None:
    names = json.loads((CONFIGS / "config_names.json").read_text(encoding="utf-8"))
    for old, new in names.items():
        prefix = "examples/configs/"
        if (not old.startswith(prefix) or not new.startswith(prefix)
                or ".." in Path(old).parts or ".." in Path(new).parts
                or PATH_ALIASES.get(old.removeprefix(prefix)) != new.removeprefix(prefix)):
            raise ValueError(f"invalid configuration rename: {old}")
    for old, new in PATH_ALIASES.items():
        source, target = CONFIGS / old, CONFIGS / new
        if source.exists() or source.is_symlink():
            raise ValueError(f"duplicate compatibility template: {old}")
        if new in PATH_ALIASES or target not in configs:
            raise ValueError(f"alias does not name a canonical template: {old}")
        if resolve_config_path(source) != target:
            raise ValueError(f"configuration alias does not resolve: {old}")
    for entry, path in DEFAULTS.items():
        if path is not None and CONFIGS / path not in configs:
            raise ValueError(f"missing default template for {entry}: {path}")


def check_index(configs: dict[Path, dict]) -> None:
    index = (CONFIGS / "CONFIGURATIONS.md").read_text(encoding="utf-8")
    links = re.findall(r"\]\(([^\s)]+\.toml)(?:#[^)]*)?\)", index)
    linked = {(CONFIGS / path).resolve() for path in links}
    actual = {path.resolve() for path in configs}
    if linked != actual:
        missing = sorted(str(path.relative_to(CONFIGS)) for path in actual - linked)
        extra = sorted(str(path) for path in linked - actual)
        raise ValueError(f"configuration index drift: missing={missing}, extra={extra}")


def main() -> int:
    try:
        configs = load_configs()
        check_aliases(configs)
        check_index(configs)
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"PASS: {len(configs)} templates and {len(PATH_ALIASES)} compatibility paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
