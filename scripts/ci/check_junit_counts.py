#!/usr/bin/env python3
"""Assert a JUnit report's counts, because pytest's exit code does not."""
import glob
import sys
import xml.etree.ElementTree as ET
from collections import Counter


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    pattern, want = argv[1], dict(kv.split("=", 1) for kv in argv[2:])
    unique = want.pop("unique", "0")
    if unique not in {"0", "1"}:
        print("FAIL: unique must be 0 or 1")
        return 2
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"FAIL: no JUnit reports match {pattern}")
        return 1
    cases = []
    for path in paths:
        try:
            cases.extend(ET.parse(path).getroot().iter("testcase"))
        except (OSError, ET.ParseError) as exc:
            print(f"FAIL: cannot read {path}: {exc}")
            return 1

    def kind(tc):
        classname = tc.get("classname") or ""
        if classname.endswith(".legs"):
            return "leg"
        if classname.endswith(".meta"):
            return "meta"
        return "base"
    totals = {
        "tests": len(cases),
        "failures": sum(tc.find("failure") is not None for tc in cases),
        "errors": sum(tc.find("error") is not None for tc in cases),
        "skipped": sum(tc.find("skipped") is not None for tc in cases),
        "base": sum(kind(tc) == "base" for tc in cases),
        "executed_legs": sum(
            kind(tc) == "leg" and tc.find("skipped") is None for tc in cases
        ),
        "skipped_legs": sum(
            kind(tc) == "leg" and tc.find("skipped") is not None for tc in cases
        ),
    }
    unknown = set(want) - set(totals)
    if unknown:
        print(f"FAIL: unknown count(s): {', '.join(sorted(unknown))}")
        return 2
    if unique == "1":
        identities = [((tc.get("classname") or ""), (tc.get("name") or ""))
                      for tc in cases]
        duplicates = sorted(ident for ident, count in Counter(identities).items()
                            if count > 1)
        if duplicates:
            for classname, name in duplicates:
                print(f"FAIL: duplicate testcase: {classname}::{name}")
            return 1
    got = {k: totals[k] for k in want}
    print(f"{pattern}: {got}")
    bad = {k: (got[k], int(v)) for k, v in want.items() if got[k] != int(v)}
    if bad:
        for k, (g, w) in bad.items():
            print(f"FAIL: {k}={g}, expected {w}")
        return 1
    names = [tc.get("name") for tc in cases]
    shown = ", ".join(names[:20])
    if len(names) > 20:
        shown += f", ... (+{len(names) - 20})"
    print(f"OK: {len(names)} cases — {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
