"""Graph snapshot admission must use the published operator payload identities."""
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "src/core/rpu_graph_program_contracts.h"


def test_program_contract_table_matches_both_release_payloads():
    lock = json.loads((ROOT / "release/runtime-dependencies.json").read_text())
    expected = {lock["artifacts"][name]["sha256"]
                for name in ("rhino_ops_main", "rhino_ops_expansion")}
    rows = re.findall(r'\{"([a-f0-9]{64})", "([A-Za-z0-9_]+)", (\d+)\}', HEADER.read_text())
    assert {digest for digest, _, _ in rows} == expected, (
        "Re-audit and bind Graph program contracts when operator payloads change"
    )
    assert len(rows) == len({(digest, name) for digest, name, _ in rows})
    assert all(int(bits) == 3 for _, _, bits in rows)


def test_snapshot_admission_rejects_unreviewed_names_and_library_bytes(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = tmp_path / "admission.cpp"
    source.write_text('''#include "rpu_graph_program_contracts.h"
#include <cassert>
#include <string>
int main() {
    using namespace rpu_graph_program_contracts;
    for (const auto& entry : kAuditedPrograms) {
        assert(audited_contracts(entry.ref_sha256, entry.program_name) == entry.contracts);
        assert(audited_contracts(entry.ref_sha256, "unreviewed_program") == 0);
        std::string changed(entry.ref_sha256);
        changed[0] = changed[0] == '0' ? '1' : '0';
        assert(audited_contracts(changed, entry.program_name) == 0);
    }
    assert(audited_contracts("", "") == 0);
}
''')
    binary = tmp_path / "admission"
    subprocess.run([compiler, "-std=c++17", "-I", str(HEADER.parent),
                    str(source), "-o", str(binary)], check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
