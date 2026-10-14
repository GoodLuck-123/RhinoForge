from __future__ import annotations
import pathlib
import re
import pytest
REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
CPP_ROOT = REPO_ROOT / 'src'
OWNING_TU = {'rpu_runtime_state.h', 'rpu_runtime_state.cpp'}
FORBIDDEN = {'g_chunk_size_override': re.compile('\\bg_chunk_size_override\\b'), 'get_chunk_size()': re.compile('\\bget_chunk_size\\s*\\('), 'set_chunk_size()': re.compile('\\bset_chunk_size\\s*\\(')}
_BLOCK_COMMENT = re.compile('/\\*.*?\\*/', re.DOTALL)
_LINE_COMMENT = re.compile('//[^\\n]*')

def _strip_comments(source: str) -> str:
    source = _BLOCK_COMMENT.sub(lambda m: '\n' * m.group(0).count('\n'), source)
    return _LINE_COMMENT.sub('', source)

def _scan(source: str):
    for (line_no, line) in enumerate(_strip_comments(source).splitlines(), 1):
        for (name, pattern) in FORBIDDEN.items():
            if pattern.search(line):
                yield (line_no, name)

def _offenders():
    found = []
    for path in sorted(CPP_ROOT.rglob('*')):
        if path.suffix not in ('.h', '.cpp') or path.name in OWNING_TU:
            continue
        for (line_no, name) in _scan(path.read_text(encoding='utf-8')):
            found.append(f'{path.relative_to(REPO_ROOT)}:{line_no}  {name}')
    return found

def test_the_scanner_catches_a_planted_reader():
    planted = 'void f() {\n    set_chunk_size_override(get_chunk_size());\n}\n'
    assert [n for (_, n) in _scan(planted)] == ['get_chunk_size()']

def test_the_scanner_does_not_fire_on_prose_or_the_per_handle_setter():
    benign = '#include "rpu_runtime_state.h"  // g_chunk_size_override lives here\n/* set_chunk_size(N) was the legacy route; see get_chunk_size(). */\n    set_chunk_size_override(0);\n    rpu_causal_decoder_set_chunk_size_cap(h, 256);\n'
    assert list(_scan(benign)) == []

def test_no_cpp_model_reads_the_process_global_chunk():
    offenders = _offenders()
    assert offenders == [], f'the process-global chunk authority still has {len(offenders)} live reference(s) outside {sorted(OWNING_TU)}:\n  ' + '\n  '.join(offenders)

def test_the_owning_tu_is_where_we_think_it_is():
    assert CPP_ROOT.is_dir(), f'{CPP_ROOT} missing — the scan covered no files'
    for name in OWNING_TU:
        matches = list(CPP_ROOT.rglob(name))
        assert len(matches) == 1, f'expected exactly one {name}, found {matches}'
if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
