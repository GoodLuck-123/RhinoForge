from __future__ import annotations
import os
import pytest
DECODER_HANDLE = '_rpu_decoder_handle'
VLM_DECODER_HANDLE = '_rpu_vlm_decoder_handle'

def test_handle_attribute_constants_are_neutral():
    from rpu_backend.runtime.hw_attrs import INTERNAL_HW_ATTRS_TRANSITIONAL
    for neutral in {DECODER_HANDLE, VLM_DECODER_HANDLE, '_rpu_action_handle', '_rpu_vision_handle'}:
        assert neutral in INTERNAL_HW_ATTRS_TRANSITIONAL, f'HND-05 regression: {neutral} missing from INTERNAL_HW_ATTRS_TRANSITIONAL'
    for old in {'_rpu_qwen3_handle', '_rpu_gemma_handle', '_rpu_adarms_handle', '_rpu_siglip_handle'}:
        assert old not in INTERNAL_HW_ATTRS_TRANSITIONAL, f'HND-05 regression: legacy {old} still in INTERNAL_HW_ATTRS_TRANSITIONAL'

def test_qwen3_handle_attribute_is_decoder_handle_in_writer():
    import ast
    src = open('python/rpu_backend/runtime/decoder.py', encoding='utf-8').read()
    tree = ast.parse(src)
    targets: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Attribute) and t.attr.startswith('_rpu_') and t.attr.endswith('_handle'):
                    targets.append(t.attr)
                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.attr.startswith('_rpu_') and t.attr.endswith('_handle'):
                    targets.append(t.attr)
    assert '_rpu_vlm_decoder_handle' not in targets, f'HND-02 regression: _rpu_vlm_decoder_handle leaked into Qwen3 writer runtime/decoder.py (Qwen3 must use _rpu_decoder_handle only; Pi05 internal Gemma owns _rpu_vlm_decoder_handle); writes seen: {sorted(set(targets))}'
    assert '_rpu_decoder_handle' in targets, f'HND-01 regression: _rpu_decoder_handle write missing from runtime/decoder.py (writes seen: {sorted(set(targets))})'
    assert '_rpu_qwen3_handle' not in targets, f'HND-01 regression: legacy _rpu_qwen3_handle write still present in runtime/decoder.py'

def test_pi05_internal_handle_attributes_use_neutral_names():
    import ast
    expected = {'python/rpu_backend/adapters/pi05/gemma.py': '_rpu_vlm_decoder_handle', 'python/rpu_backend/adapters/pi05/adarms.py': '_rpu_action_handle', 'python/rpu_backend/adapters/siglip.py': '_rpu_vision_handle'}
    legacy = {'_rpu_qwen3_handle', '_rpu_gemma_handle', '_rpu_adarms_handle', '_rpu_siglip_handle', '_rpu_decoder_handle'}
    for (path, neutral) in expected.items():
        src = open(path, encoding='utf-8').read()
        tree = ast.parse(src)
        seen: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Attribute) and t.attr.startswith('_rpu_') and t.attr.endswith('_handle'):
                        seen.add(t.attr)
            elif isinstance(node, ast.Attribute) and (node.attr.startswith('_rpu_') and node.attr.endswith('_handle')):
                seen.add(node.attr)
        assert neutral in seen, f'HND-02/03/04 regression: {path} missing {neutral} write/read (saw: {sorted(seen)})'
        for old in legacy - {neutral}:
            assert old not in seen, f'HND-02/03/04 regression: {path} contains legacy {old} (should be {neutral} per ADR §10 #12)'
