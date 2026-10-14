import sys
import pytest
_SUBMODULE_ALLOWLIST = {'runtime', 'adapters', 'models', 'tests', 'model_registry', 'api', 'graph', 'quant', 'debug'}
_HIDDEN_GRAPH_COMPAT = {'Graph', 'compile', 'dynamo_backend'}

def test_public_api_whitelist():
    import rpu_backend
    public = set((x for x in dir(rpu_backend) if not x.startswith('_') or x == '__version__'))
    expected = {'RPUCache', 'RPUModelForCausalLM', '__version__', 'reset_graph_cache'}
    assert public == expected, f'dir(rpu_backend) drift:\n  extra   = {sorted(public - expected)}\n  missing = {sorted(expected - public)}'

def test_strict_dict_contains_only_public_names():
    import rpu_backend
    import rpu_backend.quant
    extras = {name for name in vars(rpu_backend) if not name.startswith('_')} - _SUBMODULE_ALLOWLIST - _HIDDEN_GRAPH_COMPAT
    expected = {'RPUCache', 'RPUModelForCausalLM', 'reset_graph_cache'}
    assert extras == expected, f'H2/NH1 reviews iter-2/iter-3: unexpected names in rpu_backend.__dict__ (PEP 562 __getattr__ bypassed):\n  extra   = {sorted(extras - expected)}\n  missing = {sorted(expected - extras)}'
    assert '__version__' in vars(rpu_backend), '__version__ missing from __dict__'
    forbidden_canary_names = {'torch', 'os', 'sys', 'warnings', 'importlib', 'atexit', 'types', 'set_debug', 'set_profile', 'set_chunk_size', 'set_spm_mode', 'rpu_backend'}
    leaked_canary = forbidden_canary_names & set(vars(rpu_backend).keys())
    assert not leaked_canary, f'NH1 iter-3 violation: canary names leaked into rpu_backend.__dict__ (Step 1.7 requires `_`-prefixed aliases for all module-scope imports):\n  leaked = {sorted(leaked_canary)}'

def test_documented_graph_namespace_uses_production_wrappers():
    import rpu_backend
    from rpu_backend import graph
    aliases = {'GraphCache': graph.GraphCache, 'GraphSignature': graph.GraphSignature, 'get_default_graph_cache': graph.get_default_graph_cache}
    for (name, canonical) in aliases.items():
        assert name not in vars(rpu_backend)
        with pytest.warns(DeprecationWarning, match='removed in API 6.0'):
            assert getattr(rpu_backend, name) is canonical
    for method in ('freeze', 'begin_warmup', 'debug_bucket_counts', 'explain_miss'):
        assert hasattr(graph.GraphCache, method)

def main() -> int:
    test_public_api_whitelist()
    test_strict_dict_contains_only_public_names()
    print('PASS: public API + __dict__ strict (H2) [v5-01b coverage migrated to test_v5_top_level_cutover.py CUT-01a]')
    return 0
if __name__ == '__main__':
    sys.exit(main())
