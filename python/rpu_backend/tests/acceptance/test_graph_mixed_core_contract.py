from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]

def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding='utf-8')

def test_segment_uses_max_core_domain_but_preserves_each_kernel_core_list():
    source = _read('src/graph/graph_runtime_execute.cpp')
    header = _read('src/graph/graph_runtime.h')
    assert 'else if (kn_num > seg_max_num_cores)' in source
    assert 'seg.core_ids.resize(n)' in source
    flat = ' '.join(source.split())
    assert 'rpu_add_kernel_mutable_checked(wq, *kernels_[kn.kernel_idx], kn.grid_dims, kn.core_ids,' in flat
    assert 'Kernel 可混用不同 core_ids' in header
    assert 'Queue_t 构造用的最大 core 执行域' in header

def test_checked_siglip_kernel_exception_remains_explicit():
    source = _read('src/graph/graph_runtime_execute.cpp')
    assert 'siglip_should_isolate_patch_embed_kernel( graph_label, kn, runtime_policy_.siglip_isolate_patch_embed)' in ' '.join(source.split())
    assert 'observed' in source[source.index('siglip_should_isolate_patch_embed_kernel') - 900:]

def test_porting_guides_do_not_repeat_removed_mixed_core_rule():
    english = _read('docs/planner_core_topology.md')
    chinese = _read('docs/planner_core_topology.zh.md')
    assert 'Different kernels in one graph segment may use different core counts' in english
    assert '同一 graph segment 内的 kernel 可以使用不同核数' in chinese
    assert '同一个 batch 中不能混用 core count' not in chinese

def test_architecture_examples_do_not_reintroduce_cpp_graph_ids():
    for path in ('docs/architecture.md', 'docs/architecture.zh.md'):
        text = _read(path)
        assert 'cfg.main_graph_id' not in text
        assert 'cfg.weights_graph_id' not in text
        assert 'GRAPH_CACHE.begin(' not in text

def test_warmup_is_documented_as_cold_configuration():
    for path in ('docs/runtime_config.md', 'docs/runtime_config.zh.md'):
        row = next(line for line in _read(path).splitlines() if line.startswith('| `RPU_WARMUP`'))
        assert '**MODEL**' in row
        assert 'construction' in row or '构造' in row
