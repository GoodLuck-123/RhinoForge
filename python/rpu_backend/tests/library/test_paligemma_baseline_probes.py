import os
from pathlib import Path
import pytest

def _model_available() -> bool:
    from rpu_backend.model_registry import model_path
    return model_path('paligemma2-3b-pt-224').exists()

@pytest.mark.skipif(not _model_available(), reason='paligemma2-3b-pt-224 cache not present')
def test_paligemma_baseline_environment_record_contains_transformers_version():
    from rpu_backend.adapters.paligemma.generation import collect_baseline_environment
    record = collect_baseline_environment('paligemma2-3b-pt-224')
    assert record['transformers_version']
    assert record['model_name_or_path'] == 'paligemma2-3b-pt-224'
    assert record['model_repo'] == 'google/paligemma2-3b-pt-224'
    assert record['model_revision']
    assert record['resolved_path']
    assert record['local_snapshot_path']
    assert record['processor_repo'] == record['model_repo']
    assert record['tokenizer_repo'] == record['model_repo']
    assert record['processor_tokenizer_different_from_model'] is False
    assert record['paligemma_config']['model_type'] == 'paligemma'
    assert record['vision_config']['model_type'] == 'siglip_vision_model'
    assert record['text_config']['model_type'] == 'gemma2'

@pytest.mark.skipif(not _model_available(), reason='paligemma2-3b-pt-224 cache not present')
def test_paligemma_baseline_semantic_probes_pass():
    from rpu_backend.adapters.paligemma.generation import run_baseline_semantic_probes
    result = run_baseline_semantic_probes('paligemma2-3b-pt-224')
    assert result['image_feature_scale'] == 'sqrt(2304)'
    assert result['gemma2_embed_scale'] == 'sqrt(2304)'
    assert result['position_ids'] == 'cache_position+1'
    assert result['token_type_reverse'] == '1-token_type_ids'
    assert result['final_logit_softcap'] == 'recorded'
    assert result['paligemma_lm_head_softcap'] == 'not_applied'
