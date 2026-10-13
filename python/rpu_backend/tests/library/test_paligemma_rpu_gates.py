import pytest
import torch
pytestmark = pytest.mark.skip(reason='PaliGemma2 is unsupported/numerical-blocked; retained only as a recertification gate source')

def _model_available(name: str='paligemma2-3b-pt-224') -> bool:
    from rpu_backend.model_registry import model_path
    return model_path(name).exists()

def _mse(left: torch.Tensor, right: torch.Tensor) -> float:
    left_f = left.detach().to('cpu', dtype=torch.float32)
    right_f = right.detach().to('cpu', dtype=torch.float32)
    return float(torch.mean((left_f - right_f) ** 2).item())

def _cos(left: torch.Tensor, right: torch.Tensor) -> float:
    left_f = left.detach().to('cpu', dtype=torch.float32).reshape(-1)
    right_f = right.detach().to('cpu', dtype=torch.float32).reshape(-1)
    if left_f.numel() == 0 or right_f.numel() == 0:
        return 1.0
    if float(left_f.norm().item()) == 0.0 and float(right_f.norm().item()) == 0.0:
        return 1.0
    return float(torch.nn.functional.cosine_similarity(left_f, right_f, dim=0).item())

def _topk_overlap(left: torch.Tensor, right: torch.Tensor, k: int) -> float:
    left_k = torch.topk(left.detach().to('cpu', dtype=torch.float32).reshape(-1), k).indices
    right_k = torch.topk(right.detach().to('cpu', dtype=torch.float32).reshape(-1), k).indices
    return len(set(left_k.tolist()) & set(right_k.tolist())) / float(k)

def _require_metric(result: dict, key: str, *, mse: float | None=None, cosine: float | None=None, top5: float | None=None, top10: float | None=None) -> None:
    metric = result[key]
    if mse is not None:
        assert metric['mse'] <= mse, (key, metric, result.get('metadata'))
    if cosine is not None:
        assert metric['cosine'] >= cosine, (key, metric, result.get('metadata'))
    if top5 is not None:
        assert metric['top5_overlap'] >= top5, (key, metric, result.get('metadata'))
    if top10 is not None:
        assert metric['top10_overlap'] >= top10, (key, metric, result.get('metadata'))

def _assert_no_disallowed_since(adapter, start: int) -> None:
    bad = [event.to_dict() for event in adapter.fallback_recorder.events[start:] if not event.allowed]
    assert not bad, bad

@pytest.mark.skipif(not _model_available(), reason='paligemma2-3b-pt-224 cache not present')
def test_paligemma_smoke_generate_four_tokens():
    from rpu_backend.api import PaliGemmaPolicy
    from rpu_backend.model_registry import model_path
    policy = PaliGemmaPolicy.from_pretrained(str(model_path('paligemma2-3b-pt-224')), dtype=torch.float16).to('rpu')
    sample = policy._adapter.build_smoke_sample()
    output = policy.generate(**sample, max_new_tokens=4, do_sample=False, num_beams=1)
    assert output.shape[0] == 1
    assert output.shape[1] >= sample['input_ids'].shape[1] + 4
    assert torch.isfinite(output.float()).all()
    policy._adapter.fallback_recorder.assert_no_disallowed()

@pytest.mark.skipif(not _model_available(), reason='paligemma2-3b-pt-224 cache not present')
def test_paligemma_numeric_gate_probe_tensors():
    from rpu_backend.api import PaliGemmaPolicy
    from rpu_backend.model_registry import model_path
    policy = PaliGemmaPolicy.from_pretrained(str(model_path('paligemma2-3b-pt-224')), dtype=torch.float16).to('rpu')
    sample = policy._adapter.build_smoke_sample()
    result = policy._adapter.run_numeric_probe(sample, max_new_tokens=4)
    _require_metric(result, 'scaled_image_features', mse=0.001, cosine=0.999)
    _require_metric(result, 'merged_inputs_embeds', mse=0.001, cosine=0.999)
    _require_metric(result, 'decode_next_token_embeddings', mse=0.001, cosine=0.999)
    _require_metric(result, 'prefill_last_hidden_state', mse=0.1, cosine=0.995)
    _require_metric(result, 'decode_final_token_hidden', mse=0.1, cosine=0.995)
    _require_metric(result, 'logits', mse=0.5, cosine=0.995, top5=0.8, top10=0.8)

@pytest.mark.skipif(not _model_available(), reason='paligemma2-3b-pt-224 cache not present')
def test_paligemma_token_and_fallback_gates():
    from rpu_backend.api import PaliGemmaPolicy
    from rpu_backend.model_registry import model_path
    policy = PaliGemmaPolicy.from_pretrained(str(model_path('paligemma2-3b-pt-224')), dtype=torch.float16).to('rpu')
    samples = [policy._adapter.build_smoke_sample(), policy._adapter.build_offset_placeholder_sample()]
    for sample in samples:
        start = len(policy._adapter.fallback_recorder.events)
        result = policy._adapter.run_token_probe(sample, min_new_tokens=4, max_new_tokens=8)
        assert result['all_tokens_match'] is True, result
        _assert_no_disallowed_since(policy._adapter, start)

@pytest.mark.skipif(not _model_available(), reason='paligemma2-3b-pt-224 cache not present')
def test_paligemma_mask_gate_matches_hf_mapping():
    from rpu_backend.api import PaliGemmaPolicy
    from rpu_backend.model_registry import model_path
    policy = PaliGemmaPolicy.from_pretrained(str(model_path('paligemma2-3b-pt-224')), dtype=torch.float16).to('rpu')
    samples = [policy._adapter.build_smoke_sample(), policy._adapter.build_offset_placeholder_sample()]
    for sample in samples:
        result = policy._adapter.run_mask_probe(sample)
        assert result['missing_token_type_ids_raises'] is True, result
        assert result['token_type_reverse'] == '1-token_type_ids', result
        assert result['full_attention_matches_hf'] is True, result
        assert result['sliding_attention_matches_hf'] is True, result
        assert result['placeholder_count_mismatch_raises'] is True, result

@pytest.mark.skipif(not _model_available(), reason='paligemma2-3b-pt-224 cache not present')
def test_paligemma_runtime_gate_reaches_rpu_once_and_resets_spm():
    from rpu_backend.api import PaliGemmaPolicy
    from rpu_backend.model_registry import model_path
    policy = PaliGemmaPolicy.from_pretrained(str(model_path('paligemma2-3b-pt-224')), dtype=torch.float16).to('rpu')
    sample = policy._adapter.build_smoke_sample()
    result = policy._adapter.run_runtime_probe(sample)
    assert result['siglip_reached_rpu'], result
    assert result['projector_reached_rpu'], result
    assert result['gemma2_reached_rpu'], result
    assert result['siglip_sentinel_set'], result
    assert result['projector_sentinel_set'], result
    assert result['gemma2_sentinel_set'], result
    assert result['model_swizzled_sentinel_set'], result
    assert result['second_to_rpu_no_double_swizzle'], result
    assert result['spm_boundaries_reset'], result

@pytest.mark.skipif(not _model_available('paligemma2-3b-mix-224'), reason='paligemma2-3b-mix-224 cache not present')
def test_paligemma_mix_224_candidate_gates_without_formal_token_claim():
    from rpu_backend.api import PaliGemmaPolicy
    from rpu_backend.model_registry import model_path
    policy = PaliGemmaPolicy.from_pretrained(str(model_path('paligemma2-3b-mix-224')), dtype=torch.float16).to('rpu')
    sample = policy._adapter.build_smoke_sample()
    start = len(policy._adapter.fallback_recorder.events)
    output = policy.generate(**sample, max_new_tokens=4, do_sample=False, num_beams=1)
    _assert_no_disallowed_since(policy._adapter, start)
    numeric = policy._adapter.run_numeric_probe(sample, max_new_tokens=4)
    runtime = policy._adapter.run_runtime_probe(sample)
    start = len(policy._adapter.fallback_recorder.events)
    token = policy._adapter.run_token_probe(sample, min_new_tokens=4, max_new_tokens=8)
    _assert_no_disallowed_since(policy._adapter, start)
    assert output.shape[0] == 1
    assert output.shape[1] >= sample['input_ids'].shape[1] + 4
    _require_metric(numeric, 'logits', cosine=0.995)
    assert runtime['siglip_reached_rpu'], runtime
    assert runtime['projector_reached_rpu'], runtime
    assert runtime['gemma2_reached_rpu'], runtime
    assert runtime['siglip_sentinel_set'], runtime
    assert runtime['projector_sentinel_set'], runtime
    assert runtime['gemma2_sentinel_set'], runtime
    assert runtime['model_swizzled_sentinel_set'], runtime
    assert runtime['second_to_rpu_no_double_swizzle'], runtime
    assert runtime['spm_boundaries_reset'], runtime
    assert {'all_tokens_match', 'cpu_tokens', 'rpu_tokens'} <= set(token)
