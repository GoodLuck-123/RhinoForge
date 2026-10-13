import math
import pytest
import torch
from rpu_backend.api.errors import RPUBackendError
from rpu_backend.adapters.paligemma.profile import IMAGE_TOKEN_INDEX

def _valid_generate_kwargs() -> dict[str, torch.Tensor | int]:
    input_ids = torch.tensor([[1, 2, 3]])
    return {'pixel_values': torch.zeros(1, 3, 224, 224), 'input_ids': input_ids, 'attention_mask': torch.ones_like(input_ids), 'token_type_ids': torch.zeros_like(input_ids), 'max_new_tokens': 4, 'max_seq_len': 16}

def test_scale_image_features_divides_by_sqrt_hidden():
    from rpu_backend.adapters.paligemma.merge import scale_image_features
    x = torch.full((1, 2, 2304), 48.0, dtype=torch.float16)
    y = scale_image_features(x, hidden_size=2304)
    assert torch.allclose(y, torch.ones_like(y), atol=0, rtol=0)
    assert y.dtype == x.dtype
    assert y.device == x.device

def test_count_placeholders_returns_exact_count():
    from rpu_backend.adapters.paligemma.merge import _count_placeholders
    input_ids = torch.tensor([[IMAGE_TOKEN_INDEX, 7, IMAGE_TOKEN_INDEX], [8, IMAGE_TOKEN_INDEX, 9]])
    assert _count_placeholders(input_ids, IMAGE_TOKEN_INDEX) == 3

def test_merge_replaces_exact_placeholder_positions():
    from rpu_backend.adapters.paligemma.merge import merge_image_features
    input_ids = torch.tensor([[7, IMAGE_TOKEN_INDEX, IMAGE_TOKEN_INDEX, 9]])
    text = torch.zeros(1, 4, 3, dtype=torch.float16)
    image = torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.float16)
    merged = merge_image_features(input_ids=input_ids, inputs_embeds=text, image_features=image, image_token_index=IMAGE_TOKEN_INDEX, num_image_tokens=2)
    assert merged.tolist() == [[[0, 0, 0], [1, 2, 3], [4, 5, 6], [0, 0, 0]]]

def test_merge_supports_batch_when_each_row_has_expected_placeholders():
    from rpu_backend.adapters.paligemma.merge import merge_image_features
    input_ids = torch.tensor([[7, IMAGE_TOKEN_INDEX, IMAGE_TOKEN_INDEX, 9], [IMAGE_TOKEN_INDEX, 8, IMAGE_TOKEN_INDEX, 10]])
    text = torch.zeros(2, 4, 2, dtype=torch.float16)
    image = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float16)
    merged = merge_image_features(input_ids=input_ids, inputs_embeds=text, image_features=image, image_token_index=IMAGE_TOKEN_INDEX, num_image_tokens=2)
    assert merged.tolist() == [[[0, 0], [1, 2], [3, 4], [0, 0]], [[5, 6], [0, 0], [7, 8], [0, 0]]]

@pytest.mark.parametrize(('input_ids', 'inputs_embeds', 'image_features', 'match'), [(torch.tensor([IMAGE_TOKEN_INDEX]), torch.zeros(1, 1, 3), torch.zeros(1, 1, 3), 'input_ids.*rank 2'), (torch.tensor([[IMAGE_TOKEN_INDEX]]), torch.zeros(1, 3), torch.zeros(1, 1, 3), 'inputs_embeds.*rank 3'), (torch.tensor([[IMAGE_TOKEN_INDEX]]), torch.zeros(1, 1, 3), torch.zeros(1, 3), 'image_features.*rank 3')])
def test_merge_rejects_rank_mismatches(input_ids: torch.Tensor, inputs_embeds: torch.Tensor, image_features: torch.Tensor, match: str):
    from rpu_backend.adapters.paligemma.merge import merge_image_features
    with pytest.raises(RPUBackendError, match=match):
        merge_image_features(input_ids=input_ids, inputs_embeds=inputs_embeds, image_features=image_features, image_token_index=IMAGE_TOKEN_INDEX, num_image_tokens=1)

def test_merge_rejects_input_ids_inputs_embeds_shape_mismatch():
    from rpu_backend.adapters.paligemma.merge import merge_image_features
    with pytest.raises(RPUBackendError, match='input_ids.*inputs_embeds'):
        merge_image_features(input_ids=torch.tensor([[IMAGE_TOKEN_INDEX, IMAGE_TOKEN_INDEX]]), inputs_embeds=torch.zeros(1, 1, 3), image_features=torch.zeros(1, 2, 3), image_token_index=IMAGE_TOKEN_INDEX, num_image_tokens=2)

def test_merge_rejects_image_features_batch_mismatch():
    from rpu_backend.adapters.paligemma.merge import merge_image_features
    input_ids = torch.tensor([[IMAGE_TOKEN_INDEX, IMAGE_TOKEN_INDEX]])
    text = torch.zeros(1, 2, 3)
    image = torch.zeros(2, 2, 3)
    with pytest.raises(RPUBackendError, match='image_features.*batch'):
        merge_image_features(input_ids=input_ids, inputs_embeds=text, image_features=image, image_token_index=IMAGE_TOKEN_INDEX, num_image_tokens=2)

def test_merge_rejects_image_features_token_count_mismatch():
    from rpu_backend.adapters.paligemma.merge import merge_image_features
    input_ids = torch.tensor([[IMAGE_TOKEN_INDEX, IMAGE_TOKEN_INDEX]])
    text = torch.zeros(1, 2, 3)
    image = torch.zeros(1, 1, 3)
    with pytest.raises(RPUBackendError, match='image_features.*token'):
        merge_image_features(input_ids=input_ids, inputs_embeds=text, image_features=image, image_token_index=IMAGE_TOKEN_INDEX, num_image_tokens=2)

def test_merge_rejects_image_features_hidden_mismatch():
    from rpu_backend.adapters.paligemma.merge import merge_image_features
    input_ids = torch.tensor([[IMAGE_TOKEN_INDEX, IMAGE_TOKEN_INDEX]])
    text = torch.zeros(1, 2, 3)
    image = torch.zeros(1, 2, 4)
    with pytest.raises(RPUBackendError, match='image_features.*hidden'):
        merge_image_features(input_ids=input_ids, inputs_embeds=text, image_features=image, image_token_index=IMAGE_TOKEN_INDEX, num_image_tokens=2)

def test_merge_rejects_placeholder_count_mismatch():
    from rpu_backend.adapters.paligemma.merge import merge_image_features
    input_ids = torch.tensor([[IMAGE_TOKEN_INDEX]])
    text = torch.zeros(1, 1, 3, dtype=torch.float16)
    image = torch.zeros(1, 2, 3, dtype=torch.float16)
    with pytest.raises(RPUBackendError, match='placeholder'):
        merge_image_features(input_ids=input_ids, inputs_embeds=text, image_features=image, image_token_index=IMAGE_TOKEN_INDEX, num_image_tokens=2)

def test_merge_rejects_uneven_per_row_placeholder_counts():
    from rpu_backend.adapters.paligemma.merge import merge_image_features
    input_ids = torch.tensor([[IMAGE_TOKEN_INDEX, IMAGE_TOKEN_INDEX, IMAGE_TOKEN_INDEX, 7], [8, IMAGE_TOKEN_INDEX, 9, 10]])
    text = torch.zeros(2, 4, 3, dtype=torch.float16)
    image = torch.zeros(2, 2, 3, dtype=torch.float16)
    with pytest.raises(RPUBackendError, match='placeholder'):
        merge_image_features(input_ids=input_ids, inputs_embeds=text, image_features=image, image_token_index=IMAGE_TOKEN_INDEX, num_image_tokens=2)

def test_merge_does_not_mutate_inputs_embeds():
    from rpu_backend.adapters.paligemma.merge import merge_image_features
    input_ids = torch.tensor([[7, IMAGE_TOKEN_INDEX, 9]])
    text = torch.tensor([[[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]]])
    original = text.clone()
    image = torch.tensor([[[1.0, 2.0]]])
    merged = merge_image_features(input_ids=input_ids, inputs_embeds=text, image_features=image, image_token_index=IMAGE_TOKEN_INDEX, num_image_tokens=1)
    assert torch.equal(text, original)
    assert merged.tolist() == [[[10.0, 11.0], [1.0, 2.0], [14.0, 15.0]]]

def test_merge_aligns_image_features_dtype_and_device_to_inputs_embeds():
    from rpu_backend.adapters.paligemma.merge import merge_image_features
    input_ids = torch.tensor([[IMAGE_TOKEN_INDEX]])
    text = torch.zeros(1, 1, 2, dtype=torch.float32)
    image = torch.tensor([[[1.5, 2.5]]], dtype=torch.float16)
    merged = merge_image_features(input_ids=input_ids, inputs_embeds=text, image_features=image, image_token_index=IMAGE_TOKEN_INDEX, num_image_tokens=1)
    assert merged.dtype == text.dtype
    assert merged.device == text.device
    assert merged.tolist() == [[[1.5, 2.5]]]

def test_reverse_token_types_matches_hf_paligemma_contract():
    from rpu_backend.adapters.paligemma.merge import reverse_token_types
    token_type_ids = torch.tensor([[0, 0, 1, 1]])
    assert reverse_token_types(token_type_ids).tolist() == [[1, 1, 0, 0]]

def test_reverse_token_types_rejects_none():
    from rpu_backend.adapters.paligemma.merge import reverse_token_types
    with pytest.raises(RPUBackendError, match='token_type_ids'):
        reverse_token_types(None)

def test_position_ids_are_cache_position_plus_one():
    from rpu_backend.adapters.paligemma.merge import make_position_ids
    cache_position = torch.tensor([0, 1, 2, 3])
    assert make_position_ids(cache_position).tolist() == [[1, 2, 3, 4]]

def test_position_ids_reject_rank_mismatch():
    from rpu_backend.adapters.paligemma.merge import make_position_ids
    with pytest.raises(RPUBackendError, match='cache_position.*rank 1'):
        make_position_ids(torch.tensor([[0, 1]]))

def test_validate_generate_inputs_accepts_first_version_contract():
    from rpu_backend.adapters.paligemma.merge import validate_generate_inputs
    validate_generate_inputs(**_valid_generate_kwargs())

def test_validate_generate_inputs_rejects_missing_pixel_values():
    from rpu_backend.adapters.paligemma.merge import validate_generate_inputs
    kwargs = _valid_generate_kwargs()
    kwargs['pixel_values'] = None
    with pytest.raises(RPUBackendError, match='pixel_values'):
        validate_generate_inputs(**kwargs)

def test_validate_generate_inputs_rejects_missing_token_type_ids():
    from rpu_backend.adapters.paligemma.merge import validate_generate_inputs
    kwargs = _valid_generate_kwargs()
    kwargs['token_type_ids'] = None
    with pytest.raises(RPUBackendError, match='token_type_ids'):
        validate_generate_inputs(**kwargs)

def test_validate_generate_inputs_rejects_input_ids_rank_mismatch():
    from rpu_backend.adapters.paligemma.merge import validate_generate_inputs
    kwargs = _valid_generate_kwargs()
    kwargs['input_ids'] = torch.tensor([1, 2, 3])
    with pytest.raises(RPUBackendError, match='input_ids.*rank 2'):
        validate_generate_inputs(**kwargs)

def test_validate_generate_inputs_rejects_input_ids_batch_size_greater_than_one():
    from rpu_backend.adapters.paligemma.merge import validate_generate_inputs
    input_ids = torch.ones(2, 3, dtype=torch.long)
    kwargs = _valid_generate_kwargs()
    kwargs['input_ids'] = input_ids
    kwargs['attention_mask'] = torch.ones_like(input_ids)
    kwargs['token_type_ids'] = torch.zeros_like(input_ids)
    with pytest.raises(RPUBackendError, match='input_ids.*batch'):
        validate_generate_inputs(**kwargs)

@pytest.mark.parametrize('field', ['attention_mask', 'token_type_ids'])
def test_validate_generate_inputs_rejects_shape_mismatch(field: str):
    from rpu_backend.adapters.paligemma.merge import validate_generate_inputs
    kwargs = _valid_generate_kwargs()
    kwargs[field] = torch.ones(1, 2, dtype=torch.long)
    with pytest.raises(RPUBackendError, match=field):
        validate_generate_inputs(**kwargs)

def test_validate_generate_inputs_rejects_pixel_rank_mismatch():
    from rpu_backend.adapters.paligemma.merge import validate_generate_inputs
    kwargs = _valid_generate_kwargs()
    kwargs['pixel_values'] = torch.zeros(1, 3, 224)
    with pytest.raises(RPUBackendError, match='pixel_values.*rank 4'):
        validate_generate_inputs(**kwargs)

def test_validate_generate_inputs_rejects_pixel_batch_size_greater_than_one():
    from rpu_backend.adapters.paligemma.merge import validate_generate_inputs
    kwargs = _valid_generate_kwargs()
    kwargs['pixel_values'] = torch.zeros(2, 3, 224, 224)
    with pytest.raises(RPUBackendError, match='pixel_values.*batch'):
        validate_generate_inputs(**kwargs)

def test_validate_generate_inputs_rejects_max_seq_len_overflow():
    from rpu_backend.adapters.paligemma.merge import validate_generate_inputs
    kwargs = _valid_generate_kwargs()
    kwargs['max_new_tokens'] = 14
    with pytest.raises(RPUBackendError, match='max_seq_len'):
        validate_generate_inputs(**kwargs)

@pytest.mark.parametrize('max_new_tokens', [True, 1.5, '4', -1])
def test_validate_generate_inputs_rejects_invalid_max_new_tokens(max_new_tokens):
    from rpu_backend.adapters.paligemma.merge import validate_generate_inputs
    kwargs = _valid_generate_kwargs()
    kwargs['max_new_tokens'] = max_new_tokens
    with pytest.raises(RPUBackendError, match='max_new_tokens'):
        validate_generate_inputs(**kwargs)

def test_scale_constant_is_sqrt_2304():
    assert math.sqrt(2304) == 48.0
