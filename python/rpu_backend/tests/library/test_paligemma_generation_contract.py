from types import SimpleNamespace
import pytest
import torch
from rpu_backend.api.errors import RPUBackendError

def test_unknown_generate_kwarg_is_rejected_with_name():
    from rpu_backend.adapters.paligemma.generation import validate_generate_kwargs
    with pytest.raises(ValueError, match='temperature'):
        validate_generate_kwargs({'max_new_tokens': 4, 'temperature': 0.0})

@pytest.mark.parametrize('max_new_tokens', [0, 9, True, 1.5, '4'])
def test_max_new_tokens_rejects_out_of_range_and_non_int(max_new_tokens):
    from rpu_backend.adapters.paligemma.generation import validate_generate_kwargs
    with pytest.raises(ValueError, match='max_new_tokens'):
        validate_generate_kwargs({'max_new_tokens': max_new_tokens})

@pytest.mark.parametrize('do_sample', [True, 0, None, 'False'])
def test_do_sample_must_be_exactly_false_when_provided(do_sample):
    from rpu_backend.adapters.paligemma.generation import validate_generate_kwargs
    with pytest.raises(ValueError, match='do_sample'):
        validate_generate_kwargs({'max_new_tokens': 4, 'do_sample': do_sample})

@pytest.mark.parametrize('num_beams', [2, True, '1'])
def test_num_beams_must_be_exactly_one_int(num_beams):
    from rpu_backend.adapters.paligemma.generation import validate_generate_kwargs
    with pytest.raises(ValueError, match='num_beams'):
        validate_generate_kwargs({'max_new_tokens': 4, 'num_beams': num_beams})

def test_use_cache_must_be_true_when_provided():
    from rpu_backend.adapters.paligemma.generation import validate_generate_kwargs
    with pytest.raises(ValueError, match='use_cache'):
        validate_generate_kwargs({'max_new_tokens': 4, 'use_cache': False})

@pytest.mark.parametrize('flag', ['return_dict_in_generate', 'output_scores', 'output_logits', 'output_attentions', 'output_hidden_states'])
def test_output_flags_must_be_false_when_provided(flag):
    from rpu_backend.adapters.paligemma.generation import validate_generate_kwargs
    with pytest.raises(ValueError, match=flag):
        validate_generate_kwargs({'max_new_tokens': 4, flag: True})

def test_defaults_are_greedy_short_decode():
    from rpu_backend.adapters.paligemma.generation import validate_generate_kwargs
    opts = validate_generate_kwargs({'max_new_tokens': 4})
    assert opts.max_new_tokens == 4
    assert opts.do_sample is False
    assert opts.num_beams == 1
    assert opts.use_cache is True
    assert opts.return_dict_in_generate is False

@pytest.mark.parametrize('stage', ['prefill', 'decode'])
def test_cpu_lm_head_fallback_records_softcap_metadata_without_applying(stage):
    from rpu_backend.adapters.paligemma.fallback import FallbackRecorder
    from rpu_backend.adapters.paligemma.generation import compute_logits_cpu_fallback
    calls = []

    class LMHead:

        def __call__(self, x):
            calls.append(x)
            assert x.device.type == 'cpu'
            assert x.requires_grad is False
            return torch.tensor([[[0.0, 2.0, 4.0]]], dtype=torch.float32)
    recorder = FallbackRecorder()
    hidden = torch.ones(1, 1, 2, dtype=torch.float16, requires_grad=True)
    logits = compute_logits_cpu_fallback(lm_head=LMHead(), hidden_states=hidden, recorder=recorder, stage=stage, final_logit_softcap=2.0)
    assert torch.equal(logits, torch.tensor([[[0.0, 2.0, 4.0]]]))
    assert len(calls) == 1
    assert calls[0].dtype == torch.float16
    event = recorder.events[0]
    assert event.allowed is True
    assert event.component == 'lm_head'
    assert event.op == 'cpu_lm_head'
    assert event.stage == stage
    assert event.reason == 'first-version logits fallback'
    assert event.shape == {'hidden_states': [1, 1, 2], 'final_logit_softcap': 2.0, 'applied_final_logit_softcap': False}
    assert event.dtype == 'torch.float16'
    assert event.device_from == 'cpu'
    assert event.device_to == 'cpu'
    assert event.bytes_moved == hidden.numel() * hidden.element_size()

def test_cpu_lm_head_fallback_applies_softcap_when_requested():
    from rpu_backend.adapters.paligemma.fallback import FallbackRecorder
    from rpu_backend.adapters.paligemma.generation import compute_logits_cpu_fallback
    recorder = FallbackRecorder()
    logits = compute_logits_cpu_fallback(lm_head=lambda x: torch.tensor([[[0.0, 2.0, 4.0]]]), hidden_states=torch.zeros(1, 1, 2, dtype=torch.float16), recorder=recorder, stage='decode', final_logit_softcap=2.0, apply_final_logit_softcap=True)
    expected = torch.tanh(torch.tensor([[[0.0, 1.0, 2.0]]])) * 2.0
    assert torch.allclose(logits, expected)
    assert recorder.events[0].shape['applied_final_logit_softcap'] is True

def test_cpu_lm_head_fallback_rejects_non_bool_softcap_apply_flag():
    from rpu_backend.adapters.paligemma.fallback import FallbackRecorder
    from rpu_backend.adapters.paligemma.generation import compute_logits_cpu_fallback
    with pytest.raises(ValueError, match='apply_final_logit_softcap'):
        compute_logits_cpu_fallback(lm_head=lambda x: x, hidden_states=torch.zeros(1, 1, 2, dtype=torch.float16), recorder=FallbackRecorder(), stage='decode', final_logit_softcap=2.0, apply_final_logit_softcap=1)

def test_cpu_lm_head_fallback_rejects_invalid_stage_before_recording():
    from rpu_backend.adapters.paligemma.fallback import FallbackRecorder
    from rpu_backend.adapters.paligemma.generation import compute_logits_cpu_fallback
    recorder = FallbackRecorder()
    hidden = torch.zeros(1, 1, 2, dtype=torch.float16)
    with pytest.raises(ValueError, match='stage'):
        compute_logits_cpu_fallback(lm_head=lambda x: x, hidden_states=hidden, recorder=recorder, stage='generate', final_logit_softcap=None)
    assert recorder.events == []

@pytest.mark.parametrize('softcap', [0, -1.0, True, '30', float('nan')])
def test_cpu_lm_head_fallback_rejects_invalid_softcap(softcap):
    from rpu_backend.adapters.paligemma.fallback import FallbackRecorder
    from rpu_backend.adapters.paligemma.generation import compute_logits_cpu_fallback
    with pytest.raises(ValueError, match='final_logit_softcap'):
        compute_logits_cpu_fallback(lm_head=lambda x: x, hidden_states=torch.zeros(1, 1, 2, dtype=torch.float16), recorder=FallbackRecorder(), stage='decode', final_logit_softcap=softcap)

def test_adapter_owns_fallback_recorder_and_requires_rpu_after_validation():
    from rpu_backend.adapters.paligemma import PaliGemmaAdapter
    from rpu_backend.adapters.paligemma.fallback import FallbackRecorder
    adapter = PaliGemmaAdapter(SimpleNamespace())
    assert isinstance(adapter.fallback_recorder, FallbackRecorder)
    with pytest.raises(RPUBackendError, match="\\.to\\('rpu'\\) first"):
        adapter.generate(max_new_tokens=4)

def test_adapter_rejects_unsupported_kwargs_before_readiness_check():
    from rpu_backend.adapters.paligemma import PaliGemmaAdapter
    adapter = PaliGemmaAdapter(SimpleNamespace())
    with pytest.raises(ValueError, match='temperature'):
        adapter.generate(max_new_tokens=4, temperature=0.0)

def test_adapter_accepts_public_inputs_before_readiness_check():
    from rpu_backend.adapters.paligemma import PaliGemmaAdapter
    adapter = PaliGemmaAdapter(SimpleNamespace())
    with pytest.raises(RPUBackendError, match="\\.to\\('rpu'\\) first"):
        adapter.generate(pixel_values=torch.zeros(1, 3, 224, 224), input_ids=torch.ones(1, 4, dtype=torch.long), attention_mask=torch.ones(1, 4, dtype=torch.long), token_type_ids=torch.zeros(1, 4, dtype=torch.long), max_new_tokens=4)

def test_adapter_generate_rejects_positional_args():
    from rpu_backend.adapters.paligemma import PaliGemmaAdapter
    adapter = PaliGemmaAdapter(SimpleNamespace(_rpu_swizzled=True))
    with pytest.raises(ValueError, match='keyword inputs only'):
        adapter.generate(torch.zeros(1), max_new_tokens=1)

def test_additive_prefill_mask_shape_and_image_block_visibility():
    from rpu_backend.adapters.paligemma.generation import build_prefill_additive_attention_mask
    mask = build_prefill_additive_attention_mask(attention_mask=torch.tensor([[1, 1, 1, 1]], dtype=torch.long), token_type_ids=torch.tensor([[0, 0, 1, 1]], dtype=torch.long), dtype=torch.float16)
    assert tuple(mask.shape) == (1, 1, 4, 4)
    assert mask.dtype == torch.float16
    assert mask[0, 0, 0, 1] == 0
    assert mask[0, 0, 1, 0] == 0
    assert torch.isneginf(mask[0, 0, 2, 3])
    assert mask[0, 0, 3, 2] == 0

def test_additive_decode_mask_shape_respects_padding_keys():
    from rpu_backend.adapters.paligemma.generation import build_decode_additive_attention_mask
    mask = build_decode_additive_attention_mask(attention_mask=torch.tensor([[1, 0, 1]], dtype=torch.long), key_value_len=3, dtype=torch.float16)
    assert tuple(mask.shape) == (1, 1, 1, 3)
    assert mask[0, 0, 0, 0] == 0
    assert torch.isneginf(mask[0, 0, 0, 1])
    assert mask[0, 0, 0, 2] == 0

def test_additive_mask_helpers_reject_bad_shapes():
    from rpu_backend.adapters.paligemma.generation import build_decode_additive_attention_mask, build_prefill_additive_attention_mask
    with pytest.raises(RPUBackendError, match='rank 2'):
        build_prefill_additive_attention_mask(attention_mask=torch.ones(1, 2, 3), token_type_ids=torch.ones(1, 2, dtype=torch.long))
    with pytest.raises(RPUBackendError, match='key_value_len'):
        build_decode_additive_attention_mask(attention_mask=torch.ones(1, 2, dtype=torch.long), key_value_len=3)

def test_adapter_smoke_sample_has_one_image_and_one_text_token():
    from rpu_backend.adapters.paligemma import PaliGemmaAdapter
    from rpu_backend.adapters.paligemma.profile import IMAGE_TOKEN_INDEX, NUM_IMAGE_TOKENS
    sample = PaliGemmaAdapter(SimpleNamespace()).build_smoke_sample()
    assert sample['pixel_values'].shape == (1, 3, 224, 224)
    assert sample['pixel_values'].dtype == torch.float16
    assert sample['input_ids'].shape == (1, NUM_IMAGE_TOKENS + 1)
    assert sample['attention_mask'].shape == sample['input_ids'].shape
    assert sample['token_type_ids'].shape == sample['input_ids'].shape
    assert int((sample['input_ids'] == IMAGE_TOKEN_INDEX).sum().item()) == NUM_IMAGE_TOKENS
    assert sample['input_ids'][0, -1].item() != IMAGE_TOKEN_INDEX
    assert sample['token_type_ids'][0, :NUM_IMAGE_TOKENS].tolist() == [0] * NUM_IMAGE_TOKENS
    assert sample['token_type_ids'][0, -1].item() == 1

def test_adapter_generate_strips_model_inputs_before_kwarg_validation(monkeypatch):
    import rpu_backend.adapters.paligemma as paligemma
    seen = {}

    def fake_validate(kwargs):
        seen.update(kwargs)
        raise ValueError('stop after validation')
    monkeypatch.setattr('rpu_backend.adapters.paligemma.generation.validate_generate_kwargs', fake_validate)
    adapter = paligemma.PaliGemmaAdapter(SimpleNamespace())
    with pytest.raises(ValueError, match='stop after validation'):
        adapter.generate(pixel_values=torch.zeros(1, 3, 224, 224), input_ids=torch.ones(1, 257, dtype=torch.long), attention_mask=torch.ones(1, 257, dtype=torch.long), token_type_ids=torch.zeros(1, 257, dtype=torch.long), max_new_tokens=1)
    assert seen == {'max_new_tokens': 1}

def test_rpu_siglip_fused_output_skips_second_projector_call():
    from rpu_backend.adapters.paligemma import _image_features_from_vision_output

    class Projector:
        linear = SimpleNamespace(in_features=1152)

        def __init__(self):
            self.called = False

        def __call__(self, last_hidden_state):
            self.called = True
            raise AssertionError('projector should not be called twice')
    hidden = torch.ones(1, 256, 2304, dtype=torch.float16)
    projector = Projector()
    result = _image_features_from_vision_output(vision=SimpleNamespace(vision_model=SimpleNamespace(_rpu_vision_handle=1)), image_outputs=SimpleNamespace(last_hidden_state=hidden), projector=projector, decoder=SimpleNamespace(config=SimpleNamespace(hidden_size=2304)))
    assert result is hidden
    assert projector.called is False

def test_cpu_siglip_output_still_uses_projector():
    from rpu_backend.adapters.paligemma import _image_features_from_vision_output

    class Projector:
        linear = SimpleNamespace(in_features=1152)

        def __init__(self):
            self.called_with = None

        def __call__(self, last_hidden_state):
            self.called_with = last_hidden_state
            return torch.ones(1, 256, 2304, dtype=torch.float16)
    hidden = torch.ones(1, 256, 1152, dtype=torch.float16)
    projector = Projector()
    result = _image_features_from_vision_output(vision=SimpleNamespace(), image_outputs=SimpleNamespace(last_hidden_state=hidden), projector=projector, decoder=SimpleNamespace(config=SimpleNamespace(hidden_size=2304)))
    assert projector.called_with is hidden
    assert tuple(result.shape) == (1, 256, 2304)

def test_adapter_ready_generate_runs_fake_greedy_path_and_records_fallbacks(monkeypatch):
    import rpu_backend.adapters.paligemma.generation as generation
    from rpu_backend.adapters.paligemma import PaliGemmaAdapter
    from rpu_backend.adapters.paligemma.profile import IMAGE_TOKEN_INDEX, NUM_IMAGE_TOKENS
    monkeypatch.setattr(generation, '_move_tensor', lambda tensor, **kwargs: tensor.to(dtype=kwargs.get('dtype', tensor.dtype)))
    hidden_size = 4
    calls = []

    class Decoder:
        config = SimpleNamespace(hidden_size=hidden_size)

        def embed_tokens(self, input_ids):
            data = input_ids.to(torch.float32).unsqueeze(-1)
            return data.expand(*input_ids.shape, hidden_size).to(torch.float16)

        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(last_hidden_state=kwargs['inputs_embeds'].to(torch.float16), past_key_values=SimpleNamespace(position=len(calls)))

    class Vision:

        def __call__(self, pixel_values, return_dict=True):
            return SimpleNamespace(last_hidden_state=torch.ones(1, NUM_IMAGE_TOKENS, hidden_size))

    class Projector:

        def __call__(self, last_hidden_state):
            return torch.ones(1, NUM_IMAGE_TOKENS, hidden_size, dtype=torch.float16)

    class LMHead:

        def __init__(self):
            self.calls = 0

        def __call__(self, hidden_states):
            self.calls += 1
            logits = torch.zeros(hidden_states.shape[0], hidden_states.shape[1], 8)
            logits[..., self.calls + 2] = 1.0
            return logits
    lm_head = LMHead()
    decoder = Decoder()
    model = SimpleNamespace(_rpu_swizzled=True, vision_tower=Vision(), multi_modal_projector=Projector(), language_model=SimpleNamespace(model=decoder, lm_head=lm_head), config=SimpleNamespace(text_config=SimpleNamespace(final_logit_softcapping=30.0)))
    adapter = PaliGemmaAdapter(model)
    adapter._rpu_is_ready = True
    input_ids = torch.tensor([[IMAGE_TOKEN_INDEX] * NUM_IMAGE_TOKENS + [2]], dtype=torch.long)
    out = adapter.generate(pixel_values=torch.zeros(1, 3, 224, 224, dtype=torch.float16), input_ids=input_ids, attention_mask=torch.ones_like(input_ids), token_type_ids=torch.tensor([[0] * NUM_IMAGE_TOKENS + [1]], dtype=torch.long), max_new_tokens=2)
    assert out.shape == (1, NUM_IMAGE_TOKENS + 3)
    assert out[0, -2:].tolist() == [3, 4]
    assert calls[0]['use_cache'] is True
    assert tuple(calls[0]['attention_mask'].shape[-2:]) == (NUM_IMAGE_TOKENS + 1, NUM_IMAGE_TOKENS + 1)
    assert tuple(calls[1]['attention_mask'].shape[-2:]) == (1, NUM_IMAGE_TOKENS + 2)
    event_keys = {(event.component, event.op, event.stage) for event in adapter.fallback_recorder.events}
    assert ('paligemma_outer', 'placeholder_mask_creation', 'prefill') in event_keys
    assert ('paligemma_outer', 'masked_scatter_merge', 'prefill') in event_keys
    assert ('paligemma_outer', 'attention_mask_mapping', 'prefill') in event_keys
    assert ('paligemma_outer', 'attention_mask_mapping', 'decode') in event_keys
    assert ('gemma2', 'cpu_embed_tokens', 'decode') in event_keys
    assert ('lm_head', 'cpu_lm_head', 'prefill') in event_keys
    assert ('lm_head', 'cpu_lm_head', 'decode') in event_keys
    lm_head_events = [event for event in adapter.fallback_recorder.events if event.component == 'lm_head']
    assert all((event.shape['applied_final_logit_softcap'] is False for event in lm_head_events))
    adapter.fallback_recorder.assert_no_disallowed()

def test_generate_documents_top_level_lm_head_softcap_is_not_applied():
    from pathlib import Path
    source = Path('python/rpu_backend/adapters/paligemma/__init__.py').read_text()
    assert 'HF PaliGemma top-level lm_head parity' in source
    assert 'apply_final_logit_softcap=False' in source

def test_baseline_probe_inspects_actual_paligemma_softcap_and_token_type_paths():
    from pathlib import Path
    source = Path('python/rpu_backend/adapters/paligemma/generation.py').read_text()
    assert 'create_causal_mask_mapping' in source
    assert 'token_type_ids=1-token_type_ids' in source
    assert 'PaliGemmaForConditionalGeneration.forward' in source
    assert 'paligemma_lm_head_softcap' in source
    assert 'self.lm_head(hidden_states' in source
