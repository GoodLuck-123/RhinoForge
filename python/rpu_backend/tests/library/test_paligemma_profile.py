from types import SimpleNamespace
import pytest
import torch
from rpu_backend.api.errors import RPUBackendError, UnsupportedModelError

class FakeWeight:

    def __init__(self, shape, dtype=torch.float16):
        self.shape = shape
        self.dtype = dtype

    def is_floating_point(self):
        is_floating = getattr(self.dtype, 'is_floating_point', None)
        if is_floating is not None:
            return bool(is_floating)
        return self.dtype in (torch.float16, torch.float32, torch.bfloat16)

class HFLikeModel:

    def __init__(self, embed_shape=(257216, 2304), lm_head_shape=(257216, 2304), lm_head_bias=None, named_parameters=()):
        self.model = SimpleNamespace(language_model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=FakeWeight(embed_shape))))
        self.lm_head = SimpleNamespace(weight=FakeWeight(lm_head_shape), bias=lm_head_bias)
        self._named_parameters = tuple(named_parameters)

    def named_parameters(self):
        return iter(self._named_parameters)

def _cfg(**overrides):
    text = SimpleNamespace(architectures=['Gemma2ForCausalLM'], model_type='gemma2', num_hidden_layers=26, hidden_size=2304, intermediate_size=9216, num_attention_heads=8, num_key_value_heads=4, head_dim=256, hidden_activation='gelu_pytorch_tanh', attention_bias=False, rms_norm_eps=1e-06, query_pre_attn_scalar=256, sliding_window=4096, cache_implementation='hybrid', attn_logit_softcapping=50.0, final_logit_softcapping=30.0, max_position_embeddings=8192, layer_types=['sliding_attention', 'full_attention'] * 13, use_bidirectional_attention=True, vocab_size=257216, num_image_tokens=256)
    vision = SimpleNamespace(model_type='siglip_vision_model', image_size=224, patch_size=14, num_positions=256, hidden_size=1152, intermediate_size=4304, num_attention_heads=16, num_hidden_layers=27, projection_dim=2304, vision_use_head=False)
    outer = SimpleNamespace(architectures=['PaliGemmaForConditionalGeneration'], model_type='paligemma', image_token_index=257152, projection_dim=2304, hidden_size=2048, vocab_size=257152, torch_dtype='bfloat16', text_config=text, vision_config=vision)
    for (key, value) in overrides.items():
        setattr(outer, key, value)
    return outer

def test_profile_accepts_first_version_tuple():
    from rpu_backend.adapters.paligemma.profile import validate_config_profile
    validate_config_profile(_cfg())

def test_profile_rejects_unsupported_resolution():
    from rpu_backend.adapters.paligemma.profile import validate_config_profile
    cfg = _cfg()
    cfg.vision_config.image_size = 448
    with pytest.raises(UnsupportedModelError, match='image_size'):
        validate_config_profile(cfg)

def test_profile_rejects_missing_bidirectional_attention():
    from rpu_backend.adapters.paligemma.profile import validate_config_profile
    cfg = _cfg()
    cfg.text_config.use_bidirectional_attention = False
    with pytest.raises(UnsupportedModelError, match='use_bidirectional_attention'):
        validate_config_profile(cfg)

def test_profile_accepts_image_token_id_without_image_token_index():
    from rpu_backend.adapters.paligemma.profile import validate_config_profile
    cfg = _cfg()
    delattr(cfg, 'image_token_index')
    cfg.image_token_id = 257152
    validate_config_profile(cfg)

def test_profile_accepts_absent_outer_hidden_size():
    from rpu_backend.adapters.paligemma.profile import validate_config_profile
    cfg = _cfg()
    delattr(cfg, 'hidden_size')
    validate_config_profile(cfg)

def test_profile_accepts_private_vocab_size_without_public_vocab_size():
    from rpu_backend.adapters.paligemma.profile import validate_config_profile
    cfg = _cfg()
    delattr(cfg, 'vocab_size')
    cfg._vocab_size = 257152
    validate_config_profile(cfg)

def test_profile_accepts_absent_outer_vocab_size():
    from rpu_backend.adapters.paligemma.profile import validate_config_profile
    cfg = _cfg()
    delattr(cfg, 'vocab_size')
    validate_config_profile(cfg)

def test_profile_accepts_absent_torch_dtype():
    from rpu_backend.adapters.paligemma.profile import validate_config_profile
    cfg = _cfg()
    delattr(cfg, 'torch_dtype')
    validate_config_profile(cfg)

def test_profile_accepts_derived_image_size():
    from rpu_backend.adapters.paligemma.profile import validate_config_profile
    cfg = _cfg()
    delattr(cfg.vision_config, 'image_size')
    validate_config_profile(cfg)

def test_profile_rejects_bad_private_vocab_size_even_with_good_public_vocab_size():
    from rpu_backend.adapters.paligemma.profile import validate_config_profile
    cfg = _cfg()
    cfg._vocab_size = 257151
    with pytest.raises(UnsupportedModelError, match='_vocab_size'):
        validate_config_profile(cfg)

def test_post_load_weight_checks_accept_real_hf_layout_shapes():
    from rpu_backend.adapters.paligemma.profile import validate_loaded_weights
    validate_loaded_weights(HFLikeModel())

def test_post_load_weight_checks_reject_bad_lm_head_shape():
    from rpu_backend.adapters.paligemma.profile import validate_loaded_weights
    model = HFLikeModel(lm_head_shape=(257152, 2304))
    with pytest.raises(RPUBackendError, match='lm_head.weight'):
        validate_loaded_weights(model)

def test_post_load_weight_checks_reject_lm_head_bias():
    from rpu_backend.adapters.paligemma.profile import validate_loaded_weights
    model = HFLikeModel(lm_head_bias=FakeWeight((257216,)))
    with pytest.raises(RPUBackendError, match='lm_head.bias'):
        validate_loaded_weights(model)

def test_post_load_weight_checks_reject_float_named_param_dtype():
    from rpu_backend.adapters.paligemma.profile import validate_loaded_weights
    model = HFLikeModel(named_parameters=(('model.language_model.layers.0.weight', FakeWeight((1,), torch.float32)),))
    with pytest.raises(RPUBackendError, match='layers.0.weight'):
        validate_loaded_weights(model)

def test_model_registry_has_paligemma2_logical_names():
    from rpu_backend.model_registry import MODELS
    assert MODELS['paligemma2-3b-pt-224'] == 'paligemma2-3b-pt-224'
    assert MODELS['paligemma2-3b-mix-224'] == 'paligemma2-3b-mix-224'
