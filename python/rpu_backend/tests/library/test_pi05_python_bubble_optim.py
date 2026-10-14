import os
import math
import sys
import threading
import pytest
import torch
import torch.nn.functional as F
pytestmark = pytest.mark.skipif(os.environ.get('PI05_HARDWARE') != '1', reason='Requires hardware; set PI05_HARDWARE=1 and run with sudo.')

def _build_old_sequential_precompute(time_mlp_in, time_mlp_out, action_in_proj, num_steps, cfg_min_period, cfg_max_period, device):
    from lerobot.policies.pi05.modeling_pi05 import create_sinusoidal_pos_embedding
    device = torch.device(device) if not isinstance(device, torch.device) else device
    dt = -1.0 / num_steps
    bsize = 1
    sin_dim = action_in_proj.out_features
    cond_list = []
    for s in range(num_steps):
        tv = 1.0 + s * dt
        t_step = torch.tensor(tv, dtype=torch.float32, device=device).expand(bsize)
        emb = create_sinusoidal_pos_embedding(t_step, sin_dim, cfg_min_period, cfg_max_period, device=device)
        x = time_mlp_in(emb.type(t_step.dtype))
        x = F.silu(x)
        x = time_mlp_out(x)
        cond_list.append(F.silu(x).squeeze(0))
    return torch.stack(cond_list, dim=0)

def _make_mock_policy(sin_dim=256, hidden_dim=1024, h_ada=1024, device='cpu'):

    class _MockCfg:
        min_period = 0.001
        max_period = 1.0

    class _MockPolicy:
        config = _MockCfg()
        action_in_proj = torch.nn.Linear(32, sin_dim, device=device)
        time_mlp_in = torch.nn.Linear(sin_dim, hidden_dim, device=device)
        time_mlp_out = torch.nn.Linear(hidden_dim, h_ada, device=device)
    return _MockPolicy()

@pytest.mark.parametrize('num_steps', [1, 3, 5, 10])
def test_b1_batched_mlp_equiv_fp16(num_steps):
    from rpu_backend.adapters.pi05.weights import _precompute_adarms_cond_all
    policy = _make_mock_policy()
    expected_fp32 = _build_old_sequential_precompute(policy.time_mlp_in, policy.time_mlp_out, policy.action_in_proj, num_steps, policy.config.min_period, policy.config.max_period, 'cpu')
    expected_fp16 = expected_fp32.to(torch.float16)
    actual_rpu_fp16 = _precompute_adarms_cond_all(policy, num_steps)
    torch.testing.assert_close(actual_rpu_fp16.cpu(), expected_fp16, rtol=0.001, atol=0.001)

def test_b1_num_steps_variation():
    from rpu_backend.adapters.pi05.weights import _precompute_adarms_cond_all
    policy = _make_mock_policy(h_ada=1024)
    for num_steps in [1, 3, 5, 10]:
        out = _precompute_adarms_cond_all(policy, num_steps)
        assert out.shape == (num_steps, 1024), f'num_steps={num_steps}: expected [{num_steps}, 1024], got {tuple(out.shape)}'
        assert out.dtype == torch.float16

def test_b1_adarms_cond_cache_by_num_steps():
    from rpu_backend.adapters.pi05.weights import _precompute_adarms_cond_all
    policy = _make_mock_policy(h_ada=1024)
    policy._rpu_adarms_cond_cache = {}
    five_a = _precompute_adarms_cond_all(policy, 5)
    five_b = _precompute_adarms_cond_all(policy, 5)
    three = _precompute_adarms_cond_all(policy, 3)
    assert five_b is five_a
    assert three is not five_a
    assert set(policy._rpu_adarms_cond_cache) == {3, 5}

def test_eval_fastpath_skips_warm_recursive_walk_and_recovers_after_train():
    from rpu_backend.adapters.pi05.patches import _install_eval_fastpath_class_patch

    class _CountingModule(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.train_calls = 0

        def train(self, mode=True):
            self.train_calls += 1
            return super().train(mode)

    class _Policy(_CountingModule):

        def __init__(self):
            super().__init__()
            self.child = _CountingModule()
    policy = _Policy()
    _install_eval_fastpath_class_patch(policy)
    policy.eval()
    policy._rpu_swizzled = True
    root_calls = policy.train_calls
    child_calls = policy.child.train_calls
    assert policy.eval() is policy
    assert policy.train_calls == root_calls
    assert policy.child.train_calls == child_calls
    policy.train()
    root_calls = policy.train_calls
    child_calls = policy.child.train_calls
    policy.eval()
    assert not policy.training
    assert policy.train_calls == root_calls + 1
    assert policy.child.train_calls == child_calls + 1

def test_resolve_rpu_device_uses_install_time_cache():
    from rpu_backend.adapters.pi05.runtime import _resolve_rpu_device

    class _Policy:
        _rpu_runtime_device = torch.device('rpu')

        @property
        def paligemma_with_expert(self):
            raise AssertionError('cached device path traversed the model tree')
    assert _resolve_rpu_device(_Policy()) == torch.device('rpu')

@pytest.mark.parametrize('prebuilt_safety', [False, True])
def test_prepare_graphs_freezes_finite_profile_and_rejects_step_miss(monkeypatch, prebuilt_safety):
    from types import SimpleNamespace
    from rpu_backend.adapters.pi05 import Pi05Adapter
    from rpu_backend.adapters.pi05.runtime import _PI05_PREPARED_GRAPH_PROFILES, _validate_prepared_graph_profile
    for name in ('RPU_PI05_SIGLIP_GRAPH', 'RPU_PI05_GEMMA_GRAPH', 'RPU_PI05_ADARMS_GRAPH', 'RPU_PI05_DENOISE_GRAPH', 'RPU_PI05_DENOISE_UNROLL', 'RPU_PI05_KVINSERT_PAD16'):
        monkeypatch.setenv(name, '1')

    class _Cache:

        def __init__(self):
            self.phase = 'CONFIGURING'
            self.entries = 0
            self.replays = 0

        def begin_warmup(self):
            self.phase = 'WARMING'

        def clear(self):
            self.entries = 0
            self.replays = 0

        def size(self):
            return self.entries

        def touch(self):
            if self.entries:
                self.replays += 1
            else:
                self.entries = 1

        def freeze(self):
            assert self.entries == 1
            self.phase = 'READY'

        def snapshot(self):
            return [SimpleNamespace(replay_count=self.replays, recapture_count=0)] if self.entries else []

        def cache_invariant_ok(self):
            return True

    class _Node:
        pass
    model = _Node()
    model.config = SimpleNamespace(num_inference_steps=10)
    model._rpu_siglip_batch_n = 3
    model._rpu_fused_denoise_handle = 1
    model._rpu_fuse_validated = prebuilt_safety
    vision = _Node()
    vision._rpu_siglip_graph_cache = _Cache()
    vision._siglip_graph_enabled = True
    vision_outer = _Node()
    vision_outer.vision_model = vision
    vlm = _Node()
    vlm._rpu_chunk_size = 400
    vlm._rpu_gemma_graph_cache = _Cache()
    vlm._gemma_graph_enabled = True
    expert = _Node()
    expert._rpu_adarms_graph_cache = _Cache()
    expert._adarms_graph_capture_enabled = True
    pawe = _Node()
    pawe.paligemma = SimpleNamespace(model=SimpleNamespace(vision_tower=vision_outer, language_model=vlm))
    pawe.gemma_expert = SimpleNamespace(model=expert)
    model.paligemma_with_expert = pawe
    model._rpu_fused_denoise_graph_cache = _Cache()
    model._pi05_denoise_graph_enabled = True
    model._pi05_denoise_unroll_enabled = True
    if prebuilt_safety:
        expert._rpu_adarms_graph_cache.touch()

    class _Policy:
        _rpu_swizzled = True

        def __init__(self):
            self.model = model

        def predict_action_chunk(self, batch, *, num_steps):
            _validate_prepared_graph_profile(self.model, num_steps)
            vision._rpu_siglip_graph_cache.touch()
            vlm._rpu_gemma_graph_cache.touch()
            model._rpu_fused_denoise_graph_cache.touch()
            if not model._rpu_fuse_validated:
                expert._rpu_adarms_graph_cache.touch()
                model._rpu_fuse_validated = True
            return torch.zeros(1, 2, 3)
    policy = _Policy()
    adapter = Pi05Adapter.__new__(Pi05Adapter)
    adapter._lerobot_policy = policy
    adapter._rpu_is_ready = True
    prepared = adapter.prepare_graphs({}, num_steps=10)
    assert prepared['phase'] == 'READY'
    assert prepared['action_bit_equal']
    assert prepared['graph_counts'] == {'vision': 1, 'prefill': 1, 'denoise': 1, 'safety_adarms': 1}
    assert all((cache.phase == 'READY' for cache in (vision._rpu_siglip_graph_cache, vlm._rpu_gemma_graph_cache, model._rpu_fused_denoise_graph_cache, expert._rpu_adarms_graph_cache)))
    counts_before = tuple((cache.size() for cache in (vision._rpu_siglip_graph_cache, vlm._rpu_gemma_graph_cache, model._rpu_fused_denoise_graph_cache, expert._rpu_adarms_graph_cache)))
    with pytest.raises(RuntimeError, match='prepared READY profile'):
        policy.predict_action_chunk({}, num_steps=5)
    counts_after = tuple((cache.size() for cache in (vision._rpu_siglip_graph_cache, vlm._rpu_gemma_graph_cache, model._rpu_fused_denoise_graph_cache, expert._rpu_adarms_graph_cache)))
    assert counts_after == counts_before
    _PI05_PREPARED_GRAPH_PROFILES.pop(model, None)

def _slow_pi05_preprocess_reference(policy, batch):
    from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch
    images = []
    img_masks = []
    image_keys = list(policy.config.image_features)
    present = [key for key in image_keys if key in batch]
    missing = [key for key in image_keys if key not in batch]
    if not present:
        raise ValueError('all image features are missing')
    device = policy.model._rpu_runtime_device
    for key in present:
        image = batch[key]
        if image.device != device:
            image = image.to(device)
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        channels_first = image.shape[1] == 3
        if channels_first:
            image = image.permute(0, 2, 3, 1)
        if tuple(image.shape[1:3]) != tuple(policy.config.image_resolution):
            image = resize_with_pad_torch(image, *policy.config.image_resolution)
        image = image * 2.0 - 1.0
        if channels_first:
            image = image.permute(0, 3, 1, 2)
        images.append(image)
        img_masks.append(torch.ones(image.shape[0], dtype=torch.bool, device=device))
    for _ in missing:
        images.append(torch.ones_like(images[-1]) * -1)
        img_masks.append(torch.zeros_like(img_masks[-1]))
    return (images, img_masks)

def _make_preprocess_policy(image_resolution=(224, 224), mismatch=False):

    class _Config:
        image_features = {'cam0': object(), 'cam1': object(), 'cam2': object()}
    _Config.image_resolution = image_resolution

    class _Model:
        _rpu_runtime_device = torch.device('cpu')

    class _Policy(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.config = _Config()
            self.model = _Model()
            self.reference_calls = 0

        def _preprocess_images(self, batch):
            self.reference_calls += 1
            (images, masks) = _slow_pi05_preprocess_reference(self, batch)
            if mismatch:
                images[0] = images[0] + 0.125
            return (images, masks)
    policy = _Policy()
    policy.eval()
    return policy

def _make_preprocess_batch(height=224, width=224):
    generator = torch.Generator().manual_seed(20260729)
    return {key: torch.rand(1, 3, height, width, generator=generator) for key in ('cam0', 'cam1', 'cam2')}

@pytest.mark.parametrize(('source_hw', 'target_hw'), [((224, 224), (224, 224)), ((180, 320), (224, 224))])
def test_batched_preprocess_matches_per_camera_reference(source_hw, target_hw):
    from rpu_backend.adapters.pi05.patches import _batched_preprocess_images, _preprocess_outputs_equal
    policy = _make_preprocess_policy(target_hw)
    batch = _make_preprocess_batch(*source_hw)
    ambient_threads = torch.get_num_threads()
    candidate = _batched_preprocess_images(policy, batch, torch.device('cpu'))
    assert torch.get_num_threads() == ambient_threads
    reference = _slow_pi05_preprocess_reference(policy, batch)
    assert candidate is not None
    assert _preprocess_outputs_equal(candidate, reference)

def test_batched_preprocess_self_validates_then_skips_reference():
    from rpu_backend.adapters.pi05.patches import _install_preprocess_images_class_patch, _preprocess_outputs_equal
    policy = _make_preprocess_policy()
    batch = _make_preprocess_batch()
    _install_preprocess_images_class_patch(policy)
    first = policy._preprocess_images(batch)
    assert policy._pi05_batch_preproc_ok is True
    assert policy.reference_calls == 1
    second = policy._preprocess_images(batch)
    assert policy.reference_calls == 1
    assert _preprocess_outputs_equal(first, second)
    policy._pi05_batch_preproc_enabled = False
    policy._preprocess_images(batch)
    assert policy.reference_calls == 2

def test_batched_preprocess_siglip_uploads_one_shared_slab():
    from rpu_backend.adapters.pi05.patches import _batched_preprocess_images
    from rpu_backend.adapters.siglip import _prepare_siglip_images
    policy = _make_preprocess_policy()
    batch = _make_preprocess_batch()
    candidate = _batched_preprocess_images(policy, batch, torch.device('cpu'), fp16_output=True)
    uploaded = _prepare_siglip_images(candidate[0])
    packed = uploaded[0]._base
    assert isinstance(packed, torch.Tensor)
    assert packed.shape == (3, 3, 224, 224)
    assert packed.dtype == torch.float16
    assert packed.device.type == 'rpu'
    for (actual, expected) in zip(uploaded, candidate[0], strict=True):
        assert actual._base is packed
        assert torch.equal(actual.cpu(), expected)

def test_prefill_preserves_shared_slab_until_siglip_and_keeps_masks_cpu():
    from rpu_backend.adapters.pi05.patches import _batched_preprocess_images
    from rpu_backend.adapters.pi05.runtime import _prepare_prefill_siglip_inputs
    policy = _make_preprocess_policy()
    (images, masks) = _batched_preprocess_images(policy, _make_preprocess_batch(), torch.device('cpu'))

    class _Model:
        _rpu_siglip_batch_n = 3
    (prepared_images, prepared_masks) = _prepare_prefill_siglip_inputs(_Model(), images, masks, torch.device('rpu'))
    assert prepared_images is images
    assert all((image.device.type == 'cpu' for image in prepared_images))
    assert prepared_masks is masks
    assert all((mask.device.type == 'cpu' for mask in prepared_masks))

def test_prefill_unmarked_images_preserve_per_image_upload_fallback():
    from rpu_backend.adapters.pi05.runtime import _prepare_prefill_siglip_inputs
    images = [torch.randn(1, 3, 16, 16, dtype=torch.float32) for _ in range(3)]
    masks = [torch.ones(1, dtype=torch.bool) for _ in range(3)]

    class _Model:
        _rpu_siglip_batch_n = 3
    (prepared_images, prepared_masks) = _prepare_prefill_siglip_inputs(_Model(), images, masks, torch.device('rpu'))
    assert all((image.device.type == 'rpu' for image in prepared_images))
    assert prepared_masks is masks
    assert all((mask.device.type == 'cpu' for mask in prepared_masks))

def test_batched_preprocess_mismatch_sticks_to_reference():
    from rpu_backend.adapters.pi05.patches import _install_preprocess_images_class_patch
    policy = _make_preprocess_policy(mismatch=True)
    batch = _make_preprocess_batch()
    _install_preprocess_images_class_patch(policy)
    with pytest.warns(RuntimeWarning, match='differs from LeRobot'):
        first = policy._preprocess_images(batch)
    assert policy._pi05_batch_preproc_ok is False
    assert policy.reference_calls == 1
    second = policy._preprocess_images(batch)
    assert policy.reference_calls == 2
    assert torch.equal(first[0][0], second[0][0])

def test_batched_preprocess_dynamic_inputs_use_reference_without_disabling():
    from rpu_backend.adapters.pi05.patches import _install_preprocess_images_class_patch
    policy = _make_preprocess_policy()
    batch = _make_preprocess_batch()
    del batch['cam1']
    _install_preprocess_images_class_patch(policy)
    (images, masks) = policy._preprocess_images(batch)
    assert policy.reference_calls == 1
    assert not hasattr(policy, '_pi05_batch_preproc_ok')
    assert len(images) == 3
    assert torch.count_nonzero(masks[-1]) == 0
    policy.train()
    policy._preprocess_images(_make_preprocess_batch())
    assert policy.reference_calls == 2
    assert not hasattr(policy, '_pi05_batch_preproc_ok')

def _make_dummy_embed_prefix_inputs(bsize=1, num_img_embs=256, num_lang_embs=200, num_cams=3):
    images = [torch.randn(bsize, 3, 224, 224) for _ in range(num_cams)]
    img_masks = [torch.ones(bsize, dtype=torch.bool) for _ in range(num_cams)]
    tokens = torch.zeros(bsize, num_lang_embs, dtype=torch.int64)
    masks = torch.ones(bsize, num_lang_embs, dtype=torch.bool)
    return (images, img_masks, tokens, masks)

def _make_deterministic_stub_pi05_pytorch(num_img_embs=256, hidden_dim=2048, device='cpu', seed=42):
    rng = torch.Generator(device='cpu').manual_seed(seed)
    img_w = torch.randn(num_img_embs, hidden_dim, generator=rng)
    lang_w = torch.randn(hidden_dim, hidden_dim, generator=rng)

    class _StubPaliGemmaWithExpert:

        def embed_image(self, image):
            B = image.shape[0]
            return img_w[None].expand(B, num_img_embs, hidden_dim).contiguous()

        def embed_language_tokens(self, tokens):
            (B, L) = tokens.shape
            return lang_w[None, :L].expand(B, L, hidden_dim).contiguous()

    class _StubPI05Pytorch(torch.nn.Module):
        paligemma_with_expert = _StubPaliGemmaWithExpert()

        def _apply_checkpoint(self, func, arg):
            return func(arg)
    stub = _StubPI05Pytorch()
    stub.eval()
    return stub

def test_b3a1_embed_prefix_equiv_3cam(monkeypatch):
    from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch
    (images, img_masks, tokens, masks) = _make_dummy_embed_prefix_inputs(num_cams=3)
    stub_a = _make_deterministic_stub_pi05_pytorch(seed=42)
    (expected_embs, expected_pad, expected_att) = PI05Pytorch.embed_prefix(stub_a, images, img_masks, tokens, masks)
    stub_b = _make_deterministic_stub_pi05_pytorch(seed=42)
    from rpu_backend.adapters.pi05.patches import _patched_embed_prefix
    (actual_embs, actual_pad, actual_att) = _patched_embed_prefix(stub_b, images, img_masks, tokens, masks)
    assert torch.equal(actual_embs, expected_embs), 'embs content differs'
    assert torch.equal(actual_pad, expected_pad), 'pad_masks content differs'
    assert torch.equal(actual_att, expected_att), 'att_masks content differs'

def test_b3a1_embed_prefix_equiv_missing_cam(monkeypatch):
    from rpu_backend.adapters.pi05.patches import _patched_embed_prefix
    stub = _make_deterministic_stub_pi05_pytorch()
    (images, img_masks, tokens, masks) = _make_dummy_embed_prefix_inputs(num_cams=3)
    img_masks[1] = torch.zeros(1, dtype=torch.bool)
    (embs, pad, att) = _patched_embed_prefix(stub, images, img_masks, tokens, masks)
    assert att.sum().item() == 0
    cam1_pad = pad[:, 256:512]
    assert cam1_pad.sum().item() == 0

def test_b3a1_embed_prefix_training_fallback(monkeypatch):
    from rpu_backend.adapters.pi05 import patches
    sentinel = ('SENTINEL_TRAINING_PATH', 'pad', 'att')
    monkeypatch.setattr(patches, '_orig_embed_prefix', lambda self, *args, **kwargs: sentinel)
    stub = _make_deterministic_stub_pi05_pytorch()
    stub.train()
    (images, img_masks, tokens, masks) = _make_dummy_embed_prefix_inputs()
    out = patches._patched_embed_prefix(stub, images, img_masks, tokens, masks)
    assert out == sentinel, f'training mode did not delegate to _orig_embed_prefix; got {out!r}'

def test_packed_prefix_device_assembly_is_exact_fresh_and_caches_text():
    from types import SimpleNamespace
    from rpu_backend.adapters.pi05.patches import _assemble_packed_prefix_ondevice
    embedding = torch.nn.Embedding(16, 32, dtype=torch.float16)

    class _PaliGemmaWithExpert:

        def __init__(self):
            self.calls = 0
            self.paligemma = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(embed_tokens=embedding)))

        def embed_language_tokens(self, tokens):
            self.calls += 1
            return embedding(tokens)

    class _Model:

        def __init__(self):
            self.paligemma_with_expert = _PaliGemmaWithExpert()
            self._pi05_language_prefix_cache = None
    model = _Model()
    packed = torch.randn(1, 6, 32, dtype=torch.float16, device='rpu')
    tokens = torch.tensor([[2, 5, 1, 9]], dtype=torch.int64)
    expected_lang = embedding(tokens) * math.sqrt(32)
    expected = torch.cat([packed.cpu(), expected_lang], dim=1)
    (first, first_lang) = _assemble_packed_prefix_ondevice(model, packed, tokens)
    (second, second_lang) = _assemble_packed_prefix_ondevice(model, packed, tokens.clone())
    assert model.paligemma_with_expert.calls == 1
    assert torch.equal(first_lang, expected_lang)
    assert torch.equal(second_lang, expected_lang)
    assert torch.equal(first.cpu(), expected)
    assert torch.equal(second.cpu(), expected)
    assert first.data_ptr() != second.data_ptr()
    changed = tokens.clone()
    changed[0, 0] = 3
    _assemble_packed_prefix_ondevice(model, packed, changed)
    assert model.paligemma_with_expert.calls == 2

def test_packed_prefix_first_parity_handles_rpu_image_cpu_language(monkeypatch):
    from types import SimpleNamespace
    from rpu_backend.adapters import siglip
    from rpu_backend.adapters.pi05 import patches
    embedding = torch.nn.Embedding(16, 32, dtype=torch.float16)
    vision_tower = SimpleNamespace()

    class _PaliGemmaWithExpert:

        def __init__(self):
            self.paligemma = SimpleNamespace(model=SimpleNamespace(vision_tower=vision_tower, language_model=SimpleNamespace(embed_tokens=embedding)))

        def embed_language_tokens(self, tokens):
            return embedding(tokens)

    class _Model:
        training = False
        _rpu_siglip_batch_n = 3
        _pi05_prefix_assembly_enabled = True
        _pi05_prefix_assembly_ok = None
        _pi05_language_prefix_cache = None

        def __init__(self):
            self.paligemma_with_expert = _PaliGemmaWithExpert()
    packed = torch.randn(1, 6, 32, dtype=torch.float16, device='rpu')
    monkeypatch.setattr(siglip, '_siglip_forward_images', lambda _model, _images: packed)
    tokens = torch.tensor([[2, 5, 1, 9]], dtype=torch.int64)
    masks = torch.ones_like(tokens, dtype=torch.bool)
    images = [torch.zeros(1)] * 3
    image_masks = [torch.ones(1, dtype=torch.bool) for _ in images]
    model = _Model()
    (embs, pad_masks, att_masks) = patches._patched_embed_prefix(model, images, image_masks, tokens, masks)
    expected_language = embedding(tokens) * math.sqrt(32)
    expected = torch.cat([packed.cpu(), expected_language], dim=1)
    assert embs.device.type == 'rpu'
    assert torch.equal(embs.cpu(), expected)
    assert model._pi05_prefix_assembly_ok is True
    assert pad_masks.shape == att_masks.shape == (1, 10)

@pytest.mark.parametrize('failure_mode', ['none', 'raise'])
def test_packed_prefix_assembly_failure_uses_mixed_device_safe_fallback(monkeypatch, failure_mode):
    from types import SimpleNamespace
    from rpu_backend.adapters import siglip
    from rpu_backend.adapters.pi05 import patches
    embedding = torch.nn.Embedding(16, 32, dtype=torch.float16)
    vision_tower = SimpleNamespace()

    class _PaliGemmaWithExpert:

        def __init__(self):
            self.paligemma = SimpleNamespace(model=SimpleNamespace(vision_tower=vision_tower, language_model=SimpleNamespace(embed_tokens=embedding)))

        def embed_language_tokens(self, tokens):
            return embedding(tokens)

    class _Model:
        training = False
        _rpu_siglip_batch_n = 3
        _pi05_prefix_assembly_enabled = True
        _pi05_prefix_assembly_ok = None
        _pi05_language_prefix_cache = None

        def __init__(self):
            self.paligemma_with_expert = _PaliGemmaWithExpert()
    packed = torch.randn(1, 6, 32, dtype=torch.float16, device='rpu')
    monkeypatch.setattr(siglip, '_siglip_forward_images', lambda _model, _images: packed)
    tokens = torch.tensor([[2, 5, 1, 9]], dtype=torch.int64)
    masks = torch.ones_like(tokens, dtype=torch.bool)
    images = [torch.zeros(1)] * 3
    image_masks = [torch.ones(1, dtype=torch.bool) for _ in images]
    expected_language = embedding(tokens) * math.sqrt(32)
    expected = torch.cat([packed.cpu(), expected_language], dim=1)

    def fail_assembly(*_args, **_kwargs):
        if failure_mode == 'raise':
            raise RuntimeError('injected assembly failure')
        return (None, expected_language)
    monkeypatch.setattr(patches, '_assemble_packed_prefix_ondevice', fail_assembly)
    model = _Model()
    if failure_mode == 'raise':
        with pytest.warns(RuntimeWarning, match='assembly failed'):
            result = patches._patched_embed_prefix(model, images, image_masks, tokens, masks)
    else:
        result = patches._patched_embed_prefix(model, images, image_masks, tokens, masks)
    (embs, pad_masks, att_masks) = result
    assert embs.device.type == 'rpu'
    assert torch.equal(embs.cpu(), expected)
    assert pad_masks.shape == att_masks.shape == (1, 10)

def test_b3a1_embed_prefix_env_optout(monkeypatch):
    monkeypatch.setenv('RPU_PI05_EMBED_PREFIX_PATCH', '0')
    from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch
    original_method = PI05Pytorch.embed_prefix
    had_sentinel = getattr(PI05Pytorch, '_rpu_patched_embed_prefix', False)
    try:
        if had_sentinel:
            from rpu_backend.adapters.pi05.patches import _orig_embed_prefix
            PI05Pytorch.embed_prefix = _orig_embed_prefix
            PI05Pytorch._rpu_patched_embed_prefix = False
        from rpu_backend.adapters.pi05.patches import _install_embed_prefix_class_patch
        _install_embed_prefix_class_patch()
        assert not getattr(PI05Pytorch, '_rpu_patched_embed_prefix', False)
    finally:
        PI05Pytorch.embed_prefix = original_method
        PI05Pytorch._rpu_patched_embed_prefix = had_sentinel
OPENPI_ATTENTION_MASK_VALUE = -2.3819763e+38

def _mock_prepare_4d_fn(att_2d_masks):
    att_2d_masks_4d = att_2d_masks[:, None, :, :]
    return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

@pytest.fixture(autouse=True)
def _reset_cache_between_b3a2_tests(request):
    if 'b3a2' not in request.node.name:
        yield
        return
    from rpu_backend.adapters.pi05 import runtime as _rt
    _rt._reset_prefix_mask_cache()
    yield
    _rt._reset_prefix_mask_cache()

def test_b3a2_cache_miss_then_hit():
    from rpu_backend.adapters.pi05 import runtime as _rt
    pad = torch.ones(1, 968, dtype=torch.bool)
    att = torch.zeros(1, 968, dtype=torch.bool)
    (m1, p1) = _rt._cached_build_prefix_mask_4d(pad, att, _mock_prepare_4d_fn)
    assert _rt._PREFIX_MASK_CACHE_MISSES == 1
    assert _rt._PREFIX_MASK_CACHE_HITS == 0
    (m2, p2) = _rt._cached_build_prefix_mask_4d(pad, att, _mock_prepare_4d_fn)
    assert _rt._PREFIX_MASK_CACHE_MISSES == 1
    assert _rt._PREFIX_MASK_CACHE_HITS == 1
    assert m1 is m2
    assert p1 is p2
    pad2 = pad.clone()
    pad2[0, 100] = False
    (m3, p3) = _rt._cached_build_prefix_mask_4d(pad2, att, _mock_prepare_4d_fn)
    assert _rt._PREFIX_MASK_CACHE_MISSES == 2
    assert _rt._PREFIX_MASK_CACHE_HITS == 1
    (m4, p4) = _rt._cached_build_prefix_mask_4d(pad2, att, _mock_prepare_4d_fn)
    assert _rt._PREFIX_MASK_CACHE_HITS == 2
    assert m3 is m4

def test_b3a2_attn_nonzero_fallback():
    from rpu_backend.adapters.pi05 import runtime as _rt
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
    pad = torch.ones(1, 968, dtype=torch.bool)
    att = torch.zeros(1, 968, dtype=torch.bool)
    att[0, 500] = True
    (mask4d, pos_ids) = _rt._cached_build_prefix_mask_4d(pad, att, _mock_prepare_4d_fn)
    att_2d_expected = make_att_2d_masks(pad, att)
    mask4d_expected = _mock_prepare_4d_fn(att_2d_expected)
    pos_ids_expected = torch.cumsum(pad, dim=1) - 1
    assert torch.equal(mask4d, mask4d_expected)
    assert torch.equal(pos_ids, pos_ids_expected)

def test_b3a2_env_optout(monkeypatch):
    monkeypatch.setenv('RPU_PI05_PREFIX_MASK_CACHE', '0')
    from rpu_backend.adapters.pi05 import runtime as _rt
    import lerobot.policies.pi05.modeling_pi05 as _lerobot_mod
    call_count = [0]
    real_fn = _lerobot_mod.make_att_2d_masks

    def _tracking_fn(pad_masks, att_masks):
        call_count[0] += 1
        return real_fn(pad_masks, att_masks)
    monkeypatch.setattr(_lerobot_mod, 'make_att_2d_masks', _tracking_fn)
    pad = torch.ones(1, 968, dtype=torch.bool)
    att = torch.zeros(1, 968, dtype=torch.bool)
    _rt._reset_prefix_mask_cache()
    _ = _rt._cached_build_prefix_mask_4d(pad, att, _mock_prepare_4d_fn)
    _ = _rt._cached_build_prefix_mask_4d(pad, att, _mock_prepare_4d_fn)
    assert call_count[0] == 2, f'env opt-out should call upstream every time; got {call_count[0]}'
    assert _rt._PREFIX_MASK_CACHE_MISSES == 0
    assert _rt._PREFIX_MASK_CACHE_HITS == 0
    assert _rt._PREFIX_MASK_SNAPSHOT is None

def test_b3a2_thread_safety():
    from rpu_backend.adapters.pi05 import runtime as _rt
    pad_a = torch.ones(1, 968, dtype=torch.bool)
    pad_b = pad_a.clone()
    pad_b[0, 500] = False
    att = torch.zeros(1, 968, dtype=torch.bool)
    errors = []

    def _worker(pad):
        try:
            for _ in range(50):
                (m, p) = _rt._cached_build_prefix_mask_4d(pad, att, _mock_prepare_4d_fn)
                expected_pos = torch.cumsum(pad, dim=1) - 1
                assert torch.equal(p, expected_pos), f'thread saw mismatched pos_ids for pad sum={pad.sum().item()}'
        except AssertionError as e:
            errors.append(str(e))
    t1 = threading.Thread(target=_worker, args=(pad_a,))
    t2 = threading.Thread(target=_worker, args=(pad_b,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert not errors, f'thread-safety violation: {errors}'

def test_b3a2_device_mismatch():
    from rpu_backend.adapters.pi05 import runtime as _rt
    import warnings
    pad_cpu = torch.zeros(1, 10, dtype=torch.bool)
    att_cpu = torch.zeros(1, 10, dtype=torch.bool)
    pad_rpu = pad_cpu.to('rpu')
    att_rpu = att_cpu.to('rpu')
    _rt._HASH_MASK_NONCPU_WARNED.clear()
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        _ = _rt._hash_mask(pad_rpu)
    assert any((issubclass(w.category, RuntimeWarning) and 'rpu' in str(w.message).lower() for w in ws)), f'no RuntimeWarning fired for non-CPU mask; got {[str(w.message) for w in ws]}'
    key_cpu = _rt._make_cache_key(pad_cpu, att_cpu)
    key_rpu = _rt._make_cache_key(pad_rpu, att_rpu)
    assert key_cpu != key_rpu, 'device should be part of the key'
    assert isinstance(key_cpu, tuple) and len(key_cpu) == 8

def test_denoise_mask_cache_lru1_content_key():
    from rpu_backend.adapters.pi05.runtime import _build_attention_mask_4d

    class _Policy:
        _rpu_denoise_mask_cache = None

        @staticmethod
        def _prepare_attention_masks_4d(att_2d_masks):
            return _mock_prepare_4d_fn(att_2d_masks)
    policy = _Policy()
    pad_a = torch.ones(1, 8, dtype=torch.bool)
    mask_a = _build_attention_mask_4d(policy, pad_a, 8, 1, 4, torch.float16, torch.device('cpu'))
    mask_a_again = _build_attention_mask_4d(policy, pad_a.clone(), 8, 1, 4, torch.float16, torch.device('cpu'))
    assert mask_a_again is mask_a
    pad_b = pad_a.clone()
    pad_b[0, 3] = False
    mask_b = _build_attention_mask_4d(policy, pad_b, 8, 1, 4, torch.float16, torch.device('cpu'))
    assert mask_b is not mask_a
    assert not torch.equal(mask_b, mask_a)
