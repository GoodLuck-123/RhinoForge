"""Saved Qwen3-VL input replay must preserve geometry and normal M-RoPE."""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

from examples.common.config import ConfigError, load_config
from examples.runners.qwen3_vl.inputs import load_prepared_input, validate_prepared_positions
from examples.runners.qwen3_vl import runner

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'examples/configs/qwen3_vl/vl/4b/w8a16_text_head_vision.toml'


@pytest.fixture
def frozen(tmp_path):
    # Independent witness for one 30x40 patch grid, 300 merged image tokens,
    # vision boundaries and 128 body tokens: maximum position149, delta-280.
    ids = torch.tensor([[151652] + [151655] * 300 + [151653] + [42] * 128])
    positions = torch.zeros((3, 1, 430), dtype=torch.int64)
    positions[0, 0, 1:301] = 1
    positions[1, 0, 1:301] = torch.arange(1, 16).repeat_interleave(20)
    positions[2, 0, 1:301] = torch.arange(1, 21).repeat(15)
    positions[:, 0, 301:] = torch.arange(21, 150)
    tensors = {'input_ids': ids, 'attention_mask': torch.ones_like(ids),
               'image_grid_thw': torch.tensor([[1, 30, 40]]),
               'pixel_values': torch.zeros((1200, 1536), dtype=torch.float32),
               'position_ids': positions}
    tensors['pixel_values'][0, 0] = 0.25
    path = tmp_path / 'input.safetensors'
    save_file(tensors, path)
    inputs = {'component': 'multimodal', 'prepared_input_safetensors': str(path),
              'prepared_input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'decode_steps': 2, 'stop_on_eos': False}
    return tensors, inputs


def config_file(tmp_path, inputs):
    import json
    input_text = '\n'.join(f'{name} = {json.dumps(value)}' for name, value in inputs.items())
    body = TEMPLATE.read_text()
    before, remainder = body.split('[input]\n', 1)
    after = remainder.split('[run]\n', 1)[1]
    path = tmp_path / 'example.toml'
    path.write_text(before + '[input]\n' + input_text + '\n\n[run]\n' + after)
    return path


def test_saved_input_real_hf_positions_and_all_tensor_identities(frozen):
    expected, inputs = frozen
    tensors, metadata = load_prepared_input(inputs)
    encoded = validate_prepared_positions(tensors, metadata, Qwen3VLConfig())
    assert metadata['file_sha256'] == inputs['prepared_input_sha256']
    assert metadata['normal_mrope_positions_verified']
    assert metadata['rope_deltas'] == [[-280]]
    for name, value in expected.items():
        assert torch.equal(tensors[name], value)
        assert metadata['tensors'][name]['sha256'] == hashlib.sha256(value.numpy().tobytes()).hexdigest()
    assert 'position_ids' not in encoded
    assert torch.equal(encoded['mm_token_type_ids'], (expected['input_ids'] == 151655).long())


def test_file_tamper_rejected_before_any_checkpoint_context(frozen, monkeypatch):
    _, inputs = frozen
    path = Path(inputs['prepared_input_safetensors'])
    with path.open('ab') as stream:
        stream.write(b'tamper')
    context = Mock()
    monkeypatch.setattr(runner, 'text_context', context)
    config = {'example': {'target': 'qwen3_vl'}, 'input': inputs}
    with pytest.raises(ValueError, match='SHA256'):
        runner.prepare(config, SimpleNamespace(owner=None))
    context.assert_not_called()


@pytest.mark.parametrize('corruption,match', [
    ('mask', 'only ones'), ('dtype', 'dtype'), ('patches', 'patches matching'),
    ('grid', 'single-frame'), ('position_shape', 'position_ids'), ('finite', 'finite patches'),
    ('patch_width', 'patch/merge'), ('placeholder', 'placeholder block'),
    ('position_value', 'normal M-RoPE'),
])
def test_invalid_saved_geometry_or_positions_rejected(frozen, corruption, match):
    tensors, inputs = frozen
    if corruption == 'mask': tensors['attention_mask'][0, -1] = 0
    elif corruption == 'dtype': tensors['input_ids'] = tensors['input_ids'].int()
    elif corruption == 'patches': tensors['pixel_values'] = tensors['pixel_values'][:-1]
    elif corruption == 'grid': tensors['image_grid_thw'][0, 0] = 2
    elif corruption == 'position_shape': tensors['position_ids'] = tensors['position_ids'][:, :, :-1].contiguous()
    elif corruption == 'finite': tensors['pixel_values'][0, 0] = float('nan')
    elif corruption == 'patch_width': tensors['pixel_values'] = tensors['pixel_values'][:, :-1].contiguous()
    elif corruption == 'placeholder': tensors['input_ids'][0, 30] = 42
    elif corruption == 'position_value': tensors['position_ids'][2, 0, -1] += 1
    path = Path(inputs['prepared_input_safetensors'])
    save_file(tensors, path)
    inputs['prepared_input_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match=match):
        parsed, metadata = load_prepared_input(inputs)
        validate_prepared_positions(parsed, metadata, Qwen3VLConfig())


@pytest.mark.parametrize('change,match', [
    ({'prepared_input_sha256': None}, 'SHA256'),
    ({'prepared_input_safetensors': None}, 'non-empty path'),
    ({'prepared_input_sha256': 'A' * 64}, 'SHA256'),
    ({'images': ['image.png']}, 'prompt/images'),
    ({'prompt': 'replace saved tokens'}, 'prompt/images'),
    ({'prefill_tokens': 430}, 'input extensions'),
    ({'component': 'text'}, 'multimodal'),
])
def test_config_rejects_ambiguous_prepared_input(tmp_path, frozen, change, match):
    _, inputs = frozen
    inputs.update(change)
    # TOML has no None; omission exercises pair validation separately below.
    inputs = {name: value for name, value in inputs.items() if value is not None}
    if 'prepared_input_sha256' not in inputs or 'prepared_input_safetensors' not in inputs:
        match = 'supplied together'
    with pytest.raises(ConfigError, match=match):
        load_config(config_file(tmp_path, inputs))


@pytest.mark.parametrize('target,component', [('qwen3', 'multimodal'), ('qwen3_vl', 'vision')])
def test_prepared_input_cannot_change_family_or_component(frozen, target, component):
    from examples.common.config import validate_generation_options
    _, inputs = frozen
    inputs['component'] = component
    with pytest.raises(ConfigError, match='Qwen3-VL multimodal'):
        validate_generation_options({'example': {'target': target}, 'input': inputs})


def test_relative_prepared_path_and_check_config_without_torch(tmp_path, frozen):
    _, inputs = frozen
    inputs['prepared_input_safetensors'] = 'input.safetensors'
    path = config_file(tmp_path, inputs)
    assert load_config(path)['input']['prepared_input_safetensors'] == str(tmp_path / 'input.safetensors')
    script = '''
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*args,**kwargs):
  if fullname.split('.')[0] in {'torch','rpu_backend','transformers','safetensors'}:
   raise AssertionError('check-config imported ' + fullname)
sys.meta_path.insert(0,Block())
from examples.common.cli import main
raise SystemExit(main(['--config',sys.argv[1],'--check-config']))
'''
    result = subprocess.run([sys.executable, '-c', script, str(path)], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_position_witness_failure_precedes_processor_or_model_install(tmp_path, frozen, monkeypatch):
    import transformers
    from examples.common import config as common
    tensors, inputs = frozen
    tensors['position_ids'][0, 0, -1] += 1
    path = Path(inputs['prepared_input_safetensors'])
    save_file(tensors, path)
    inputs['prepared_input_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    config = load_config(config_file(tmp_path, inputs))
    monkeypatch.setattr(runner, 'text_context', lambda *_:
                        (config['example'], inputs, 'qwen3_vl', '/unused/checkpoint', None, 2))
    monkeypatch.setattr(common, 'validate_checkpoint_profile', lambda *_: None)
    monkeypatch.setattr(transformers.AutoConfig, 'from_pretrained', lambda *_args, **_kwargs: Qwen3VLConfig())
    processor_load = Mock(side_effect=AssertionError('positions must be checked before processor/model'))
    monkeypatch.setattr(transformers.AutoProcessor, 'from_pretrained', processor_load)
    api = ModuleType('rpu_backend.api')
    model_load = Mock(side_effect=AssertionError('positions must be checked before model install'))
    api.RPUModelForConditionalGeneration = SimpleNamespace(from_pretrained=model_load)
    monkeypatch.setitem(sys.modules, 'rpu_backend.api', api)
    with pytest.raises(ValueError, match='normal M-RoPE'):
        runner.prepare(config, SimpleNamespace(owner=None))
    processor_load.assert_not_called()
    model_load.assert_not_called()


def test_prepared_caller_handoff_keeps_normal_prefill_and_decode_positions(tmp_path, frozen, monkeypatch):
    import transformers
    from examples.common import config as common
    expected, inputs = frozen
    config = load_config(config_file(tmp_path, inputs))
    cfg = Qwen3VLConfig()
    checkpoint = Mock(return_value=(config['example'], config['input'], 'qwen3_vl',
                                    '/unused/checkpoint', config['rpu_execution'], 2))
    monkeypatch.setattr(runner, 'text_context', checkpoint)
    monkeypatch.setattr(common, 'validate_checkpoint_profile', lambda *_: None)
    monkeypatch.setattr(transformers.AutoConfig, 'from_pretrained', lambda *_args, **_kwargs: cfg)
    processor = SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda *_args, **_kwargs: ''))
    processor.apply_chat_template = Mock(side_effect=AssertionError('saved inputs must not use chat'))
    monkeypatch.setattr(transformers.AutoProcessor, 'from_pretrained', lambda *_args, **_kwargs: processor)
    cache = SimpleNamespace(position=0)
    cache.reset = lambda: setattr(cache, 'position', 0)
    cache.get_seq_length = lambda: cache.position
    inner = SimpleNamespace(config=cfg, rope_deltas=None, language_model=object())
    inner.get_vision_position_ids = MethodType(Qwen3VLModel.get_vision_position_ids, inner)
    inner.get_rope_index = MethodType(Qwen3VLModel.get_rope_index, inner)
    calls = []

    class Model:
        model = inner
        config = cfg
        def __call__(self, input_ids, past_key_values, **kwargs):
            assert 'position_ids' not in kwargs  # witness must not bypass delta initialization
            positions = Qwen3VLModel.compute_3d_position_ids(
                inner, input_ids, torch.zeros((1, input_ids.shape[1], 1)),
                image_grid_thw=kwargs.get('image_grid_thw'),
                attention_mask=kwargs.get('attention_mask'),
                past_key_values=past_key_values, mm_token_type_ids=kwargs.get('mm_token_type_ids'))
            calls.append((input_ids.clone(), kwargs, positions.clone()))
            past_key_values.position += input_ids.shape[1]
            return SimpleNamespace(logits=torch.zeros((1, 1, 10)))

    model = Model()
    api = ModuleType('rpu_backend.api')
    api.RPUModelForConditionalGeneration = SimpleNamespace(from_pretrained=Mock(return_value=model))
    api.RPUCache = SimpleNamespace(from_model=Mock(return_value=cache))
    api.greedy_token_ids = lambda logits: logits.argmax(-1, keepdim=True)
    monkeypatch.setitem(sys.modules, 'rpu_backend', ModuleType('rpu_backend'))
    monkeypatch.setitem(sys.modules, 'rpu_backend.api', api)
    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, 'to', lambda self, *args, **kwargs:
                        original_to(self, 'cpu' if args and args[0] == 'rpu' else args[0], **kwargs)
                        if args else original_to(self, **kwargs))
    state = SimpleNamespace(owner=None)
    infer, owner = runner.prepare(config, state)
    result = infer()
    assert owner is state.owner is model
    assert torch.equal(calls[0][0], expected['input_ids'])
    for name in ['attention_mask', 'pixel_values', 'image_grid_thw']:
        assert torch.equal(calls[0][1][name], expected[name])
    assert torch.equal(calls[0][1]['mm_token_type_ids'], (expected['input_ids'] == 151655).long())
    assert torch.equal(calls[0][2], expected['position_ids'])
    assert inner.rope_deltas.tolist() == [[-280]]
    assert calls[1][2].flatten().tolist() == [150, 150, 150]
    assert calls[2][2].flatten().tolist() == [151, 151, 151]
    assert result['generation']['prepared_input']['rope_deltas'] == [[-280]]
    processor.apply_chat_template.assert_not_called()
