import ast
import os
import re
from pathlib import Path
from rpu_backend.runtime import rpu_env_bool
ROOT = Path(__file__).resolve().parents[4]
RUNTIME = ROOT / 'python' / 'rpu_backend' / 'adapters' / 'wall_oss' / 'runtime.py'
CAMERAS = {'face_view': 'front view', 'left_wrist_view': 'left wrist view', 'right_wrist_view': 'right wrist view'}

def _prompt_builder():
    tree = ast.parse(RUNTIME.read_text())
    function = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_build_flow_prompt'))
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {'os': os, 're': re, '_CAM_NAME_MAPPING': CAMERAS, 'rpu_env_bool': rpu_env_bool}
    exec(compile(module, str(RUNTIME), 'exec'), namespace)
    return namespace['_build_flow_prompt']

def _set_switches(monkeypatch, *, multisuite: str, generic: str):
    monkeypatch.setenv('RPU_WALL_OSS_MULTISUITE_PROLOGUE', multisuite)
    monkeypatch.setenv('RPU_WALL_OSS_GENERIC_PROLOGUE', generic)
    monkeypatch.delenv('RPU_WALL_OSS_SHORT_PROMPT', raising=False)
    monkeypatch.delenv('RPU_WALL_OSS_PROLOGUE_FILLER', raising=False)
    monkeypatch.delenv('RPU_WALL_OSS_RTC_EMBODIMENT', raising=False)

def test_multisuite_prologue_is_exact_bytes(monkeypatch):
    _set_switches(monkeypatch, multisuite='1', generic='0')
    actual = _prompt_builder()('pick up the mug', ['face_view', 'left_wrist_view', 'right_wrist_view'], 'libero_all', False, '1 2 3')
    expected = '<|im_start|>system\nYou are an embodied vision-language-action (VLA) model controlling the robot with language instructions.\n Embodiment: Single Franka\n Camera Setup: front view, left wrist view, right wrist view,\n Frequency: 10HZ\n Action Space: Abs EEF\n<|im_end|>\n<|im_start|>user\nObservation: front view: <|vision_start|><|image_pad|><|vision_end|> left wrist view: <|vision_start|><|image_pad|><|vision_end|> right wrist view: <|vision_start|><|image_pad|><|vision_end|>\nInstruction: pick up the mug\nPredict the next action in robot action.\nProprioception: 1 2 3\n<|im_end|>\n<|im_start|>assistant\n'
    assert actual == expected

def test_v728_rtc_prompt_is_customer_exact_bytes(monkeypatch):
    _set_switches(monkeypatch, multisuite='0', generic='0')
    actual = _prompt_builder()('sort parcel [robot: 10146; speed: 1.4]', ['face_view', 'left_wrist_view', 'right_wrist_view'], 'ex_normal', True, '1 2 3', rtc_enabled=True)
    expected = '<|im_start|>system\nYou are an embodied vision-language-action (VLA) model controlling the robot with language instructions.\n Embodiment: X2 ex001_desktop\n Camera Setup: front view, left wrist view, right wrist view,\n Frequency: 32HZ\n Action Space: Rel EEF\n<|im_end|>\n<|im_start|>user\nObservation: front view: <|vision_start|><|image_pad|><|vision_end|> left wrist view: <|vision_start|><|image_pad|><|vision_end|> right wrist view: <|vision_start|><|image_pad|><|vision_end|>\nInstruction: sort parcel [robot: 10146]\nPredict the next action in robot action.\nProprioception: 1 2 3\n<|im_end|>\n<|im_start|>assistant\n'
    assert actual == expected

def test_v728_rtc_rejects_non_training_prompt_switches(monkeypatch):
    _set_switches(monkeypatch, multisuite='0', generic='1')
    try:
        _prompt_builder()('task', ['face_view'], 'ex_normal', True, '1', rtc_enabled=True)
    except ValueError as exc:
        assert 'default v728 embodied prompt' in str(exc)
    else:
        raise AssertionError('RTC must reject a non-training prompt recipe')

def test_prologue_switches_are_mutually_exclusive(monkeypatch):
    _set_switches(monkeypatch, multisuite='1', generic='1')
    try:
        _prompt_builder()('task', ['face_view'], 'libero_all', False, '1 2 3')
    except ValueError as exc:
        assert 'mutually exclusive' in str(exc)
    else:
        raise AssertionError('conflicting prologue switches must fail')

def test_generic_prologue_filler_is_opt_in(monkeypatch):
    _set_switches(monkeypatch, multisuite='0', generic='1')
    builder = _prompt_builder()
    baseline = builder('task', ['face_view'], 'libero_all', False, '1 2 3')
    monkeypatch.setenv('RPU_WALL_OSS_PROLOGUE_FILLER', '<diag-padding>')
    diagnostic = builder('task', ['face_view'], 'libero_all', False, '1 2 3')
    assert '<diag-padding>' not in baseline
    assert diagnostic == baseline.replace('Action Space: Abs EEF\n', 'Action Space: Abs EEF<diag-padding>\n', 1)
