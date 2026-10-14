from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from rpu_backend.api.lingbot2 import Lingbot2Policy


class _Tokenizer:
    def __init__(self) -> None:
        self.chat_inputs = []
        self.tokenize_inputs = []

    def apply_chat_template(self, messages, **kwargs):
        self.chat_inputs.append((messages, kwargs))
        return "<chat>rewritten prompt</chat>"

    def __call__(self, prompt, **kwargs):
        self.tokenize_inputs.append((prompt, kwargs))
        return {
            "input_ids": torch.arange(72).reshape(1, 72),
            "attention_mask": torch.tensor([[1] * 70 + [0, 0]]),
        }


@pytest.mark.parametrize("instruction", ["pick", "<chat>preformatted prompt</chat>"])
def test_lingbot_instruction_preserves_prompt_and_token_mask(instruction) -> None:
    tokenizer = _Tokenizer()
    policy = Lingbot2Policy.__new__(Lingbot2Policy)
    policy._processor = SimpleNamespace(tokenizer=tokenizer)
    policy._max_lang_tokens = 72

    tokens, mask = policy._encode_instruction(instruction)

    assert tokens.shape == mask.shape == (1, 72)
    assert tokens.dtype == torch.long and mask.dtype == torch.bool
    assert mask.tolist() == [[True] * 70 + [False, False]]
    assert tokenizer.chat_inputs == []
    assert tokenizer.tokenize_inputs == [
        (instruction, {"return_tensors": "pt", "padding": "max_length",
                       "truncation": True, "max_length": 72})
    ]


@pytest.mark.parametrize("qwen_base", [None, "base-config"])
def test_lingbot_processor_is_loaded_lazily_from_checkpoint(
    tmp_path, monkeypatch, qwen_base
) -> None:
    from transformers import AutoProcessor

    processor = object()
    load = Mock(return_value=processor)
    monkeypatch.setattr(AutoProcessor, "from_pretrained", load)
    policy = Lingbot2Policy.from_checkpoint(tmp_path, qwen3vl_base=qwen_base)
    load.assert_not_called()
    assert policy._get_processor() is processor
    assert policy._get_processor() is processor
    load.assert_called_once_with(str(tmp_path))
    assert policy._rpu_build_started is False


def test_lingbot_processor_load_failure_keeps_cold_policy(tmp_path, monkeypatch) -> None:
    from transformers import AutoProcessor

    load = Mock(side_effect=OSError("processor fixture missing"))
    monkeypatch.setattr(AutoProcessor, "from_pretrained", load)
    policy = Lingbot2Policy.from_checkpoint(tmp_path)
    with pytest.raises(OSError, match="processor fixture missing"):
        policy._get_processor()
    assert policy._processor is None
    assert policy._policy is None
    assert policy._rpu_build_started is False
    assert policy._rpu_ready is False


def test_lingbot_training_config_is_stored_without_prompt_rewriting(tmp_path) -> None:
    config = tmp_path / "training.yaml"
    config.write_text("model:\n  use_qwen3_chat_template: true\n", encoding="utf-8")
    policy = Lingbot2Policy.from_checkpoint(tmp_path, training_config_path=config)
    assert policy._training_config_path == str(config)
    tokenizer = _Tokenizer()
    policy._processor = SimpleNamespace(tokenizer=tokenizer)
    policy._encode_instruction("pick")
    assert tokenizer.tokenize_inputs[0][0] == "pick"
    assert tokenizer.chat_inputs == []


class _ImageProcessor:
    def __init__(self) -> None:
        self.inputs = []

    def __call__(self, *, images, return_tensors):
        assert return_tensors == "pt"
        self.inputs.append([image.copy() for image in images])
        height, width = images[0].shape[:2]
        return {
            "pixel_values": torch.from_numpy(np.stack(images)).reshape(-1, 3),
            "image_grid_thw": torch.tensor([[1, height, width]] * len(images)),
        }


def test_lingbot_exact_images_use_resized_uint8_batch() -> None:
    from torchvision.transforms.v2 import Resize

    frames = [
        np.array(
            [[[0 + offset, 11 + offset, 20 + offset], [31 + offset, 40 + offset, 51 + offset]],
             [[60 + offset, 71 + offset, 80 + offset], [91 + offset, 100 + offset, 111 + offset]]],
            dtype=np.uint8,
        )
        for offset in (0, 20, 40)
    ]
    image_processor = _ImageProcessor()
    policy = Lingbot2Policy.__new__(Lingbot2Policy)
    policy._processor = SimpleNamespace(image_processor=image_processor)
    policy._n_cameras = 3
    policy._image_size = 3
    policy._preproc_mode = "exact"

    pixels, grids = policy._encode_images(frames)

    resize = Resize((3, 3), antialias=True)
    resized = [
        resize(torch.from_numpy(frame).permute(2, 0, 1).contiguous().float())
        .permute(1, 2, 0)
        for frame in frames
    ]
    expected_inputs = [value.clamp(0, 255).to(torch.uint8).numpy() for value in resized]
    assert any(not torch.equal(value, value.to(torch.uint8).float()) for value in resized)
    assert len(image_processor.inputs) == 1
    assert len(image_processor.inputs[0]) == 3
    for actual, expected in zip(image_processor.inputs[0], expected_inputs, strict=True):
        assert actual.dtype == np.uint8
        np.testing.assert_array_equal(actual, expected)
    expected_pixels = torch.from_numpy(np.stack(expected_inputs)).reshape(3, 9, 3).float()
    assert torch.equal(pixels, expected_pixels)
    assert grids.tolist() == [[1, 3, 3], [1, 3, 3], [1, 3, 3]]
