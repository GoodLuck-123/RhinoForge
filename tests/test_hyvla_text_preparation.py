"""Text preparation keeps observation tensors and excludes padded language rows."""
import importlib.util
from pathlib import Path
import re

import pytest
import torch


@pytest.fixture
def prep():
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_hyvla_text_batches.py"
    spec = importlib.util.spec_from_file_location("hyvla_text_preparation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WordTokenizer:
    def __call__(self, texts, *, max_length, **kwargs):
        words = re.findall(r"\w+", texts[0].split("<｜hy_Assistant｜>")[0])
        valid = min(len(words) + 1, max_length)
        ids = torch.zeros(1, max_length, dtype=torch.long)
        ids[0, :valid] = torch.arange(1, valid + 1)
        return {"input_ids": ids, "attention_mask": ids != 0}


@pytest.mark.parametrize("length", [16, 32, 48, 64])
def test_requested_length_uses_words_and_rebuilds_attention(prep, length):
    source = {"images": object(), "noise": torch.ones(1, 50, 32), "state": torch.zeros(1, 32)}
    sentence = " ".join(f"word{i}" for i in range(90))
    result = prep.make_text_batch(source, WordTokenizer(), sentence, length)
    obs = result["observation"]
    assert result["instruction"].split() == sentence.split()[:length - 1]
    assert obs["images"] is source["images"]
    assert obs["noise"] is source["noise"]
    assert obs["state"] is source["state"]
    assert obs["n_valid"] == 176 + length
    assert obs["lang_tokens"].count_nonzero() == length
    assert obs["prefill_mask"].shape == (240, 240)
    assert not bool(obs["prefill_mask"][176 + length:, :].any())
    assert not bool(obs["denoise_mask"][:, 176 + length:240].any())
    assert torch.equal(obs["suffix_pos"], torch.arange(176 + length, 227 + length))


def test_short_instruction_is_not_filled_by_repeated_ids(prep):
    with pytest.raises(ValueError, match="no round-trippable prefix"):
        prep.instruction_with_tokens(WordTokenizer(), "walk forward", 16)


def test_unsupported_length_is_rejected(prep):
    with pytest.raises(ValueError, match="text length"):
        prep.make_text_batch({}, WordTokenizer(), "walk forward", 24)
