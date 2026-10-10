import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("audio_benchmark", Path(__file__).parents[1]/"examples/benchmark_qwen3_audio.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_edit_counts_and_language_normalization():
    assert m.units("你好， 世界！１２", "Chinese") == list("你好世界12")
    assert m.units("DON’T Stop!", "English") == ["dont", "stop"]
    result = m.errors("a b c", "a x c y", "English")
    assert (result["substitutions"], result["deletions"], result["insertions"]) == (1, 0, 1)
    assert m.errors("a b", "a", "English")["deletions"] == 1
    assert m.errors("你好", "你好！", "Chinese")["exact"]
    assert m.errors("你好", "", "Chinese")["error_rate"] == 1
    with pytest.raises(ValueError):m.errors("!", "hello", "English")


def test_corpus_error_rate_is_weighted_by_reference_length():
    rows=[dict(score=m.errors("a b c d", "a b c d", "English")), dict(score=m.errors("e", "f", "English"))]
    result=m.score_summary(rows)
    assert result["error_rate"] == 0.2 and result["exact_sentence_rate"] == 0.5
    result=m.timing_summary([dict(seconds=2,audio_seconds=4),dict(seconds=8,audio_seconds=6)])
    assert result["aggregate_rtf"] == 1 and result["median_seconds"] == 5
