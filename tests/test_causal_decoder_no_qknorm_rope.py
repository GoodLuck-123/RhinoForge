"""Host regression for head boundaries in both no-QK-norm schedules."""
import math
from pathlib import Path
import re

import pytest


SOURCE = (Path(__file__).resolve().parents[1] /
          "src/fused/rpu_qwen3_model.h").read_text()


def _rope(values, heads, width, position):
    result = []
    for head in range(heads):
        row = values[head * width:(head + 1) * width]
        half = width // 2
        for index in range(width):
            phase = position / (10000 ** (2 * (index % half) / width))
            rotated = -row[index + half] if index < half else row[index - half]
            result.append(row[index] * math.cos(phase) + rotated * math.sin(phase))
    return result


@pytest.mark.parametrize("schedule", ["kv_first", "sequential"])
@pytest.mark.parametrize("kv_heads", [8, 16])
@pytest.mark.parametrize("position", [1, 7])
@pytest.mark.parametrize("codec_mode", [False, True])
def test_no_qknorm_key_rope_is_scoped_to_codec(schedule, kv_heads, position, codec_mode):
    # Exercise the arguments emitted by each real source branch. In particular,
    # two heads must not become one doubled-width head (including its RoPE table).
    calls = re.findall(r'emit_qk_rope\("k", "k",\s*([^,]+),\s*([^,]+), seq_len,', SOURCE)
    assert len(calls) == 2
    head_expr, width_expr = calls[0 if schedule == "kv_first" else 1]
    batch = 1 if schedule == "kv_first" else 2
    env = dict(nkv=kv_heads, tp=8, hd=64, bs=batch,
               local_kv_dim=kv_heads * 64 // 8)
    def evaluate(expression):
        if "?" in expression:
            condition, branches = expression.split("?")
            assert condition.strip() == "tts_codec_mode_"
            expression = branches.split(":")[0 if codec_mode else 1]
        return int(eval(expression, {"__builtins__": {}}, env))

    heads, width = evaluate(head_expr), evaluate(width_expr)
    per_core_heads = batch * kv_heads // 8
    values = [math.sin(index * 0.37) for index in range(per_core_heads * 64)]
    actual = _rope(values, heads, width, position)
    expected = (_rope(values, per_core_heads, 64, position) if codec_mode else
                _rope(values, batch, kv_heads * 64 // 8, position))
    if codec_mode:
        assert width // 2 == 32  # The codec checkpoint table has 32 columns.
    assert actual == pytest.approx(expected, abs=1e-12)


def test_codec_mode_is_explicit_and_default_off():
    assert "bool tts_codec_mode_ = false;" in SOURCE
    setter = SOURCE.split("void set_tts_codec_mode()", 1)[1].split("\n    }", 1)[0]
    assert "num_layers() == 8" in setter and "hidden_size() == 512" in setter
    assert "!has_qk_norm_" in setter
    assert setter.index("TORCH_CHECK") < setter.index("tts_codec_mode_ = true")
    assert "invalidate_model_state();" in setter
