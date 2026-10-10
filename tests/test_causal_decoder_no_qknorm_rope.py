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
def test_no_qknorm_key_rope_preserves_individual_heads(schedule, kv_heads, position):
    # Exercise the arguments emitted by each real source branch. In particular,
    # two heads must not become one doubled-width head (including its RoPE table).
    calls = re.findall(r'emit_qk_rope\("k", "k", ([^,]+), ([^,]+), seq_len,', SOURCE)
    assert len(calls) == 2
    head_expr, width_expr = calls[0 if schedule == "kv_first" else 1]
    batch = 1 if schedule == "kv_first" else 2
    env = dict(nkv=kv_heads, tp=8, hd=64, bs=batch,
               local_kv_dim=kv_heads * 64 // 8)
    heads = int(eval(head_expr, {"__builtins__": {}}, env))
    width = int(eval(width_expr, {"__builtins__": {}}, env))
    per_core_heads = batch * kv_heads // 8
    values = [math.sin(index * 0.37) for index in range(per_core_heads * 64)]
    actual = _rope(values, heads, width, position)
    expected = _rope(values, per_core_heads, 64, position)
    assert width // 2 == 32  # The installed checkpoint table has 32 columns.
    assert actual == pytest.approx(expected, abs=1e-12)
