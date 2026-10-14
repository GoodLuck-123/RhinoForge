"""Pure CPU checks for checkpoint-defined flow; no backend/device import."""
import importlib.util
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import torch


SPEC = importlib.util.spec_from_file_location(
    "rhino_action_host", Path(__file__).resolve().parents[1]
    / "python/rpu_backend/adapters/rhinovla/action.py")
action = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(action)


@pytest.mark.parametrize("direction, start, sign", [
    ("legacy_ascending", 0.0, 1), ("official_descending", 1.0, -1),
])
def test_schedule_preserves_upstream_float32_recurrence(direction, start, sign):
    times, dt, words = action.denoise_schedule(10, direction)
    expected = []
    time = torch.full((1,), start, dtype=torch.float32)
    for _ in range(10):
        expected.append(time.item())
        time = time + sign / 10
        time = time.clamp(min=0.0) if sign < 0 else time.clamp(max=1.0)
    assert times == tuple(expected)
    assert dt == sign / 10
    assert words == action.denoise_schedule(10, direction)[2]
    assert words != action.denoise_schedule(9, direction)[2]
    assert words != action.denoise_schedule(10, (
        "legacy_ascending" if sign < 0 else "official_descending"))[2]
    vectorized = start + sign * torch.arange(10, dtype=torch.float32) / 10
    assert not torch.equal(torch.tensor(times), vectorized)


@pytest.mark.parametrize("steps", [0, -1, True, 1.5])
def test_invalid_schedule_fails_before_execution(steps):
    with pytest.raises(ValueError, match="positive integer"):
        action.denoise_schedule(steps)


def test_official_loop_matches_source_order_mask_and_rope_delta():
    seen_times, seen_inputs = [], []
    def embed(state, x, time):
        assert torch.equal(state, torch.tensor([[0.5, 0.0]]))
        seen_times.append(time.item())
        seen_inputs.append(x.clone())
        return x, torch.ones(x.shape[:2], dtype=torch.bool), time
    def expert(embeds, **kwargs):
        assert torch.equal(kwargs["position_ids"], torch.tensor([[[2, 3]]]).expand(3, 1, 2))
        assert kwargs["prefix_rope_deltas"].item() == -2
        return torch.full_like(embeds, kwargs["adarms_cond"].item() + 0.25)
    io = SimpleNamespace(embed_suffix=embed, decode_actions=lambda x: x)
    mask = torch.tensor([[[1.0, 0.5], [1.0, 0.0]]])
    noise = torch.ones(1, 2, 2)
    actual, velocities = action.rpu_denoise(
        expert, io, None, prefix_key_values=[], prefix_mask=torch.ones(1, 4, dtype=torch.bool),
        state=torch.ones(1, 2), state_mask=torch.tensor([[0.5, 0.0]]),
        action_mask=mask, x0=noise, steps=10,
        flow_direction="official_descending", prefix_rope_deltas=torch.tensor([[-2]]),
        return_velocities=True)
    expected = noise * mask
    time = torch.ones(1)
    for index in range(10):
        assert torch.equal(seen_inputs[index], expected)
        velocity = torch.full_like(noise, time.item() + 0.25)
        assert torch.equal(velocities[index], velocity)  # Do not mask the velocity twice.
        expected = (expected - 0.1 * velocity) * mask
        time = (time - 0.1).clamp(min=0.0)
    assert torch.equal(actual, expected)
    assert tuple(seen_times) == action.denoise_schedule(10, "official_descending")[0]


def test_mask_projection_broadcast_and_required_masks():
    io = SimpleNamespace(state_proj=object(), embed_suffix=lambda *args: (
        torch.zeros(1, 3, 2), torch.ones(1, 3, dtype=torch.bool), None))
    projectors = {"state_mask_proj": torch.nn.Identity(), "action_mask_proj": torch.nn.Identity()}
    sm, am = torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])
    embeds, _, _ = action.build_suffix(io, projectors, None, None, None, sm, am)
    assert torch.equal(embeds, torch.tensor([[[1., 0.], [0., 1.], [0., 1.]]]))
    with pytest.raises(ValueError, match="requires state_mask"):
        action.build_suffix(io, projectors, None, None, None, None, am)
    with pytest.raises(ValueError, match="requires action_mask"):
        action.build_suffix(io, projectors, None, None, None, sm, None)
    with pytest.raises(ValueError, match="tokens"):
        action.build_suffix(io, projectors, None, None, None, sm, torch.ones(1, 4, 2))


def test_rope_delta_validation():
    pm, sm = torch.ones(1, 4, dtype=torch.bool), torch.ones(1, 2, dtype=torch.bool)
    for delta in (True, 0.5, torch.tensor([[0.5]])):
        with pytest.raises(TypeError):
            action.suffix_position_ids(pm, sm, delta)
    with pytest.raises(ValueError, match="non-negative"):
        action.suffix_position_ids(pm, sm, -5)
    with pytest.raises(ValueError, match="shape"):
        action.suffix_position_ids(pm, sm, torch.zeros(1, 2, dtype=torch.long))


def test_native_gated_residual_off_and_on_are_valid_positive_selectors(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/fused/rpu_rhino_vla_model.cpp").read_text()
    helper = re.search(
        r"constexpr int64_t rhino_gated_residual_selector\(bool no_sub\) \{[^}]+\}", source
    ).group()
    assert source.count("rhino_gated_residual_selector(") == 3
    assert "rhino_gated_residual_selector(gated_no_sub)" in source
    assert "rhino_gated_residual_selector(cold_config_.gated_no_sub)" in source
    cpp = tmp_path / "gated_selector.cpp"
    cpp.write_text("#include <cstdint>\n" + helper + "\n"
                   "static_assert(rhino_gated_residual_selector(false) == 1);\n"
                   "static_assert(rhino_gated_residual_selector(true) == 2);\nint main() {}\n")
    compiler = shutil.which("g++")
    assert compiler
    subprocess.run([compiler, "-std=c++17", "-fsyntax-only", str(cpp)], check=True)


def test_native_rhino_precision_identity_inspects_real_weights_not_profile_labels():
    source = (Path(__file__).resolve().parents[1] / "src/fused/rpu_rhino_vla_model.cpp").read_text()
    start = source.index("std::vector<int64_t> kvinsert_cost_weight_identity() const override")
    body = source[start:source.index("// ========================================================================", start)]
    assert "if (layer_weights_.empty()) return {};" in body
    fields = source[source.index("struct LayerWeights {"):source.index("RhinoVLAModel() = default;")]
    for declaration in re.findall(r"at::Tensor ([^;]+);", fields):
        for name in declaration.split(","):
            assert f"&layer.{name.strip()}" in body
    for name in ("action_in_w_", "action_out_w_", "state_mask_w_", "time_proj_table_",
                 "adarms_pair_table_", "adarms_final_table_"):
        assert f"&{name}" in body
    assert body.count("append_kvinsert_cost_tensor_identity(identity, *tensor)") == 2
    assert "get_chunk_size_override" not in body
    assert "cold_config_.gated_no_sub" in body and "static_cast<int64_t>(dt_.x)" in body
