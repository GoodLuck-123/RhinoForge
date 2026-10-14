from __future__ import annotations

import pytest
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from rpu_backend.adapters.navdp.runtime import (
    ACTION_DIM,
    DDPM_STEPS,
    NavdpRuntime,
    PREDICT_SIZE,
)


def _draw(generator: torch.Generator, sample_num: int = 2) -> torch.Tensor:
    return torch.randn(
        sample_num, PREDICT_SIZE, ACTION_DIM, generator=generator
    )


@pytest.mark.parametrize("explicit_init", [False, True])
def test_navdp_host_loop_uses_independent_initial_and_step_generators(monkeypatch, explicit_init):
    monkeypatch.delenv("RPU_NAVDP_DENOISE_UNROLL", raising=False)
    runtime = NavdpRuntime.__new__(NavdpRuntime)
    runtime._require_execution_session = lambda: None
    runtime._denoise_unroll = False
    seen = []
    def predict(action, *args, **kwargs):
        seen.append(action.clone())
        return torch.zeros_like(action)
    runtime._predict_noise = predict
    goal = torch.zeros(1, 1, 384)
    rgbd = torch.zeros(1, 32, 384)
    supplied = torch.full((2, PREDICT_SIZE, ACTION_DIM), 0.25) if explicit_init else None
    original = supplied.clone() if explicit_init else None
    actual = runtime.predict_action(goal, rgbd, sample_num=2, seed=17, init_noise=supplied)

    scheduler = DDPMScheduler(num_train_timesteps=DDPM_STEPS,
        beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type="epsilon")
    scheduler.set_timesteps(DDPM_STEPS)
    expected = supplied.clone() if explicit_init else _draw(torch.Generator().manual_seed(17))
    assert torch.equal(seen[0], expected)
    generator = torch.Generator().manual_seed(17)
    for timestep in scheduler.timesteps:
        expected = scheduler.step(model_output=torch.zeros_like(expected), timestep=timestep,
                                  sample=expected, generator=generator).prev_sample
    assert torch.equal(actual, expected)
    if explicit_init:
        assert torch.equal(supplied, original)


def test_navdp_repeat_calls_restart_both_streams(monkeypatch):
    monkeypatch.delenv("RPU_NAVDP_DENOISE_UNROLL", raising=False)
    runtime = NavdpRuntime.__new__(NavdpRuntime)
    runtime._require_execution_session = lambda: None
    runtime._denoise_unroll = False
    runtime._predict_noise = lambda action, *args, **kwargs: torch.zeros_like(action)
    args = (torch.zeros(1, 1, 384), torch.zeros(1, 32, 384))
    assert torch.equal(runtime.predict_action(*args, sample_num=2, seed=17),
                       runtime.predict_action(*args, sample_num=2, seed=17))
