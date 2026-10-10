"""Explicit Pi input noise is validated and forwarded without diagnostic state."""
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.api.policy import Pi05Policy


def make_policy():
    calls = []
    policy = object.__new__(Pi05Policy)
    policy._rpu_ready = True
    policy._optimized_profile = None
    policy._rpu_execution = None
    def record(batch, **kwargs):
        calls.append(kwargs)
        return kwargs.get("noise")
    policy._lerobot_policy = SimpleNamespace(
        config=SimpleNamespace(chunk_size=50, max_action_dim=32),
        predict_action_chunk=record)
    policy._adapter = SimpleNamespace(prepare_graphs=record)
    return policy, calls


@pytest.mark.parametrize("method", ["prepare_graphs", "predict_action_chunk"])
def test_explicit_noise_preserves_input_and_uses_normal_policy_call(method):
    policy, calls = make_policy()
    noise = torch.randn(1, 50, 32)
    original = noise.clone()
    batch = {"observation.state": torch.ones(1, 32)}
    result = getattr(policy, method)(batch, noise=noise, num_steps=10)
    assert result is noise and calls[0]["noise"] is noise
    assert torch.equal(noise, original)


@pytest.mark.parametrize("method", ["prepare_graphs", "predict_action_chunk"])
@pytest.mark.parametrize("kind", ["shape", "dtype", "nan", "state", "object"])
def test_bad_noise_fails_before_policy_or_graph_mutation(method, kind):
    policy, calls = make_policy()
    noise = torch.zeros(1, 50, 32)
    batch = {"observation.state": torch.ones(1, 32)}
    if kind == "shape": noise = noise[:, :, :7]
    elif kind == "dtype": noise = noise.half()
    elif kind == "nan": noise[0, 0, 0] = float("nan")
    elif kind == "state": batch = {}
    elif kind == "object": noise = {"noise": noise}
    with pytest.raises(ValueError, match="noise"):
        getattr(policy, method)(batch, noise=noise, num_steps=10)
    assert not calls


@pytest.mark.parametrize("method", ["prepare_graphs", "predict_action_chunk"])
def test_omitted_noise_preserves_legacy_calls_without_config(method):
    policy, calls = make_policy()
    del policy._lerobot_policy.config
    getattr(policy, method)({}, noise=None)
    assert calls == (
        [{"num_steps": None, "precompute_adarms": False}]
        if method == "prepare_graphs" else [{}]
    )


class _GraphCache:
    def __init__(self, instruction_snapshot=None):
        self.frozen = False
        self.instruction_snapshot = instruction_snapshot or {}
        self.clear()

    def begin_warmup(self):
        self.frozen = False

    def clear(self):
        self.entries = self.replays = 0

    def call(self):
        if self.entries:
            self.replays += 1
        else:
            assert not self.frozen
            self.entries = 1

    def freeze(self):
        self.frozen = True

    def size(self):
        return self.entries

    def snapshot(self):
        return [SimpleNamespace(replay_count=self.replays, recapture_count=0,
                                **self.instruction_snapshot)]

    def cache_invariant_ok(self):
        return True


@pytest.mark.parametrize("explicit", [False, True])
def test_public_prepare_forwards_noise_through_build_warming_and_ready(
    monkeypatch, explicit,
):
    from rpu_backend.adapters.pi05 import Pi05Adapter, runtime

    monkeypatch.setattr(runtime, "_pi05_graph_runtime_profile", lambda *_: {
        "siglip_graph": True, "gemma_graph": True, "fused_denoise": True,
        "denoise_graph": True, "adarms_graph": False,
    })
    policy, calls = make_policy()
    model = torch.nn.Module()
    model.config = SimpleNamespace(num_inference_steps=10)
    model._rpu_fuse_validated = True
    admission = [
        dict(batch_snapshot_mode="auto", batch_snapshot_available=True,
             batch_snapshot_effective_contracts=3,
             batch_snapshot_fallback_reason=""),
        dict(batch_snapshot_mode="auto", batch_snapshot_available=False,
             batch_snapshot_effective_contracts=0,
             batch_snapshot_fallback_reason="untrusted_program:prefill"),
        dict(batch_snapshot_mode="off", batch_snapshot_available=False,
             batch_snapshot_effective_contracts=0,
             batch_snapshot_fallback_reason="policy_off"),
    ]
    caches = [_GraphCache(snapshot) for snapshot in admission]
    model.paligemma_with_expert = SimpleNamespace(
        paligemma=SimpleNamespace(model=SimpleNamespace(
            vision_tower=SimpleNamespace(_rpu_siglip_graph_cache=caches[0]),
            language_model=SimpleNamespace(_rpu_gemma_graph_cache=caches[1]),
        )),
        gemma_expert=SimpleNamespace(model=SimpleNamespace(
            _rpu_adarms_graph_cache=_GraphCache(),
        )),
    )
    model._rpu_fused_denoise_graph_cache = caches[2]

    def predict(batch, **kwargs):
        calls.append(kwargs)
        for cache in caches:
            cache.call()
        return torch.ones(1, 50, 7)

    policy._lerobot_policy.model = model
    policy._lerobot_policy.predict_action_chunk = predict
    adapter = object.__new__(Pi05Adapter)
    adapter._rpu_is_ready = True
    adapter._lerobot_policy = policy._lerobot_policy
    policy._adapter = adapter
    noise = torch.randn(1, 50, 32)
    original = noise.clone()
    batch = {"observation.state": torch.ones(1, 32)}
    try:
        result = policy.prepare_graphs(batch, **({"noise": noise} if explicit else {}))
        assert result["phase"] == "READY" and result["ready_replay_validated"]
        assert len(calls) == 3
        assert all(call["num_steps"] == 10 for call in calls)
        if explicit:
            assert all(call["noise"] is noise for call in calls)
        else:
            assert all("noise" not in call for call in calls)
        assert torch.equal(noise, original)
        assert all(cache.frozen and cache.replays == 2 for cache in caches)
        for name, snapshot in zip(("vision", "prefill", "denoise"), admission):
            assert result["ready_stats"][name]["instruction_snapshots"] == [snapshot]
    finally:
        runtime._PI05_PREPARED_GRAPH_PROFILES.pop(model, None)
