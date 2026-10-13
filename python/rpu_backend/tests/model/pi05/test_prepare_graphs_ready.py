'Prepare graphs ready device regression.'
from __future__ import annotations

import os

import pytest
import torch


pytestmark = pytest.mark.skipif(
    os.environ.get("PI05_HARDWARE") != "1",
    reason="Requires device hardware; set PI05_HARDWARE=1 and run with sudo.",
)


def _dummy_batch(config) -> dict:
    image_keys = list(config.image_features)
    height, width = map(int, config.image_resolution)
    batch = {
        key: torch.zeros(1, 3, height, width, dtype=torch.float32)
        for key in image_keys
    }
    for key in image_keys:
        batch[f"{key}_padding_mask"] = torch.ones(1, dtype=torch.bool)
    batch["observation.state"] = torch.zeros(
        1, int(config.max_state_dim), dtype=torch.float32
    )
    batch["observation.language.tokens"] = torch.arange(
        2, 34, dtype=torch.int64
    ).unsqueeze(0)
    batch["observation.language.attention_mask"] = torch.ones(
        1, 32, dtype=torch.bool
    )
    return batch


def _graph_caches(policy) -> dict:
    model = policy._lerobot_policy.model
    pawe = model.paligemma_with_expert
    vision = pawe.paligemma.model.vision_tower
    vision = vision.vision_model if hasattr(vision, "vision_model") else vision
    return {
        "vision": vision._rpu_siglip_graph_cache,
        "prefill": pawe.paligemma.model.language_model._rpu_gemma_graph_cache,
        "denoise": model._rpu_fused_denoise_graph_cache,
        "safety_adarms": pawe.gemma_expert.model._rpu_adarms_graph_cache,
    }


def _invariants(caches) -> dict:
    return {
        name: {
            "phase": cache.phase,
            "entries": cache.size(),
            "recaptures": sum(
                int(entry.recapture_count) for entry in cache.snapshot()
            ),
        }
        for name, cache in caches.items()
    }


def test_pi05_prepare_graphs_ready_lookup_only_and_bit_exact():
    import rpu_backend  # noqa: F401
    from rpu_backend.api import Pi05Policy
    from rpu_backend.model_registry import model_path

    model_name = os.environ.get(
        "PI05_READY_MODEL", "pi05-libero-finetuned-w8a16-vlm-expert"
    )
    steps = int(os.environ.get("PI05_READY_STEPS", "10"))
    chunk_size = int(os.environ.get("PI05_READY_CHUNK_SIZE", "400"))

    policy = Pi05Policy.from_pretrained(
        str(model_path(model_name)), dtype=torch.float16
    ).to("rpu")
    lerobot_policy = policy._lerobot_policy
    model = lerobot_policy.model
    vlm = model.paligemma_with_expert.paligemma.model.language_model
    vlm._rpu_chunk_size = chunk_size
    vlm._rpu_spm_mode = False
    model._rpu_noise_seed = 0
    batch = _dummy_batch(lerobot_policy.config)

    prepared = policy.prepare_graphs(batch, num_steps=steps)
    assert prepared["phase"] == "READY"
    assert prepared["ready_replay_validated"]
    assert prepared["action_bit_equal"]
    assert prepared["build_replay_bit_equal"]
    assert prepared["build_replay_max_abs"] == 0.0
    assert prepared["graph_counts"]["vision"] == 1
    assert prepared["graph_counts"]["prefill"] == 1
    assert prepared["graph_counts"]["denoise"] == 1
    assert all(
        prepared["ready_stats"][name]["replays"] >= 2
        for name in ("vision", "prefill", "denoise")
    )
    assert all(
        stats["entries"] == 1 and stats["recaptures"] == 0
        for stats in prepared["ready_stats"].values()
    )
    assert all(
        stats["invariant_ok"]
        for stats in prepared["ready_stats"].values()
    )

    caches = {
        name: cache
        for name, cache in _graph_caches(policy).items()
        if cache.size() > 0
    }
    assert all(cache.phase == "READY" for cache in caches.values())

    action_a = policy._forward_rpu(
        batch, return_chunk=True, num_steps=steps
    ).detach().cpu().clone()
    before_miss = _invariants(caches)
    miss_steps = steps - 1 if steps > 1 else steps + 1
    with pytest.raises(RuntimeError, match="prepared READY profile"):
        policy._forward_rpu(
            batch, return_chunk=True, num_steps=miss_steps
        )
    after_miss = _invariants(caches)
    assert after_miss == before_miss

    action_b = policy._forward_rpu(
        batch, return_chunk=True, num_steps=steps
    ).detach().cpu().clone()
    after_ready = _invariants(caches)
    assert {
        name: (item["phase"], item["entries"], item["recaptures"])
        for name, item in after_ready.items()
    } == {
        name: (item["phase"], item["entries"], item["recaptures"])
        for name, item in before_miss.items()
    }
    assert torch.equal(action_a, action_b)
    print(
        {
            "prepared": prepared,
            "miss_steps": miss_steps,
            "before_miss": before_miss,
            "after_ready": after_ready,
            "action_bit_equal": bool(torch.equal(action_a, action_b)),
        }
    )
