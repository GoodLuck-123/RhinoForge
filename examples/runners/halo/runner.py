"""Public halo model construction and input binding."""
from __future__ import annotations


from examples.common.inputs import _load_mapping, _controlled_checkpoint as _checkpoint

def _halo(runner):
    from rpu_backend.adapters.halo import HaloConfig, build_halo_action_expert

    config = runner.source_config
    fixture = _load_mapping(config["input"], "fixture")
    values = fixture.get("kwargs", fixture)
    cfg = HaloConfig()
    positions = values["packed_position_ids"]
    if positions.unique().numel() != 1:
        raise ValueError("HALO single action-step requires one constant packed_position_ids value")
    if (values["packed_text_indexes"].tolist() != list(cfg.text_rows)
            or values["packed_action_token_indexes"].tolist() != list(cfg.action_rows)
            or int(values["key_values_lens"][0]) != cfg.prefix_len):
        raise ValueError("HALO fixture text/action rows or prefix length do not match the single-step profile")
    cache = values["past_key_values"]
    if not isinstance(cache, dict):
        raise ValueError("HALO fixture past_key_values must be a tensor-only key_cache/value_cache mapping")
    checkpoint = _checkpoint(config) / "halo_robotwin/finetune_ckpt/cot_aug_try3/0100000/ema.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    policy = build_halo_action_expert(
        checkpoint, rope_position=int(positions.flatten()[0]),
        rpu_execution=config.get("rpu_execution"),
    )
    runner.owner = policy
    policy.insert_prefix(
        [cache["key_cache"][i] for i in range(cfg.num_layers)],
        [cache["value_cache"][i] for i in range(cfg.num_layers)],
    )
    return lambda: policy.step(values["x_t"], values["timestep"],
                               values["packed_action_position_ids"], values["packed_text_ids"])


prepare = _halo
