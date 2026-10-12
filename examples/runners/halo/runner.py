"""Public halo model construction and input binding."""
from __future__ import annotations


from examples.common.inputs import _load_mapping, _controlled_checkpoint as _checkpoint


def _validate_fixture(values, cfg):
    """Check the single-step input before constructing a device policy."""
    import torch

    if not isinstance(values, dict):
        raise ValueError("HALO fixture kwargs must be a tensor mapping")
    action_rows = len(cfg.action_rows)
    shapes = {
        "x_t": (action_rows, cfg.action_dim),
        "timestep": (action_rows,),
        "packed_action_position_ids": (action_rows,),
        "packed_text_ids": (len(cfg.text_rows),),
        "packed_position_ids": (cfg.chunk_size,),
        "packed_text_indexes": (len(cfg.text_rows),),
        "packed_action_token_indexes": (action_rows,),
        "key_values_lens": (1,),
    }
    for name, shape in shapes.items():
        value = values.get(name)
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or tuple(value.shape) != shape):
            raise ValueError(f"HALO fixture {name} must be a CPU tensor with shape {shape}")
        if name in {"x_t", "timestep"}:
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"HALO fixture {name} must contain finite floating-point values")
        elif value.dtype not in {torch.int32, torch.int64} or (value < 0).any():
            raise ValueError(f"HALO fixture {name} must contain non-negative integer indexes")
    if values["timestep"].unique().numel() != 1:
        raise ValueError("HALO single action-step requires one constant timestep value")
    if values["packed_position_ids"].unique().numel() != 1:
        raise ValueError("HALO single action-step requires one constant packed_position_ids value")
    if (values["packed_text_indexes"].tolist() != list(cfg.text_rows)
            or values["packed_action_token_indexes"].tolist() != list(cfg.action_rows)
            or int(values["key_values_lens"][0]) != cfg.prefix_len):
        raise ValueError("HALO fixture text/action rows or prefix length do not match the single-step profile")
    cache = values.get("past_key_values")
    if not isinstance(cache, dict):
        raise ValueError("HALO fixture past_key_values must be a tensor-only key_cache/value_cache mapping")
    layers = []
    shape = (cfg.prefix_len, cfg.num_kv_heads, cfg.head_dim)
    for name in ("key_cache", "value_cache"):
        bank = cache.get(name)
        if (not isinstance(bank, (dict, list, tuple, torch.Tensor))
                or (isinstance(bank, torch.Tensor) and bank.ndim == 0)
                or len(bank) != cfg.num_layers
                or (isinstance(bank, dict) and set(bank) != set(range(cfg.num_layers)))):
            raise ValueError(f"HALO fixture {name} must contain {cfg.num_layers} layers")
        rows = [bank[i] for i in range(cfg.num_layers)]
        for layer, value in enumerate(rows):
            if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                    or not value.is_floating_point() or tuple(value.shape) != shape):
                raise ValueError(
                    f"HALO fixture {name}[{layer}] must be a floating-point CPU tensor with shape {shape}"
                )
        layers.append(rows)
    return layers


def _halo(runner):
    from rpu_backend.adapters.halo import HaloConfig, build_halo_action_expert

    config = runner.source_config
    checkpoint = _checkpoint(config)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"HALO checkpoint must name a safetensors file: {checkpoint}")
    fixture = _load_mapping(config["input"], "fixture")
    values = fixture.get("kwargs", fixture)
    cfg = HaloConfig()
    k_layers, v_layers = _validate_fixture(values, cfg)
    positions = values["packed_position_ids"]
    policy = build_halo_action_expert(
        checkpoint, rope_position=int(positions.flatten()[0]),
        rpu_execution=config.get("rpu_execution"),
    )
    runner.owner = policy
    policy.insert_prefix(k_layers, v_layers)
    return lambda: policy.step(values["x_t"], values["timestep"],
                               values["packed_action_position_ids"], values["packed_text_ids"])


prepare = _halo
