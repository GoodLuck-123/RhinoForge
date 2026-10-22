"""Bounded v3 RTC request and timestep contracts (CPU, no device setup)."""
import torch

RTC_MAX_DELAY = 6
RTC_HORIZON = 30
RTC_ACTION_DIM = 96


def require_rtc_native():
    """Reject an older FP16-only native extension before installing weights."""
    for op in ("rhino_vla_set_denoise_loop_rtc_tables",
               "rhino_vla_set_denoise_loop_rtc_tables_w8a16",
               "rhino_vla_prepare_rtc", "rhino_vla_denoise_loop_rtc_forward"):
        if not hasattr(torch.ops.rpu, op):
            raise RuntimeError("RhinoVLA RTC requires a matching native RhinoForge build")


def validate_rtc_request(*, rtc, action_prefix=None, delay=None):
    if type(rtc) is not bool:
        raise TypeError("RhinoVLA rtc must be bool")
    if not rtc:
        if action_prefix is not None or delay is not None:
            raise ValueError("action_prefix/delay require a v3 RTC runtime (rtc=True)")
        return
    if action_prefix is None or delay is None:
        raise ValueError("RhinoVLA RTC requires both action_prefix and delay")
    if type(delay) is not int:
        raise TypeError("RhinoVLA RTC delay must be an integer, not bool or Tensor")
    if not 0 <= delay <= RTC_MAX_DELAY:
        raise ValueError("RhinoVLA RTC delay must be in [0, 6]")
    if (not isinstance(action_prefix, torch.Tensor)
            or action_prefix.device.type != "cpu"
            or not action_prefix.is_floating_point()):
        raise TypeError("RhinoVLA RTC action_prefix must be a floating CPU tensor")
    shape = tuple(action_prefix.shape)
    if len(shape) == 2:
        shape = (1, *shape)
    if (len(shape) != 3 or shape[0] != 1 or shape[2] != RTC_ACTION_DIM
            or not delay <= shape[1] <= RTC_HORIZON):
        raise ValueError("RhinoVLA RTC action_prefix requires [1,D,96] or [D,96], delay <= D <= 30")
    if not bool(torch.isfinite(action_prefix).all()):
        raise ValueError("RhinoVLA RTC action_prefix must be finite")


def validate_rtc_install(*, rtc, steps, flow_direction, expert_w8a16,
                         full_expert_w8a16, native_cold, action_cold):
    if type(rtc) is not bool:
        raise TypeError("RhinoVLA rtc must be bool")
    if not rtc:
        return
    if steps != 10 or flow_direction != "official_descending":
        raise ValueError("RhinoVLA RTC requires the v3 ten-step descending flow")
    if not action_cold["denoise_unroll"] or not native_cold.precompute_adarms:
        raise ValueError("RhinoVLA RTC requires denoise_unroll and PRECOMPUTE_ADARMS")
    for field in ("packed_qkv", "skip_adarms_gemv", "precompute_time_proj",
                  "fold_action_time_in"):
        if getattr(native_cold, field):
            raise ValueError(f"RhinoVLA RTC requires {field.upper()} disabled")


def normalize_rtc_request(*, rtc, action_prefix, delay, x0, state,
                          state_mask, action_mask):
    """Own a padded FP32 prefix and admit the upstream single-mask profile."""
    validate_rtc_request(rtc=rtc, action_prefix=action_prefix, delay=delay)
    if not rtc:
        return None
    shapes = {"x0": {(1, 30, 96)}, "state": {(1, 96), (1, 1, 96)},
              "state_mask": {(1, 96), (1, 1, 96)},
              "action_mask": {(1, 96), (1, 1, 96), (1, 30, 96)}}
    for name, value in (("x0", x0), ("state", state),
                        ("state_mask", state_mask), ("action_mask", action_mask)):
        if value is None and name.endswith("mask"):
            continue
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or not value.is_floating_point()):
            raise TypeError(f"RhinoVLA RTC {name} must be a floating CPU tensor")
        if tuple(value.shape) not in shapes[name] or not bool(torch.isfinite(value).all()):
            raise ValueError(f"RhinoVLA RTC {name} has invalid shape or nonfinite values")
        if name == "action_mask" and value.ndim == 3 and value.shape[1] == 30:
            if not torch.equal(value, value[:, :1].expand_as(value)):
                raise ValueError("RhinoVLA RTC requires one action mask shared across the horizon")
    prefix = torch.zeros((1, 30, 96), dtype=torch.float32)
    source = action_prefix.reshape(1, -1, 96).to(torch.float32)
    prefix[:, :source.shape[1]].copy_(source)
    if action_mask is not None:
        prefix.mul_(action_mask.reshape(1, -1, 96)[:, :1])
    if not bool(torch.isfinite(prefix.to(torch.float16)).all()):
        raise ValueError("RhinoVLA RTC masked action_prefix must be representable in FP16")
    return prefix


def rtc_condition_groups(times, embedding_fn, width):
    """[delay, step, state/prefix/suffix, H], matching upstream mean(emb)."""
    times = torch.as_tensor(times, dtype=torch.float32, device="cpu")
    groups = []
    for delay in range(RTC_MAX_DELAY + 1):
        token_times = times[:, None].expand(-1, RTC_HORIZON).clone()
        token_times[:, :delay] = 0.0
        embedded = embedding_fn(token_times.flatten(), width).reshape(
            len(times), RTC_HORIZON, width).to(torch.float32)
        zero = embedding_fn(torch.zeros_like(times), width).to(torch.float32)
        groups.append(torch.stack((embedded.mean(dim=1), zero, embedded[:, -1]), dim=1))
    return torch.stack(groups).contiguous()
