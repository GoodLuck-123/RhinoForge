"""Cold options for the measured parcel-sorting Wall-OSS schedules."""

from dataclasses import dataclass
import os

from rpu_backend.runtime import rpu_env_bool


def _selector(name, choices):
    raw = os.environ.get(name, "0").strip() or "0"
    if raw not in {str(value) for value in choices}:
        raise ValueError(f"{name} accepts only {sorted(choices)}, got {raw!r}")
    return int(raw)


@dataclass(frozen=True)
class WallSortingConfig:
    fp16_mask: int = 0
    mlp_m224: bool = False
    silu_mul: bool = False
    action_reuse: bool = False
    prefill_row: bool = False
    prefill_mlp: bool = False
    prefill_paired: bool = False
    vision_row: bool = False

    def decoder_args(self):
        return (
            self.fp16_mask, self.mlp_m224, self.silu_mul, self.action_reuse,
            self.prefill_row, self.prefill_mlp, self.prefill_paired,
        )

    def vision_args(self):
        return self.fp16_mask, self.vision_row, self.silu_mul


def sorting_config_from_env():
    """Read once per cold builder; native handles own every subsequent decision."""
    mask = _selector("RPU_WALL_OSS_FP16_TRANSFER_MASK", range(8))
    reuse = _selector("RPU_WALL_OSS_W16_REUSE_MASK", (0, 8))
    row = _selector("RPU_WALL_OSS_W16_NEXT3_MASK", (0, 1))
    mlp = _selector("RPU_WALL_OSS_W16_NEXT4_MASK", range(4))
    return WallSortingConfig(
        fp16_mask=mask,
        mlp_m224=rpu_env_bool("RPU_WALL_OSS_FP16_MLP_M224"),
        silu_mul=rpu_env_bool("RPU_WALL_OSS_SORTING_SILU_MUL"),
        action_reuse=bool(reuse), prefill_row=bool(row),
        prefill_mlp=bool(mlp & 1), prefill_paired=bool(mlp & 2),
        vision_row=rpu_env_bool("RPU_WALL_OSS_VISION_ROW_WEIGHT_OUTER"),
    )
