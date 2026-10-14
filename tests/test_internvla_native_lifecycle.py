import sys
from types import ModuleType, SimpleNamespace

import pytest

from rpu_backend.adapters.internvla_n1 import rgbd_rpu


def test_rgbd_composite_build_rolls_back_completed_handles_in_reverse(
    monkeypatch, tmp_path
):
    events = []

    class Encoder:
        def __init__(self):
            self.rgb_model = SimpleNamespace()
            self.depth_model = SimpleNamespace()

        def float(self):
            return self

    rgbd_module = ModuleType("rpu_backend.adapters.internvla_n1.rgbd")
    rgbd_module.build_rgbd_encoder = lambda *args, **kwargs: Encoder()
    monkeypatch.setitem(
        sys.modules, "rpu_backend.adapters.internvla_n1.rgbd", rgbd_module
    )
    monkeypatch.setattr(rgbd_rpu, "require_controlled_evaluation", lambda _: None)
    monkeypatch.setattr(rgbd_rpu, "verify_asset_manifest", lambda *args: None)
    monkeypatch.setattr(
        rgbd_rpu, "_verify_dinov2_source_tree", lambda *args: tmp_path
    )

    names = iter(("rgb", "depth"))

    def make_tower(*args, **kwargs):
        name = next(names)
        return SimpleNamespace(
            destroy=lambda: events.append(("destroy", name))
        )

    def fail_former(*args, **kwargs):
        raise RuntimeError("synthetic former failure")

    monkeypatch.setattr(rgbd_rpu, "RGBDTowerRPU", make_tower)
    monkeypatch.setattr(rgbd_rpu, "FormerRPU", fail_former)

    with pytest.raises(RuntimeError, match="former failure"):
        rgbd_rpu.build_rgbd_encoder_rpu(
            tmp_path / "weights.safetensors", tmp_path
        )

    assert events == [("destroy", "depth"), ("destroy", "rgb")]
