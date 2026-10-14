"""Cold public-profile boundaries added with the runtime refresh."""
from types import SimpleNamespace

import pytest
import torch



@pytest.mark.parametrize("name", [
    "_rpu_decoder_topology", "_rpu_vision_num_cores", "_rpu_qwen3vl_vision_num_cores",
])
def test_installed_topology_is_admitted_and_remains_immutable(name):
    from rpu_backend.runtime import RPUConfigError
    from rpu_backend.runtime.hw_attrs import (
        install_hw_attr_validator, validate_postinstall, validate_preinstall,
    )
    owner = torch.nn.Linear(2, 2)
    validate_preinstall(owner)
    setattr(owner, name, (8, 8) if name.endswith("topology") else 8)
    validate_postinstall(owner)
    install_hw_attr_validator(owner)
    with pytest.raises(RPUConfigError, match="monotonic"):
        setattr(owner, name, 4)
    with pytest.raises(RPUConfigError, match="Unknown hardware attribute"):
        owner._rpu_unregistered_topology = 4


def test_moe_owner_state_is_admitted_after_install():
    from rpu_backend.runtime.hw_attrs import install_hw_attr_validator, validate_postinstall
    owner = torch.nn.Linear(2, 2)
    owner._rpu_qwen3_5_moe_text_install_started = True
    owner._rpu_qwen3_5_moe = SimpleNamespace(handle=1)
    validate_postinstall(owner)
    install_hw_attr_validator(owner)
    owner._rpu_qwen3_5_moe = None


@pytest.mark.parametrize("kwargs,pixels,error,message", [
    ({"_rpu_temporal_num_frames": 2}, (4, 8), ValueError, "whole number of frame groups"),
    ({"_rpu_camera_batch_count": 2}, (4, 8), ValueError, "K1 B2 camera group is unavailable"),
    ({}, (2, 3, 32, 32), RuntimeError, "compact G0.5 vision input requires"),
])
def test_qwen35_rejects_invalid_packed_inputs_before_dispatch(kwargs, pixels, error, message):
    from rpu_backend.adapters.qwen3_5 import vision
    owner = SimpleNamespace(_rpu_vision_handle=1, _rpu_vision_spatial_merge_size=2,
                            _rpu_vision_num_layers=0, _rpu_vision_kv_cache=object())
    before = vars(owner).copy()
    with pytest.raises(error, match=message):
        vision._rpu_vision_forward(owner, torch.empty(pixels), torch.tensor([[1, 2, 2]]), **kwargs)
    assert vars(owner) == before


def test_qwen35_dense_precision_tracks_only_cold_settings():
    from rpu_backend.adapters.qwen3_5 import vision
    assert vision._qwen35_vision_precision({}) == (False, False)
    assert vision._qwen35_vision_precision({"vision": {"linear_acc32": True}}) == (True, True)




def test_moe_direct_loader_forwards_remote_code_policy(monkeypatch):
    from rpu_backend.adapters.qwen3_5_moe import loader
    class ReachedConfig(RuntimeError):
        pass
    def config_io(path, **kwargs):
        assert path == "unread-checkpoint" and kwargs["trust_remote_code"] is True
        raise ReachedConfig()
    monkeypatch.setattr(loader.AutoConfig, "from_pretrained", config_io)
    with pytest.raises(ReachedConfig):
        loader.load_qwen3_5_moe_model("unread-checkpoint", trust_remote_code=True)


def test_wall_public_processor_expands_token_ids_and_falls_back_on_marker_mismatch(monkeypatch):
    from rpu_backend.adapters.wall_oss.runtime import WallOssVLA
    class Processor:
        image_token_id = 42
        image_processor = SimpleNamespace(merge_size=2)
        calls = 0
        def tokenizer(self, texts, **kwargs):
            assert texts == ["image request"]
            return {"input_ids": torch.tensor([[7, 42, 8]])}
        def __call__(self, **kwargs):
            self.calls += 1
            return dict(input_ids=torch.tensor([[99]]), pixel_values=torch.ones(1), image_grid_thw=torch.tensor([[1, 2, 4]]))
    owner = SimpleNamespace(_processor=Processor(), _fast_processor=True, _fast_proc_ok=True,
                            _image_preproc=lambda images: (torch.ones(1), torch.tensor([[1, 2, 4]])))
    ids, _, _ = WallOssVLA._process_inputs(owner, "image request", [])
    assert ids.tolist() == [[7, 42, 42, 8]] and owner._processor.calls == 0
    owner._image_preproc = lambda images: (torch.ones(1), torch.tensor([[1, 2, 4], [1, 2, 4]]))
    with pytest.warns(UserWarning, match="marker count mismatch"):
        ids, _, _ = WallOssVLA._process_inputs(owner, "image request", [])
    assert ids.tolist() == [[99]] and owner._fast_proc_ok is False and owner._processor.calls == 1
