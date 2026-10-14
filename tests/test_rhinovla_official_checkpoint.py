"""CPU-only contracts for the explicit official plain-weight action loader."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


PATH = (Path(__file__).resolve().parents[1]
        / "python/rpu_backend/adapters/rhinovla/checkpoint.py")


class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Transformer(nn.Module):
    def __init__(self, config, qwen_rotary_emb):
        super().__init__()
        layer = nn.Module()
        layer.self_attn = nn.ModuleDict({
            "q_proj": nn.Linear(4, 4, bias=False),
            "k_proj": nn.Linear(4, 2, bias=False),
            "q_norm": nn.LayerNorm(2, bias=False),
        })
        layer.mlp = nn.ModuleDict({"gate_proj": nn.Linear(4, 6, bias=False)})
        self.layers = nn.ModuleList([layer])


class IO(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.action_in_proj = nn.Linear(3, 4)
        self.state_proj = nn.Linear(5, 4)


@pytest.fixture
def loader():
    spec = importlib.util.spec_from_file_location("rhino_checkpoint_host", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def plain_case(loader, monkeypatch, tmp_path):
    checkpoint = tmp_path / "official.ckpt"
    checkpoint.touch()
    root = tmp_path / "source"
    source = root / "rhinovla/model/modules/action_expert.py"
    source.parent.mkdir(parents=True)
    source.write_text("# reviewed unit-test source\n")
    expert, io = Transformer(None, None), IO(None)
    state = {f"action_expert.{k}": v for k, v in expert.state_dict().items()}
    state.update({f"action_io.{k}": v for k, v in io.state_dict().items()})
    for name, dim in (("state", 5), ("action", 3)):
        state.update({f"{name}_mask_proj.{k}": v
                      for k, v in nn.Linear(dim, 4).state_dict().items()})
    module = SimpleNamespace(
        RhinoActionExpertConfig=Config,
        _RhinoActionTransformer=Transformer,
        RhinoActionIO=IO,
    )
    monkeypatch.setattr(torch, "load", lambda *a, **kw: state)
    monkeypatch.setattr(loader, "_load_official_expert_module", lambda root: module)
    monkeypatch.setattr(loader, "_create_rotary", lambda **kw: object())
    return checkpoint, root, state


def call(loader, case, **kwargs):
    checkpoint, root, _ = case
    return loader.load_official_action_bundle(
        checkpoint, model_root=root, action_horizon=30, **kwargs,
    )


def test_plain_weights_are_exact_and_keep_transformer_contract(loader, plain_case):
    with pytest.warns(UserWarning, match="executes Python source"):
        expert, io, masks, config = call(loader, plain_case, trust_model_code=True)
    assert isinstance(expert, Transformer)
    assert config.use_mask_condition is True
    assert config.action_horizon == 30
    assert config.depth == 1
    assert not any(m.training for m in (expert, io, masks))
    for key, value in expert.state_dict().items():
        assert torch.equal(value, plain_case[2][f"action_expert.{key}"])
    for key, value in io.state_dict().items():
        assert torch.equal(value, plain_case[2][f"action_io.{key}"])


def test_plain_loader_still_requires_reviewed_source_opt_in(loader, plain_case):
    with pytest.raises(PermissionError, match="trust_model_code=True"):
        call(loader, plain_case)


@pytest.mark.parametrize("key", ["lora_a", "lora_b", "base.weight"])
def test_mixed_lora_never_silently_drops_keys(loader, plain_case, key):
    plain_case[2][f"action_expert.layers.0.mlp.gate_proj.{key}"] = torch.ones(1)
    with pytest.raises(ValueError, match="plain weights"):
        call(loader, plain_case, trust_model_code=True)


@pytest.mark.parametrize("prefix", [
    "action_expert", "action_io", "state_mask_proj", "action_mask_proj",
])
def test_unknown_component_keys_are_strict_errors(loader, plain_case, prefix):
    plain_case[2][f"{prefix}.unreviewed.weight"] = torch.ones(1)
    with pytest.warns(UserWarning), pytest.raises(RuntimeError, match="Unexpected key"):
        call(loader, plain_case, trust_model_code=True)


@pytest.mark.parametrize("bad", ["missing", "shape"])
def test_component_state_is_loaded_strictly(loader, plain_case, bad):
    key = "action_io.action_in_proj.bias"
    if bad == "missing":
        del plain_case[2][key]
    else:
        plain_case[2][key] = torch.ones(7)
    with pytest.warns(UserWarning), pytest.raises(RuntimeError):
        call(loader, plain_case, trust_model_code=True)


def test_plain_does_not_change_legacy_default(loader, plain_case):
    checkpoint, root, _ = plain_case
    with pytest.raises(ValueError, match="gate_proj.base.weight"):
        loader.load_action_bundle(
            checkpoint, model_root=root, action_horizon=30, trust_model_code=True,
        )


def test_official_source_cache_uses_content_identity(loader, tmp_path):
    source = tmp_path / "rhinovla/model/modules/action_expert.py"
    source.parent.mkdir(parents=True)
    source.write_text("PROFILE = 'first'\n")
    first = loader._load_official_expert_module(tmp_path)
    source.write_text("PROFILE = 'second'\n")
    second = loader._load_official_expert_module(tmp_path)
    assert first.PROFILE == "first"
    assert second.PROFILE == "second"
    assert first is not second
