"""Exact8B FP16 storage reuse; no hardware or expanded support claim."""
import gc
from types import SimpleNamespace as NS

import pytest
import torch

from rpu_backend.adapters import qwen3_vl as adapter
from rpu_backend.adapters.qwen3_vl import vision
from rpu_backend.quant import load
from rpu_backend.tests.acceptance.test_qwen3_vl_install_lifecycle import install_runtime  # noqa: F401
from test_qwen3_vl_merger_lifecycle import _model_8b
from test_qwen3_vl_vision_weight_banks import transfers  # noqa: F401


@pytest.mark.parametrize("acc32", [None, True, False])
def test_public_8b_uses_storage_banks_without_2b4b_vision_schedules(install_runtime, monkeypatch, acc32):
    model = _model_8b()
    calls = []
    def move(value, *, dtype, per_layer=False):
        assert value is model and value._rpu_swizzle_started
        calls.append((dtype, per_layer))
    monkeypatch.setattr(load, "_move_materialized_decoder_for_rpu", move)
    execution = (None if acc32 is None else {
        "prefill": {"linear_acc32": acc32}, "vision": {"linear_acc32": acc32}})
    if execution is not None:
        model._rpu_execution = execution
    owner = adapter.Qwen3VLAdapter(model)
    assert owner.to_rpu() is model
    assert calls == [(torch.float16, True)]
    assert install_runtime["vision_kwargs"]["_pack_fp16_block_weights"] is True
    assert install_runtime["vision_kwargs"]["_allow_padded_8b"] is True
    assert not install_runtime["vision_kwargs"].get("w8a16", False)
    actual_acc32 = False if acc32 is None else acc32
    assert ("set_linear_acc32", 22, actual_acc32) in install_runtime["events"]
    assert ("set_vision_linear_acc32", 11, actual_acc32) in install_runtime["events"]
    forbidden = {"set_vision_large_image_auto_chunk", "set_fp16_chunked_merger",
                 "set_fp16_compact_encoder", "qwen3vl_vision_set_fast_replay",
                 "qwen3vl_vision_set_preload_replay_skip"}
    assert not any(event[0] in forbidden for event in install_runtime["events"])
    assert owner.to_rpu() is model and len(calls) == 1
    owner.close()


def test_full_8b_text_layer_banks_have_exact_spans_and_owners(monkeypatch):
    shapes = {"self_attn.q_proj": (4096, 4096), "self_attn.k_proj": (1024, 4096),
              "self_attn.v_proj": (1024, 4096), "self_attn.o_proj": (4096, 4096),
              "mlp.gate_proj": (12288, 4096), "mlp.up_proj": (12288, 4096),
              "mlp.down_proj": (4096, 12288)}
    names = tuple(tuple(f"model.language_model.layers.{i}.{role}.weight" for role in shapes)
                  for i in range(36))
    raw = {name: torch.empty(shape, device="meta", dtype=torch.float16)
           for layer_names in names for name, shape in zip(layer_names, shapes.values())}
    original, allocations = torch.empty, []
    def empty(count, **kwargs):
        assert kwargs == {"device": "rpu", "dtype": torch.float16}
        allocations.append(count * 2)
        return original(count, device="meta", dtype=torch.float16)
    monkeypatch.setattr(torch, "empty", empty)
    views = [load._allocate_decoder_projection_views(
        NS(get_parameter=raw.__getitem__), (layer,), dtype=torch.float16) for layer in names]
    assert allocations == [385875968] * 36
    assert len({v.untyped_storage()._cdata for layer in views for v in layer.values()}) == 36
    for layer_names, layer_views in zip(names, views):
        end = 0
        for name in layer_names:
            view = layer_views[name]
            assert view.shape == raw[name].shape and view.is_contiguous()
            assert view.storage_offset() * 2 == end and end % 256 == 0
            end += view.numel() * 2
        assert end == 385875968 < 2**32


def test_8b_bank_install_failure_remains_poisoned(install_runtime, monkeypatch):
    model = _model_8b()
    def fail(value, *, dtype, per_layer=False):
        assert value is model and value._rpu_swizzle_started
        assert dtype == torch.float16 and per_layer
        raise RuntimeError("8B layer bank allocation failed")
    monkeypatch.setattr(load, "_move_materialized_decoder_for_rpu", fail)
    owner = adapter.Qwen3VLAdapter(model)
    with pytest.raises(RuntimeError, match="bank allocation failed"):
        owner.to_rpu()
    assert model._rpu_swizzle_started and not getattr(model, "_rpu_swizzled", False)
    assert ("install", "vision") not in install_runtime["events"]
    with pytest.raises(adapter.RPUBackendError, match="prior swizzle attempt"):
        owner.to_rpu()


@pytest.mark.parametrize("fault", [None, "private_flag", "quant", "depth", "head",
                                  "channels", "positions", "32b", "blocks"])
def test_private_8b_vision_storage_requires_exact_padded_fp16_profile(monkeypatch, fault):
    config = _model_8b().config.vision_config
    visual = NS(config=config, blocks=[None] * 27)
    kwargs = {"_allow_padded_8b": True, "_pack_fp16_block_weights": True}
    if fault == "private_flag": kwargs["_allow_padded_8b"] = False
    elif fault == "quant": kwargs["w8a16"] = True
    elif fault == "depth": config.depth = 26
    elif fault == "head": config.num_heads = 18
    elif fault == "channels": config.in_channels = 4
    elif fault == "positions": config.num_position_embeddings = 2305
    elif fault == "32b": config.out_hidden_size = 5120
    elif fault == "blocks": visual.blocks.pop()
    calls = []
    monkeypatch.setattr(vision, "_install_qwen3_vl_vision_for_rpu_impl",
                        lambda *a, **kw: calls.append(kw) or 17)
    if fault is None:
        assert vision.install_qwen3_vl_vision_for_rpu(visual, **kwargs) == 17
        assert calls[0]["_pack_fp16_block_weights"] and calls[0]["_allow_padded_8b"]
        assert not calls[0]["w8a16"]
    else:
        with pytest.raises(ValueError):
            vision.install_qwen3_vl_vision_for_rpu(visual, **kwargs)
        assert calls == []


def test_real_padded_8b_vision_conversion_bank_preserves_all_bytes_and_lifetime(transfers):
    def linear(k, n):
        result = torch.nn.Linear(k, n, dtype=torch.float16)
        with torch.no_grad():
            # Low-magnitude, nonconstant deterministic rows make byte movement observable.
            result.weight.copy_((torch.arange(n).remainder(17).half() / 32)[:, None])
            result.bias.copy_(torch.arange(n).remainder(13).half() / 32)
        return result
    block = NS(attn=NS(qkv=linear(1152, 3456), proj=linear(1152, 1152)),
               mlp=NS(linear_fc1=linear(1152, 4304), linear_fc2=linear(4304, 1152)))
    vision._convert_vision_block_weights_for_rpu(
        block, 16, 1152, physical_head_dim=80, physical_intermediate_size=4352)
    weights = (block._rpu_q_w, block._rpu_k_w, block._rpu_v_w,
               block.attn.proj.weight.detach(), block.mlp.linear_fc1.weight.detach(),
               block.mlp.linear_fc2.weight.detach())
    assert [tuple(w.shape) for w in weights] == [(1280, 1152)] * 3 + [
        (1152, 1280), (4352, 1152), (1152, 4352)]
    assert all(getattr(block, f"_rpu_{role}_ws") is None for role in ("q", "k", "v", "o", "fc1", "fc2"))
    for role in ("q", "k", "v"):
        assert torch.count_nonzero(getattr(block, f"_rpu_{role}_b").view(16, 80)[:, 72:]) == 0
    assert torch.count_nonzero(block._rpu_fc1_b[4304:]) == 0
    snapshots = tuple(w.clone() for w in weights)
    views = vision._vision_block_weights_to_rpu(weights, w8a16=False, pack_weights=True)
    assert len(transfers) == 1
    assert len({view.untyped_storage().data_ptr() for view in views}) == 1
    end = 0
    for view, before in zip(views, snapshots):
        assert torch.equal(view, before) and view.stride() == before.stride()
        assert view.storage_offset() * 2 == end and end % 256 == 0
        end += view.numel() * 2
        assert view.untyped_storage().nbytes() == 31850496
    assert end == 31850496
    # A surviving tensor view must retain its owner after conversion temporaries die.
    retained = views[-1]
    del views, weights, block
    gc.collect()
    assert torch.equal(retained, snapshots[-1])
