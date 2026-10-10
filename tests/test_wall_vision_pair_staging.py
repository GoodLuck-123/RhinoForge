"""CPU execution of the real pair staging and ordinary patch consumer."""
from types import MethodType, SimpleNamespace as NS

import pytest
import torch
import torch.nn.functional as F

from test_wall_vision_pair_252 import helpers


def byte_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def make_owner(monkeypatch, *, host_cache=True, device_patch=True, folded=True):
    h = helpers()
    real_to = torch.Tensor.to
    uploads = []

    def cpu_device_boundary(tensor, *args, **kwargs):
        if (args and args[0] == "rpu") or kwargs.get("device") == "rpu":
            if tensor.ndim == 2 and tensor.shape == (672, 1184):
                uploads.append(tensor.clone())
            if args and args[0] == "rpu":
                args = ("cpu", *args[1:])
            if kwargs.get("device") == "rpu":
                kwargs = dict(kwargs, device="cpu")
            return real_to(tensor, *args, **kwargs).clone()
        return real_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_device_boundary)
    monkeypatch.setattr(torch.ops.rpu, "gather_embedding",
                        lambda x, ids: x[ids.long()].clone(), raising=False)
    # A small output width keeps this a staging check; all 1176 data columns,
    # the homogeneous bias and seven pad columns are still exercised.
    rng = torch.Generator().manual_seed(51252)
    weight = (torch.randn(3, 1184, generator=rng) * .001).half()
    weight[:, 1176] = torch.tensor([.125, -.25, .5])
    owner = NS(_host_cache=host_cache, _device_patch_embed=device_patch,
               _linear_acc32=False,
               _patch_w_fold_rpu=weight if folded else None,
               _patch_w_rpu=weight.clone(), _patch_w_fold_cpu=weight.float(),
               _patch_w=weight[:, :1176].float(), _pe_k=1184, _pe_buf_cache={}, _index_rpu_cache={})
    for name in ("_patch_embed", "_patch_embed_pair_252", "_consume_patch_embed_input", "_patch_perm_ids",
                 "_prepare_patch_pixels", "_index_rpu"):
        setattr(owner, name, MethodType(h[name], owner))
    window = h["_pair_252_layout"](torch.tensor([[1, 18, 18]] * 2))[1]
    return owner, window, uploads


def old_padded(pixels):
    padded = pixels.new_zeros((2, 336, 1176))
    padded[:, :324].copy_(pixels.reshape(2, 324, 1176))
    return padded.reshape(672, 1176)


def test_direct_pair_staging_matches_old_input_and_output_across_a_b_a_and_uint8_edges(monkeypatch):
    owner, window, uploads = make_owner(monkeypatch)
    a = torch.arange(648 * 1176, dtype=torch.int32).remainder(256).to(torch.uint8).reshape(648, 1176)
    b = 255 - a
    pair_address = None
    first = None
    saved_first = None
    old_new_zeros = torch.Tensor.new_zeros

    def reject_intermediate(tensor, size, *args, **kwargs):
        assert not (tensor.dtype == torch.uint8 and tuple(size) == (2, 336, 1176)), \
            "optimized pair allocated the old uint8 padded intermediate"
        return old_new_zeros(tensor, size, *args, **kwargs)

    for iteration, pixels in enumerate((a, b, a, torch.zeros_like(a), torch.full_like(a, 255))):
        original = pixels.clone()
        uploads.clear()
        reference = owner._patch_embed(old_padded(pixels), window, 672, 4)
        expected_input = uploads[0]
        # A different 672-row geometry overwrites all ordinary cached data,
        # including rows that are dummy in pair252. It must not poison pair252.
        owner._patch_embed(torch.full((672, 1176), 251, dtype=torch.uint8), window, 672, 4)
        ordinary = owner._pe_buf_cache[(672, True)]
        assert bool((ordinary[:, :1176] == 251).all())
        uploads.clear()
        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "new_zeros", reject_intermediate)
            actual = owner._patch_embed_pair_252(pixels, window)
        assert len(uploads) == 1 and byte_equal(uploads[0], expected_input)
        assert byte_equal(actual, reference)
        assert byte_equal(pixels, original)
        x = uploads[0].reshape(2, 336, 1184)
        assert byte_equal(x[:, :324, :1176], pixels.reshape(2, 324, 1176).half())
        assert bool((x[:, 324:, :1176] == 0).all())
        assert bool((x[:, :, 1176] == 1).all())
        assert bool((x[:, :, 1177:] == 0).all())
        keys = [key for key in owner._pe_buf_cache if key != (672, True)]
        assert len(keys) == 1
        pair = owner._pe_buf_cache[keys[0]]
        assert pair.data_ptr() != ordinary.data_ptr()
        assert bool((ordinary[:, :1176] == 251).all())
        if pair_address is None:
            pair_address = pair.data_ptr()
        assert pair.data_ptr() == pair_address
        if iteration == 0:
            first, saved_first = actual, actual.clone()
        elif iteration == 1:
            assert not byte_equal(actual, first)
        elif iteration == 2:
            assert byte_equal(actual, first)
        assert byte_equal(first, saved_first)  # Returned DDR/output remains fresh.
    # Raw dummy zero still projects the normalize bias, not zero embedding.
    dummy = F.linear(uploads[0].reshape(2, 336, 1184)[:, 324:], owner._patch_w_fold_rpu)
    assert byte_equal(dummy, owner._patch_w_fold_rpu[:, 1176].expand(2, 12, 3))


@pytest.mark.parametrize("options,dtype", [
    ({"host_cache": False}, torch.uint8),
    ({"device_patch": False}, torch.uint8),
    ({"folded": False}, torch.uint8),
    ({}, torch.float32),
])
def test_pair_staging_fallback_preserves_original_padding_and_projection(monkeypatch, options, dtype):
    owner, window, _ = make_owner(monkeypatch, **options)
    pixels = torch.arange(648 * 1176, dtype=torch.int32).remainder(256).to(dtype).reshape(648, 1176)
    reference = owner._patch_embed(old_padded(pixels), window, 672, 4)
    result = owner._patch_embed_pair_252(pixels, window)
    assert byte_equal(result, reference)
    assert all(isinstance(key[0], int) for key in owner._pe_buf_cache)
