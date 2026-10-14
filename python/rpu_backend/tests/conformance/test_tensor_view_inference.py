import pytest
import torch
import rpu_backend
VIEWS = {'view': lambda x: x.view(4, 4), 'unsqueeze': lambda x: x.unsqueeze(0), 'transpose': lambda x: x.view(4, 4).transpose(0, 1), 'expand': lambda x: x.expand(2, 16), 'squeeze': lambda x: x.view(1, 16, 1).squeeze(), 'squeeze_dim': lambda x: x.view(1, 16, 1).squeeze(0), 'as_strided': lambda x: x.as_strided((2, 4), (4, 1), 4)}

@pytest.mark.parametrize('base_inference', [False, True])
@pytest.mark.parametrize('caller_inference', [False, True])
@pytest.mark.parametrize('operation', VIEWS)
def test_view_matches_cpu_across_inference_contexts(base_inference, caller_inference, operation):
    with torch.inference_mode(base_inference):
        cpu = torch.arange(16, dtype=torch.float16)
        base = cpu.to('rpu')
    with torch.inference_mode(caller_inference):
        expected = VIEWS[operation](cpu)
        actual = VIEWS[operation](base)
        assert torch.is_inference(actual) == torch.is_inference(expected)
        assert actual.shape == expected.shape
        assert all((actual.stride(dim) == expected.stride(dim) for (dim, size) in enumerate(expected.shape) if size > 1))
        assert actual.is_contiguous() == expected.is_contiguous()
        assert actual.storage_offset() == expected.storage_offset()
        assert actual.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
        assert torch.equal(actual.cpu(), expected)
        if not torch.is_inference(base):
            assert actual._version == base._version

def test_as_strided_refuses_a_view_past_storage_end():
    base = torch.empty(16, dtype=torch.float16, device='rpu')
    with pytest.raises(RuntimeError):
        base.as_strided((2, 8), (8, 1), 1)

@pytest.mark.parametrize('dim', [None, 0])
def test_squeeze_single_element_becomes_scalar(dim):
    cpu = torch.ones(1, dtype=torch.float16)
    rpu = cpu.to('rpu')
    expected = cpu.squeeze() if dim is None else cpu.squeeze(dim)
    actual = rpu.squeeze() if dim is None else rpu.squeeze(dim)
    assert actual.shape == expected.shape == torch.Size([])
    assert torch.equal(actual.cpu(), expected)
