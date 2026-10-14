import json as _json
import sys
import pytest
import torch

def main() -> int:
    from rpu_backend import RPUCache, RPUModelForCausalLM
    from rpu_backend.api.errors import RPUBackendError, UnsupportedModelError, RPUUnsupportedDtypeError, RPUSingleHandleError, SPMExhaustionError, WeightShapeMismatchError
    from rpu_backend.api.errors import RPUBackendError as _Base2, UnsupportedModelError as _UME2, RPUUnsupportedDtypeError as _DTE2, RPUSingleHandleError as _SHE2, SPMExhaustionError as _SPM2, WeightShapeMismatchError as _WSM2
    assert RPUBackendError is _Base2
    assert UnsupportedModelError is _UME2
    assert RPUUnsupportedDtypeError is _DTE2
    assert RPUSingleHandleError is _SHE2
    assert SPMExhaustionError is _SPM2
    assert WeightShapeMismatchError is _WSM2
    for cls in (UnsupportedModelError, RPUUnsupportedDtypeError, RPUSingleHandleError, SPMExhaustionError, WeightShapeMismatchError):
        assert issubclass(cls, RPUBackendError), f'{cls.__name__} must subclass RPUBackendError (D-18)'
    from rpu_backend.api.cache import RPUCache as _RPUCache
    assert RPUCache is _RPUCache, 'RPUCache top-level + api.cache identity broken'
    with pytest.raises(ImportError):
        from rpu_backend.core.weights import RPUCache as _CoreRPUCache
    cache = RPUCache(num_layers=1, batch_size=1, max_seq_len=16, num_kv_heads=1, head_dim=16, device='cpu', dtype=torch.float16)
    assert cache.num_layers == 1
    try:
        RPUModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', dtype=torch.bfloat16, local_files_only=True)
        assert False, 'D-19 dtype guard MUST raise on torch.bfloat16'
    except RPUUnsupportedDtypeError as e:
        msg = str(e).lower()
        assert 'fp16' in msg or 'float16' in msg, f'D-19 remediation message must mention fp16: {e}'
    for reserved in ('torch_dtype', 'device_map', 'low_cpu_mem_usage', 'tp_plan'):
        try:
            RPUModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', dtype=torch.float16, local_files_only=True, **{reserved: 'value'})
            assert False, f'reserved kwarg {reserved!r} guard MUST raise'
        except ValueError as e:
            assert 'reserved key' in str(e), f"Reserved-kwarg ValueError must say 'reserved key': {e}"
    from rpu_backend.runtime.registry import ADAPTERS
    assert 'Qwen3ForCausalLM' in ADAPTERS, f'Qwen3ForCausalLM missing from built-in manifest: {sorted(ADAPTERS)}'
    from unittest.mock import patch, MagicMock
    fake_cfg = MagicMock()
    fake_cfg.architectures = ['DefinitelyNotRegistered_ScaffoldSmokeTest']
    fake_cfg.torch_dtype = torch.float16
    with patch('rpu_backend.api.causal_lm.AutoConfig.from_pretrained', return_value=fake_cfg):
        try:
            RPUModelForCausalLM.from_pretrained('unused/path', dtype=torch.float16, local_files_only=True)
            assert False, 'Expected UnsupportedModelError for unregistered architecture'
        except UnsupportedModelError as e:
            assert 'DefinitelyNotRegistered_ScaffoldSmokeTest' in str(e), f'Error message must name the missing arch: {e}'
    result = {'plan': '02-01', 'imports_ok': True, 'rpu_cache_identity': True, 'errors_count': 6, 'dtype_guard_fired': True, 'missing_adapter_fired': True, 'reserved_kwargs_guarded': True}
    print(f'RESULT: {_json.dumps(result, sort_keys=True)}')
    print('[PASS] Plan 02-01 scaffold smoke — LIB-01 + LIB-03 acceptance gates green')
    return 0
if __name__ == '__main__':
    sys.exit(main())

def test_public_import_cache_and_loader_guards():
    assert main() == 0
