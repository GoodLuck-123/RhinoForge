import builtins
import pytest
from rpu_backend.api.errors import UnsupportedModelError

def test_paligemma_policy_imports_from_api_submodule():
    from rpu_backend.api.paligemma import PaliGemmaPolicy
    assert PaliGemmaPolicy.__name__ == 'PaliGemmaPolicy'

def test_paligemma_policy_imports_from_api_package():
    from rpu_backend.api import PaliGemmaPolicy
    assert PaliGemmaPolicy.__name__ == 'PaliGemmaPolicy'

def test_paligemma_policy_direct_constructor_rejected():
    from rpu_backend.api import PaliGemmaPolicy
    with pytest.raises(UnsupportedModelError, match='fixed and recertified'):
        PaliGemmaPolicy()

def test_from_pretrained_fails_before_loader_import(monkeypatch):
    from rpu_backend.api import PaliGemmaPolicy
    imported = []
    real_import = builtins.__import__

    def tracking_import(name, *args, **kwargs):
        if name == 'rpu_backend.adapters.paligemma.loader':
            imported.append(name)
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', tracking_import)
    with pytest.raises(UnsupportedModelError, match='fixed and recertified'):
        PaliGemmaPolicy.from_pretrained('unused')
    assert 'rpu_backend.adapters.paligemma.loader' not in imported

def test_from_hf_model_fails_before_adapter_construction():
    from rpu_backend.api import PaliGemmaPolicy
    with pytest.raises(UnsupportedModelError, match='numerical gates'):
        PaliGemmaPolicy.from_hf_model(object())

def test_legacy_instance_cannot_reenter_any_device_path():
    from rpu_backend.api import PaliGemmaPolicy
    legacy = object.__new__(PaliGemmaPolicy)
    with pytest.raises(UnsupportedModelError, match='numerical gates'):
        legacy.to('cpu')

def test_top_level_all_unchanged_and_no_top_level_export():
    import rpu_backend
    assert rpu_backend.__all__ == ['__version__', 'RPUCache', 'RPUModelForCausalLM', 'reset_graph_cache']
    with pytest.raises(AttributeError, match='rpu_backend.api.PaliGemmaPolicy'):
        _ = rpu_backend.PaliGemmaPolicy

def test_paligemma_not_registered_as_causal_lm_adapter():
    from rpu_backend.runtime.registry import list_adapters
    assert 'PaliGemmaForConditionalGeneration' not in list_adapters()
