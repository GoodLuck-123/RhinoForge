'Hy policy exports and action horizon contracts.'
MODEL_HORIZON = 50
ACTION_HORIZON = 49


def test_public_names_are_exported() -> None:
    '两个公共名字在 `rpu_backend.api.__all__` 里，且是同一个对象。'
    import rpu_backend.api as api

    assert "HyEmbodiedPolicy" in api.__all__
    assert "HyEmbodiedActionOutput" in api.__all__
    from rpu_backend.api import HyEmbodiedActionOutput, HyEmbodiedPolicy

    assert api.HyEmbodiedPolicy is HyEmbodiedPolicy
    assert api.HyEmbodiedActionOutput is HyEmbodiedActionOutput

def test_horizon_contract_is_50_and_49() -> None:
    '`model_horizon`(50) 与 `action_horizon`(49) 是两个数，不是一个。'
    from rpu_backend.api.hy_embodied import MODEL_ACTION_STEPS
    from rpu_backend.api import HyEmbodiedPolicy

    inst = HyEmbodiedPolicy.__new__(HyEmbodiedPolicy)
    assert inst.model_horizon == MODEL_ACTION_STEPS == MODEL_HORIZON
    assert inst.action_horizon == MODEL_ACTION_STEPS - 1 == ACTION_HORIZON
    assert inst.model_horizon != inst.action_horizon
