def _test_only_unregister(arch: str) -> None:
    from rpu_backend.runtime.registry import ADAPTERS
    ADAPTERS.pop(arch, None)
