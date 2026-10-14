from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[4]
_BACKEND_SOURCE = _REPO_ROOT / 'src/core/rpu_backend.cpp'
_TENSOR_FRAGMENT = _REPO_ROOT / 'src/core/rpu_tensor_ops.inc'
_REGISTRATION_FRAGMENT = _REPO_ROOT / 'src/core/rpu_dispatch_registrations.inc'
_PYBIND_FRAGMENT = _REPO_ROOT / 'src/core/rpu_pybind.inc'

def test_privateuse1_registration_is_owned_by_explicit_startup_chain():
    backend = _BACKEND_SOURCE.read_text(encoding='utf-8')
    tensor = _TENSOR_FRAGMENT.read_text(encoding='utf-8')
    pybind = _PYBIND_FRAGMENT.read_text(encoding='utf-8')
    assert 'REGISTER_ALLOCATOR' not in tensor
    assert 'C10_REGISTER_GUARD_IMPL' not in tensor
    assert 'void initialize_rpu_tensor_runtime()' in tensor
    assert 'c10::SetAllocator(' in tensor
    assert 'c10::impl::registerDeviceGuard(' in tensor
    init_start = backend.index('static void initialize_rpu_backend_runtime()')
    init_end = backend.index('// Shutdown:', init_start)
    init_body = backend[init_start:init_end]
    assert init_body.index('initialize_rpu_tensor_runtime();') < init_body.index('KernelCache::instance().initialize();')
    assert 'initialize_rpu_backend_runtime();' in pybind[pybind.index('PYBIND11_MODULE(rpu_backend, m)'):]

def test_allocator_fails_before_sdk_allocation_when_device_nodes_are_unusable():
    backend = _BACKEND_SOURCE.read_text(encoding='utf-8')
    tensor = _TENSOR_FRAGMENT.read_text(encoding='utf-8')
    pybind = _PYBIND_FRAGMENT.read_text(encoding='utf-8')
    allocate_start = tensor.index('at::DataPtr allocate(size_t nbytes) override')
    allocate_end = tensor.index('void copy_data', allocate_start)
    allocate_body = tensor[allocate_start:allocate_end]
    assert allocate_body.index('rpu_device_nodes_accessible()') < allocate_body.index('rpu_allocate_registered_ddr(nbytes)')
    assert 'm.def("rpu_runtime_is_available"' in pybind

def test_sdk_sensitive_fragments_remain_in_composition_translation_unit():
    backend = _BACKEND_SOURCE.read_text(encoding='utf-8')
    registrations = _REGISTRATION_FRAGMENT.read_text(encoding='utf-8')
    pybind = _PYBIND_FRAGMENT.read_text(encoding='utf-8')
    assert '#include "rpu_kernel_cache.inc"' in backend
    assert '#include "rpu_tensor_ops.inc"' in backend
    assert '#include "rpu_dispatch_registrations.inc"' in backend
    assert '#include "rpu_pybind.inc"' in backend
    assert 'TORCH_LIBRARY(rpu, m)' in registrations
    assert 'PYBIND11_MODULE(rpu_backend, m)' in pybind
    assert 'g_rpu_backend_lifecycle != RpuBackendLifecycleState::kReady' in backend

def test_public_graph_reset_is_real_and_prepared_batch_stubs_are_gone():
    backend = _BACKEND_SOURCE.read_text(encoding='utf-8')
    pybind = _PYBIND_FRAGMENT.read_text(encoding='utf-8')
    bootstrap = (_REPO_ROOT / 'python/rpu_backend/_bootstrap.py').read_text(encoding='utf-8')
    torch_rpu = (_REPO_ROOT / 'python/rpu_backend/_torch_rpu.py').read_text(encoding='utf-8')
    control = (_REPO_ROOT / 'python/rpu_backend/runtime/control.py').read_text(encoding='utf-8')
    assert 'm.def("reset_graph_cache", []()' not in pybind
    assert 'm.def("reset_graph_cache", &invalidate_registered_rpu_kernel_graphs' in pybind
    assert 'reset_graph_cache = _cpp_ext.reset_graph_cache' in bootstrap
    assert 'reset_prepared_batch' not in pybind
    assert 'is_prepared_batch' not in pybind
    assert 'reset_prepared_batch' not in torch_rpu
    assert 'is_prepared_batch' not in torch_rpu
    assert 'def reset_prepared_batch' not in control
    assert 'def is_prepared_batch' not in control
