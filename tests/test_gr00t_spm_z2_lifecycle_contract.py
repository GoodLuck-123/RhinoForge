"""Board-free source contracts for the GR00T Z2 owner lifecycle."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COORDINATOR = ROOT / "src/fused/rpu_gr00t_spm_z2.cpp"
VL_MODEL = ROOT / "src/fused/rpu_gr00t_vl_encoder_model.cpp"
DENOISE_MODEL = ROOT / "src/fused/rpu_gr00t_dit_model.cpp"
DECLARATIONS = ROOT / "src/core/rpu_kernel_decls.h"
REGISTRATIONS = ROOT / "src/core/rpu_dispatch_registrations.inc"


def _between(source: str, begin: str, end: str) -> str:
    start = source.index(begin)
    return source[start : source.index(end, start)]


def test_prepare_failure_unprepares_components_in_reverse_order():
    source = COORDINATOR.read_text(encoding="utf-8")
    prepare = _between(
        source,
        "int64_t rpu_gr00t_spm_z2_prepare(",
        "void rpu_gr00t_spm_z2_unprepare(",
    )

    transaction = prepare.index("try {")
    prepare_vl = prepare.index("prepare_vl_encoder(", transaction)
    prepare_denoise = prepare.index("prepare_denoise(", prepare_vl)
    commit = prepare.index("s.plan = std::move(plan)", prepare_denoise)
    rollback_denoise = prepare.index("unprepare_denoise(", commit)
    rollback_vl = prepare.index("unprepare_vl_encoder(", rollback_denoise)
    reset = prepare.index("reset_prepared_state_locked(s)", rollback_vl)

    assert transaction < prepare_vl < prepare_denoise < commit
    assert commit < rollback_denoise < rollback_vl < reset


def test_unprepare_is_graph_external_exact_and_fail_closed():
    source = COORDINATOR.read_text(encoding="utf-8")
    unprepare = _between(
        source,
        "void rpu_gr00t_spm_z2_unprepare(",
        "int64_t rpu_gr00t_spm_z2_begin(",
    )

    no_graph = unprepare.index("TORCH_CHECK(!RpuKernelGraph::has_active()")
    no_lease = unprepare.index("TORCH_CHECK(!s.lease")
    identity = unprepare.index("check_identity(")
    endpoint = unprepare.index("validate_endpoint_specs_locked(")
    denoise = unprepare.index("unprepare_denoise(")
    vl = unprepare.index("unprepare_vl_encoder(", denoise)
    rethrow = unprepare.index("if (failure) std::rethrow_exception(failure)")
    reset = unprepare.index("reset_prepared_state_locked(s)", rethrow)

    assert no_graph < no_lease < identity < endpoint < denoise < vl
    assert vl < rethrow < reset


def test_component_unprepare_retires_only_z2_authority():
    vl_source = VL_MODEL.read_text(encoding="utf-8")
    denoise_source = DENOISE_MODEL.read_text(encoding="utf-8")
    vl = _between(
        vl_source,
        "    void unprepare_z2_layout(int64_t seq_len)",
        "    void check_z2_destroy_allowed() const",
    )
    denoise = _between(
        denoise_source,
        "    void unprepare_z2_layout(int64_t seq_len)",
        "    void check_z2_destroy_allowed() const",
    )

    assert "z2_prepared_seq_len_ = 0" in vl
    assert "z2_prepared_seq_len_ = 0" in denoise
    assert "z2_masks_prepared_ = false" in denoise
    assert "z2_masks_plan_hash_ = 0" in denoise
    assert "z2_masks_prepared_generation_ = 0" in denoise
    assert "prepared_cross_mask_ = PreparedMask{}" in denoise
    assert "prepared_cross_mask_img_ = PreparedMask{}" in denoise
    assert "*z2_masks_consumed_generation_ = *z2_mask_generation_" in denoise
    for body in (vl, denoise):
        assert "release_spm_pipeline_component" not in body
        assert "invalidate_model_state" not in body
        assert "SPM_ALLOC" not in body


def test_destroy_rejects_coordinator_plan_and_component_residue():
    coordinator = COORDINATOR.read_text(encoding="utf-8")
    vl_source = VL_MODEL.read_text(encoding="utf-8")
    denoise_source = DENOISE_MODEL.read_text(encoding="utf-8")

    vl_guard = _between(
        coordinator,
        "void check_vl_destroy_allowed(int64_t handle)",
        "void check_denoise_destroy_allowed(int64_t handle)",
    )
    denoise_guard = coordinator[
        coordinator.index("void check_denoise_destroy_allowed(int64_t handle)") :
    ]
    assert "!s.plan.has_value()" in vl_guard
    assert "!s.plan.has_value()" in denoise_guard

    vl_destroy = _between(
        vl_source,
        "void rpu_gr00t_vl_encoder_destroy(int64_t handle)",
        "void rpu_gr00t_vl_encoder_set_weights(",
    )
    denoise_destroy = _between(
        denoise_source,
        "void rpu_gr00t_dit_destroy(int64_t handle)",
        "void rpu_gr00t_dit_set_weights(",
    )
    assert vl_destroy.index("check_vl_destroy_allowed") < vl_destroy.index(
        "check_vl_component_destroy_allowed"
    ) < vl_destroy.index("Gr00tVlEncoderRegistry::destroy")
    assert denoise_destroy.index(
        "check_denoise_destroy_allowed"
    ) < denoise_destroy.index(
        "check_denoise_component_destroy_allowed"
    ) < denoise_destroy.index("Gr00tDiTRegistry::destroy")
    assert "clear the outer GraphCache and" in vl_source
    assert "clear the outer " in denoise_source
    assert "GraphCache and unprepare Z2 first" in denoise_source


def test_public_unprepare_is_declared_and_registered_once():
    declarations = DECLARATIONS.read_text(encoding="utf-8")
    registrations = REGISTRATIONS.read_text(encoding="utf-8")
    assert declarations.count("void rpu_gr00t_spm_z2_unprepare(") == 1
    assert registrations.count("gr00t_spm_z2_unprepare") == 2
    assert (
        'm.def("gr00t_spm_z2_unprepare(int vl_handle, int denoise_handle, '
        'int plan_hash) -> ()", TORCH_FN(rpu_gr00t_spm_z2_unprepare));'
        in registrations
    )
