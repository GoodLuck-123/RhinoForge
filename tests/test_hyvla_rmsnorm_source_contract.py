"""Board-free contract checks for HyVLA's ordinary FP16 RMSNorm routes."""

from pathlib import Path
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
HYVLA_CPP = (ROOT / "src/fused/rpu_hyvla_vlm_model.cpp").read_text(
    encoding="utf-8"
)
HYVLA_HEADER = (ROOT / "src/fused/rpu_hyvla_vlm_model.h").read_text(
    encoding="utf-8"
)
DECLARATIONS = (ROOT / "src/core/rpu_kernel_decls.h").read_text(
    encoding="utf-8"
)


def _block(source: str, marker: str) -> str:
    start = source.index(marker)
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def test_hyvla_runtime_config_snapshots_before_one_shot_commit_and_keeps_abi():
    runtime_config = _block(
        HYVLA_CPP, "void HyVlaVlmModel::set_runtime_config("
    )
    snapshot = runtime_config.index(
        "const RpuRmsNormSpmContract rmsnorm_spm_contract ="
    )
    invalidate = runtime_config.index("invalidate_model_state();")
    commits = [
        runtime_config.index(f"{field} =")
        for field in (
            "cold_fast_replay_",
            "cold_fast_replay_preload_",
            "cold_mask_once_",
            "cold_partial_rope_",
            "cold_mot_norm_nomerge_",
            "cold_silu_mul_",
            "cold_rmsnorm_spm_contract_",
            "cold_paired_rmsnorm_available_",
            "cold_config_bound_",
        )
    ]
    assert snapshot < invalidate < min(commits)
    assert "rpu_snapshot_rmsnorm_spm_contract(" in runtime_config
    assert "static_cast<uint32_t>(rmsnorm_capability)" in runtime_config
    assert runtime_config.index("cold_rmsnorm_spm_contract_ =") < runtime_config.index(
        "cold_config_bound_ = true;"
    )

    # This remains the same seven-field public runtime-config ABI. Payload
    # discovery is private to the handle and does not add a Python/C++ argument.
    public_signature = re.compile(
        r"void\s+rpu_hyvla_vlm_set_runtime_config\(\s*"
        r"int64_t\s+handle,\s*"
        r"bool\s+fast_replay,\s*bool\s+fast_replay_preload,\s*"
        r"bool\s+mask_once,\s*bool\s+partial_rope,\s*"
        r"int64_t\s+mot_norm_nomerge,\s*bool\s+silu_mul,\s*"
        r"int64_t\s+rmsnorm_capability\s*\)"
    )
    assert public_signature.search(DECLARATIONS)
    assert public_signature.search(HYVLA_CPP)


def test_hyvla_manifest_and_consumer_share_one_effective_contract():
    manifest = _block(
        HYVLA_CPP, "FmbPhysicalExecutionManifest HyVlaVlmModel::physical_manifest_for_candidate("
    )
    consumer = _block(
        HYVLA_CPP, "void HyVlaVlmModel::build_layer_subgraph("
    )
    for section in (manifest, consumer):
        assert "cold_rmsnorm_spm_contract_.resolve(rows, cols)" in section
        assert (
            "cold_rmsnorm_spm_contract_.manifest_arguments(rows, cols)"
            in section
        )
    for site in (
        "kHyVlaVlmRmsNormFullRowsSite",
        "kHyVlaVlmRmsNormQHeadRowsSite",
        "kHyVlaVlmRmsNormKHeadRowsSite",
    ):
        assert site in manifest
        assert site in consumer

    launches = re.findall(
        r"^[ \t]*rpu_launch_rmsnorm_spm_kernel\([\s\S]*?\);",
        consumer,
        flags=re.MULTILINE,
    )
    # The two ordinary branch pairs now share one fallback helper.
    assert len(launches) == 5
    assert sum("full_rmsnorm_route" in launch for launch in launches) == 3
    assert sum("q_head_rmsnorm_route" in launch for launch in launches) == 1
    assert sum("k_head_rmsnorm_route" in launch for launch in launches) == 1

    identity = _block(
        HYVLA_HEADER,
        "std::vector<int64_t> kvinsert_cost_weight_identity() const override",
    )
    assert "static_cast<int64_t>(cold_mot_norm_nomerge_)" in identity
    assert "cold_paired_rmsnorm_available_" in identity
    assert "cold_rmsnorm_spm_contract_.capability" in identity
    assert "cold_rmsnorm_capability_" not in HYVLA_HEADER + HYVLA_CPP


def test_rmsnorm_snapshot_intersects_requested_ceiling_with_loaded_payloads(
    tmp_path: Path,
):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")

    resolver = DECLARATIONS[
        DECLARATIONS.index("enum RpuRmsNormCapability") :
        DECLARATIONS.index(
            "inline RpuMixedRmsNormSpmRoute rpu_resolve_mixed_rmsnorm_spm_route"
        )
    ]
    rmsnorm_source = (ROOT / "src/ops/rpu_rmsnorm.cpp").read_text(
        encoding="utf-8"
    )
    snapshot = _block(
        rmsnorm_source, "RpuRmsNormSpmContract rpu_snapshot_rmsnorm_spm_contract("
    )
    program = (
        r'''
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("contract rejected"); } while (0)
enum class KernelId { RMS_NORM_SPM_V16, RMS_NORM_SPM_V32 };
struct KernelCache {
    bool v16 = false;
    bool v32 = false;
    static KernelCache& instance() { static KernelCache cache; return cache; }
    bool has_loaded(KernelId id) const {
        return id == KernelId::RMS_NORM_SPM_V16 ? v16 : v32;
    }
};
'''
        + resolver
        + snapshot
        + r'''
void expect(uint32_t requested, bool v16, bool v32, uint32_t effective) {
    auto& cache = KernelCache::instance();
    cache.v16 = v16;
    cache.v32 = v32;
    const auto contract = rpu_snapshot_rmsnorm_spm_contract(requested);
    assert(contract.capability == effective);
}
int main() {
    expect(RPU_RMSNORM_CAP_ALL, false, false, RPU_RMSNORM_CAP_BASE);
    expect(RPU_RMSNORM_CAP_ALL, true, false,
           RPU_RMSNORM_CAP_BASE | RPU_RMSNORM_CAP_V16);
    expect(RPU_RMSNORM_CAP_BASE | RPU_RMSNORM_CAP_V16, false, true,
           RPU_RMSNORM_CAP_BASE);
    expect(RPU_RMSNORM_CAP_BASE | RPU_RMSNORM_CAP_V16, true, true,
           RPU_RMSNORM_CAP_BASE | RPU_RMSNORM_CAP_V16);
    expect(RPU_RMSNORM_CAP_ALL, true, true, RPU_RMSNORM_CAP_ALL);

    auto& cache = KernelCache::instance();
    cache.v16 = true;
    cache.v32 = true;
    const auto all = rpu_snapshot_rmsnorm_spm_contract(RPU_RMSNORM_CAP_ALL);
    assert(all.resolve(192, 2048) == RpuRmsNormSpmRoute::V32_GRID8);
    assert(all.resolve(240, 2048) == RpuRmsNormSpmRoute::V16_GRID8);
    assert(all.resolve(768, 128) == RpuRmsNormSpmRoute::V32_GRID8);
    assert(all.resolve(960, 128) == RpuRmsNormSpmRoute::V32_GRID8);
    assert(all.manifest_arguments(240, 2048) == std::vector<int64_t>({
        240, 2048, RPU_RMSNORM_CAP_ALL}));

    // The published expansion's corrected copy is distinct from generic main
    // V16, and the physical descriptor binds that choice even on a BASE shape.
    auto bitcopy = rpu_snapshot_rmsnorm_spm_contract(
        RPU_RMSNORM_CAP_BASE | RPU_RMSNORM_CAP_V16);
    bitcopy.v16_bitcopy = true;
    assert(bitcopy.resolve(64, 1024) == RpuRmsNormSpmRoute::HYVLA_V16_BITCOPY_GRID8);
    assert(bitcopy.resolve(51, 1024) == RpuRmsNormSpmRoute::BASE);
    assert(bitcopy.manifest_arguments(64, 1024) == std::vector<int64_t>({
        64, 1024, RPU_RMSNORM_CAP_BASE | RPU_RMSNORM_CAP_V16, 1}));
    assert(bitcopy.manifest_arguments(51, 1024).back() == 1);
    assert(all.resolve(240, 2048) == RpuRmsNormSpmRoute::V16_GRID8);

    cache.v16 = false;
    const auto v32_only =
        rpu_snapshot_rmsnorm_spm_contract(RPU_RMSNORM_CAP_ALL);
    assert(v32_only.resolve(240, 2048) == RpuRmsNormSpmRoute::BASE);
    assert(v32_only.resolve(192, 2048) == RpuRmsNormSpmRoute::V32_GRID8);

    bool rejected = false;
    try { (void)rpu_snapshot_rmsnorm_spm_contract(0); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
}
'''
    )
    source = tmp_path / "rmsnorm_snapshot.cpp"
    source.write_text(program, encoding="utf-8")
    executable = tmp_path / "rmsnorm_snapshot"
    subprocess.run(
        [compiler, "-std=c++17", str(source), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(executable)], check=True)


def test_expert_pad16_selects_actual_vector_routes_and_owns_k_padding():
    source = (ROOT / "src/fused/rpu_hyvla_expert_model.cpp").read_text()
    header = (ROOT / "src/fused/rpu_hyvla_expert_model.h").read_text()
    setter = _block(source, "void HyVlaExpertModel::set_runtime_config(")
    assert 'get_kernel("hyvla_expert_rmsnorm_v16_bitcopy")' in setter
    assert "vector_available = rmsnorm_pad16 &&" in setter
    assert "RPU_RMSNORM_CAP_BASE | (vector_available ? RPU_RMSNORM_CAP_V16 : 0u)" in setter
    assert "rmsnorm_contract.v16_bitcopy = vector_available;" in setter
    assert setter.index("rpu_snapshot_rmsnorm_spm_contract(") < setter.index("invalidate_model_state();")
    assert setter.index("invalidate_model_state();") < setter.index("cold_fast_replay_ =")
    assert setter.index("cold_rmsnorm_spm_contract_ =") < setter.index("cold_config_bound_ = true;")

    buffers = _block(source, "std::vector<BufferDecl> HyVlaExpertModel::baseline_buffer_declarations(")
    # K normalization writes padded rows even with KV-insert padding disabled.
    assert "if (cold_kvpad16_ || cold_rmsnorm_pad16_)" in buffers
    assert "hyvla_grow_kv_slots(" in buffers
    for name in ("residual1", "input_norm", "q", "output"):
        assert f'hyvla_grow_slot(d, "{name}"' in buffers

    manifest = _block(source, "FmbPhysicalExecutionManifest HyVlaExpertModel::physical_manifest_for_candidate(")
    consumer = _block(source, "void HyVlaExpertModel::build_layer_subgraph(")
    for section in (manifest, consumer):
        assert "cold_rmsnorm_spm_contract_.resolve(rows, cols)" in section
        assert "cold_rmsnorm_spm_contract_.manifest_arguments(rows, cols)" in section
        for site in ("FULL", "Q", "K"):
            assert f"HYVLA_EXPERT_{site}_RMSNORM_ROUTE_SITE" in section
    launches = re.findall(r"rpu_launch_rmsnorm_spm_kernel\([\s\S]*?\);", consumer)
    assert len(launches) == 5
    assert sum("full_norm_route" in launch for launch in launches) == 3
    assert sum("q_head_norm_route" in launch for launch in launches) == 1
    assert sum("k_head_norm_route" in launch for launch in launches) == 1
    assert "cold_rmsnorm_spm_contract_.capability" in _block(
        header, "std::vector<int64_t> kvinsert_cost_weight_identity() const override")
    assert "cold_rmsnorm_spm_contract_.v16_bitcopy" in _block(
        header, "std::vector<int64_t> kvinsert_cost_weight_identity() const override")
