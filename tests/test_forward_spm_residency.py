"""Exercise the shared lifetime decision with the production SPM allocator."""
from pathlib import Path
import shutil
import subprocess

import pytest

from test_fmb_standalone_layout_rebuild import _definition

ROOT = Path(__file__).resolve().parents[1]


def test_forward_residency_preserves_capacity_and_rejects_unsafe_lifetimes(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    fmb = (ROOT / "src/core/fused_model_base.h").read_text()
    alloc_h = (ROOT / "src/core/rpu_spm_allocator.h").read_text()
    source = """
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <vector>
class SpmAllocator { public:
static constexpr size_t ALIGN = 256;
static size_t align_up(size_t n) {return (n+ALIGN-1)&~(ALIGN-1);}
"""
    for marker in ("struct AllocRequest", "struct AliasedPlan"):
        source += _definition(alloc_h, marker) + ";\n"
    source += "static AliasedPlan plan_temporary_aliased(const std::vector<AllocRequest>&);};\n"
    source += _definition((ROOT / "src/core/rpu_spm_allocator.cpp").read_text(),
                          "SpmAllocator::AliasedPlan SpmAllocator::plan_temporary_aliased(")
    source += "\nnamespace v3 {class FusedModelBase;\n"
    for marker in ("enum class StorageClass", "enum class BufferScope", "struct BufferDecl"):
        source += _definition(fmb, marker) + ";\n"
    source += "namespace detail {int64_t estimate_temporary_total(const std::vector<BufferDecl>&);}\n"
    source += _definition((ROOT / "src/core/fused_model_base.cpp").read_text(),
                          "int64_t detail::estimate_temporary_total(") + "\n}\n"
    header = (ROOT / "src/core/rpu_spm_residency.h").read_text()
    source += header[header.index("namespace v3::detail {"):]
    source += r'''
using namespace v3;
BufferDecl temp(const char* name,int64_t bytes,int first,int last,
                BufferScope scope=BufferScope::LayerWide) {
    return {name,bytes,first,last,StorageClass::Temp,0,nullptr,scope};
}
void unchanged(const std::vector<BufferDecl>& a,const std::vector<BufferDecl>& b) {
    assert(a.size()==b.size());
    for(size_t i=0;i<a.size();++i) {
        assert(a[i].name==b[i].name && a[i].size==b[i].size);
        assert(a[i].phase_start==b[i].phase_start && a[i].phase_end==b[i].phase_end);
        assert(a[i].scope==b[i].scope && a[i].storage==b[i].storage);
        assert(a[i].alias_of==b[i].alias_of && a[i].per_layer==b[i].per_layer);
    }
}
int main() {
    // Existing all-body slot: extending it does not consume new capacity.
    std::vector<BufferDecl> safe={temp("mask",256,0,9),temp("scratch",1024,0,9)};
    auto alias=temp("scratch_alias",1024,0,9);alias.alias_of="scratch";safe.push_back(alias);
    auto plan=detail::plan_forward_spm_residency(safe,{"mask"});
    assert(plan.retained && plan.original_peak_bytes==1280 && plan.retained_peak_bytes==1280);
    assert(safe[0].phase_start==0 && safe[0].phase_end==9);
    // Disjoint per-layer slots currently reuse bytes. Retention must fall back.
    std::vector<BufferDecl> growth={temp("mask",256,3,5),temp("hook",1024,0,0)};
    auto original=growth;
    plan=detail::plan_forward_spm_residency(growth,{"mask"});
    assert(!plan.retained && plan.original_peak_bytes==1024 && plan.retained_peak_bytes==1280);
    unchanged(growth,original);
    // A growing candidate is pure and carries the complete declarations. It
    // does not silently authorize the extra capacity or change the old caller.
    const auto candidate=detail::make_forward_spm_residency_candidate(growth,{"mask"});
    assert(candidate.valid && candidate.original_peak_bytes==1024 && candidate.retained_peak_bytes==1280);
    unchanged(growth,original);
    assert(candidate.declarations.size()==growth.size());
    assert(candidate.declarations[0].phase_start==0 && candidate.declarations[0].phase_end==5);
    assert(detail::estimate_temporary_total(candidate.declarations)==1280);
    // The allocator aliases outside-loop scopes even against LayerWide.
    auto outside=safe;outside.push_back(temp("pre",256,0,9,BufferScope::OutsideLayerLoop));
    auto saved=outside;assert(!detail::plan_forward_spm_residency(outside,{"mask"}).retained);
    unchanged(outside,saved);
    const auto unsafe=detail::make_forward_spm_residency_candidate(outside,{"mask"});
    assert(!unsafe.valid && unsafe.declarations.empty());
    // Aliasing the operand is another writer; unrelated aliases above are fine.
    auto writer=safe;writer.back().alias_of="mask";
    assert(!detail::plan_forward_spm_residency(writer,{"mask"}).retained);
    auto aliased=safe;aliased[0].alias_of="scratch";
    assert(!detail::plan_forward_spm_residency(aliased,{"mask"}).retained);
    auto perlayer=safe;perlayer[0].storage=StorageClass::TempPerLayer;
    assert(!detail::plan_forward_spm_residency(perlayer,{"mask"}).retained);
    auto missing=safe;assert(!detail::plan_forward_spm_residency(missing,{"absent"}).retained);
    assert(!detail::plan_forward_spm_residency(missing,{"mask","mask"}).retained);
    assert(!detail::plan_forward_spm_residency(missing,{nullptr}).retained);
    auto duplicate=safe;duplicate.push_back(temp("mask",256,0,9));
    assert(!detail::plan_forward_spm_residency(duplicate,{"mask"}).retained);
    std::cout << "capacity, outside-loop, alias, duplicate and missing-slot boundaries passed\n";
}
'''
    cpp = tmp_path / "residency.cpp"
    cpp.write_text(source)
    binary = tmp_path / "residency"
    subprocess.run([compiler, "-std=c++17", "-O2", str(cpp), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_common_descriptor_binding_preserves_owner_selection(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    header = (ROOT / "src/core/fused_model_base.h").read_text()
    source = (ROOT / "src/core/fused_model_base.cpp").read_text()
    program = r'''
#include <cassert>
#include <cstdint>
#include <optional>
#include <stdexcept>
#define TORCH_CHECK(c, ...) do {if (!(c)) throw std::runtime_error("rejected");} while(0)
enum class AttentionExecutionPolicy {DDR_KV,SPM_KV_BY_MHA};
enum class FmbRopeTableResidency {UNSPECIFIED,SPM};
'''
    for marker in ("enum class FmbForwardOperandResidency", "struct LayoutContext"):
        program += _definition(header, marker) + ";\n"
    program += r'''
enum class FmbPhysicalManifestState {UNSPECIFIED,COMPLETE};
struct FmbPhysicalExecutionManifest {
 FmbPhysicalManifestState state=FmbPhysicalManifestState::COMPLETE;
 uint64_t fingerprint=71;
};
uint64_t fmb_physical_manifest_fingerprint(const FmbPhysicalExecutionManifest& m){return m.fingerprint;}
FmbRopeTableResidency fmb_rope_table_residency(const FmbPhysicalExecutionManifest&){return FmbRopeTableResidency::SPM;}
struct FmbPhysicalManifestConsumer {
 explicit FmbPhysicalManifestConsumer(const FmbPhysicalExecutionManifest&) {}
 std::optional<AttentionExecutionPolicy> attention_layout_policy() const {
  return AttentionExecutionPolicy::SPM_KV_BY_MHA;
 }
};
struct FusedModelBase {
 FmbForwardOperandResidency selected=FmbForwardOperandResidency::UNSPECIFIED;
 FmbForwardOperandResidency physical_forward_operand_residency(const FmbPhysicalExecutionManifest&)const{return selected;}
 LayoutContext bind_physical_layout_context(LayoutContext,const FmbPhysicalExecutionManifest&)const;
};
'''
    program += _definition(source, "LayoutContext FusedModelBase::bind_physical_layout_context(")
    program += r'''
int main() {
 FusedModelBase owner; LayoutContext input; input.chunk_size=18;input.planning_chunk_capacity=32;
 FmbPhysicalExecutionManifest manifest;
 auto old=owner.bind_physical_layout_context(input,manifest);
 assert(old.forward_operand_residency==FmbForwardOperandResidency::UNSPECIFIED);
 for(auto mode:{FmbForwardOperandResidency::PER_LAYER,FmbForwardOperandResidency::FORWARD}) {
  owner.selected=mode; const auto bound=owner.bind_physical_layout_context(input,manifest);
  assert(bound.forward_operand_residency==mode && bound.physical_manifest_fingerprint==71);
  assert(bound.rope_table_residency==FmbRopeTableResidency::SPM);
  assert(bound.attention_policy==AttentionExecutionPolicy::SPM_KV_BY_MHA);
  assert(bound.chunk_size==18 && bound.planning_chunk_capacity==32);
 }
 owner.selected=static_cast<FmbForwardOperandResidency>(3);bool rejected=false;
 try{owner.bind_physical_layout_context(input,manifest);}catch(const std::runtime_error&){rejected=true;}
 assert(rejected);
 manifest.state=FmbPhysicalManifestState::UNSPECIFIED;
 const auto legacy=owner.bind_physical_layout_context(input,manifest);
 assert(legacy.forward_operand_residency==FmbForwardOperandResidency::UNSPECIFIED);
 assert(legacy.physical_manifest_fingerprint==0);
}
'''
    cpp, binary = tmp_path / "binding.cpp", tmp_path / "binding"
    cpp.write_text(program)
    subprocess.run([compiler, "-std=c++17", str(cpp), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
