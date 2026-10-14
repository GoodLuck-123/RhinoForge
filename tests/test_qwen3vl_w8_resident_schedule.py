"""Ordinary W8/packed FP16 residency admission, without hardware claims."""

import json
from pathlib import Path
import shutil
import subprocess

import pytest
from test_fmb_standalone_layout_rebuild import _definition


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/fused/rpu_qwen3vl_vision_model.cpp"


def _method(source, marker):
    start = source.index(marker)
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end].replace(" override", "")


def test_native_resident_schedule_admission(tmp_path):
    """Compile the production predicates: reject nearby profiles and chunking."""
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ is needed for native admission checks")
    source = SOURCE.read_text()
    helpers = (ROOT / "src/core/rpu_helpers.h").read_text()
    header = (ROOT / "src/core/fused_model_base.h").read_text()
    raw_predicate = "\n".join(_definition(helpers, marker) for marker in (
        "inline uint32_t GetMhaAttnUnivVlmNeed(",
        "inline bool sdpa_by_mha_spm_is_valid("))
    vlm_define = next(line for line in helpers.splitlines()
                      if line.startswith("#define SDPA_MAX_VLM_SIZE "))
    methods = "\n".join(_method(source, marker) for marker in (
        "bool ordinary_w8_split_raw_geometry() const",
        "bool ordinary_w8_resident_profile() const",
        "bool ordinary_w8_resident_chunk(",
        "bool ordinary_w8_split_raw_profile() const",
        "bool ordinary_w8_raw_chunk(",
        "ModelDynamicConfig dynamic_config("))
    harness = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
namespace at {
constexpr int kChar = 1, kHalf = 2;
struct Tensor { int dtype = kChar; int scalar_type() const { return dtype; } };
}
enum class ChunkMode { SEQUENTIAL, KV_FIRST };
enum class InterLayerIO { SPM_RESIDENT, DDR_PINGPONG };
''' + _definition(header, "enum class AttentionExecutionPolicy") + ";\n" + r'''
template<class T> T CeilDiv(T a,T b) { return (a+b-1)/b; }
''' + vlm_define + "\n" + raw_predicate + r'''
struct ChunkPlan { int64_t chunk_size, num_chunks; };
struct ModelDynamicConfig {
    ChunkMode chunk_mode;
    InterLayerIO inter_layer_io;
    AttentionExecutionPolicy attention_policy;
};
struct Weights { at::Tensor q_w, k_w, v_w, o_w, fc1_w, fc2_w; };
struct Vision {
    bool pipeline = false, delivery = false, adaptive = true, merger = true;
    bool linear_acc32_ = false;
    int64_t image_batch_count_ = 1, cores = 8, layers = 24, hidden = 1024;
    int64_t intermediate = 4096, heads = 16, dim = 64, orig_head_dim_ = 64;
    int64_t merger_out_hidden_ = 2560, current_num_patches_ = 784;
    std::vector<Weights> layer_weights_ = std::vector<Weights>(24);
    at::Tensor merger_m0_w_, merger_m2_w_;
    bool pipeline_dispatch_active() const { return pipeline; }
    bool qwen3vl_2b_delivery_profile_active() const { return delivery; }
    bool adaptive_w8_merger() const { return adaptive; }
    bool merger_active() const { return merger; }
    // This fixture covers the original route with later cold capabilities off.
    // Their enabled declarations and consumers use the raw-SPM harness.
    bool fp16_chunked_merger_active() const { return false; }
    bool compact_encoder_chunk(int64_t, int64_t) const { return false; }
    bool lingbot_residual_resident_chunk(int64_t) const { return false; }
    bool lingbot_spm_kv_chunk(int64_t) const { return false; }
    int64_t num_cores() const { return cores; }
    int64_t num_layers() const { return layers; }
    int64_t hidden_size() const { return hidden; }
    int64_t intermediate_size() const { return intermediate; }
    int64_t num_q_heads() const { return heads; }
    int64_t head_dim() const { return dim; }
''' + methods + r'''
};
int main() {
    for (int64_t out_hidden : {2048, 2560}) {
        Vision exact; exact.merger_out_hidden_ = out_hidden;
        assert(exact.ordinary_w8_resident_profile());
        auto plan = exact.dynamic_config({784, 1});
        assert(plan.inter_layer_io == InterLayerIO::SPM_RESIDENT);
        exact.current_num_patches_ = 256;
        assert(exact.ordinary_w8_raw_chunk(256, 256));
    }
    Vision good;
    assert(good.ordinary_w8_resident_chunk(784, 784));
    assert(good.ordinary_w8_resident_chunk(800, 784));
    assert(!good.ordinary_w8_resident_chunk(768, 784));
    assert(!good.ordinary_w8_resident_chunk(0, 0));
    auto one = good.dynamic_config({784, 1});
    assert(one.chunk_mode == ChunkMode::SEQUENTIAL);
    assert(one.inter_layer_io == InterLayerIO::SPM_RESIDENT);
    assert(one.attention_policy == AttentionExecutionPolicy::DDR_KV);
    auto many = good.dynamic_config({256, 4});
    assert(many.chunk_mode == ChunkMode::KV_FIRST);
    assert(many.inter_layer_io == InterLayerIO::DDR_PINGPONG);
    assert(many.attention_policy == AttentionExecutionPolicy::DDR_KV);
    for (int64_t rows : {256, 512, 768, 1024}) {
        Vision v; v.current_num_patches_ = rows;
        auto cfg = v.dynamic_config({rows, 1});
        assert(cfg.chunk_mode == ChunkMode::SEQUENTIAL);
        assert(cfg.inter_layer_io == InterLayerIO::SPM_RESIDENT);
        assert(cfg.attention_policy == (rows == 1024
            ? AttentionExecutionPolicy::DDR_KV : AttentionExecutionPolicy::AUTO));
        assert(v.ordinary_w8_raw_chunk(rows, rows) == (rows != 1024));
        assert(v.dynamic_config({128, rows / 128}).attention_policy == AttentionExecutionPolicy::DDR_KV);
    }
    Vision pair; pair.current_num_patches_ = 512; pair.image_batch_count_ = 2;
    assert(pair.ordinary_w8_resident_chunk(512,512));
    assert(pair.ordinary_w8_raw_chunk(512,512));
    assert(pair.dynamic_config({512,1}).inter_layer_io == InterLayerIO::SPM_RESIDENT);
    assert(pair.dynamic_config({512,1}).attention_policy == AttentionExecutionPolicy::AUTO);
    assert(!pair.ordinary_w8_raw_chunk(256,512));
    assert(!pair.ordinary_w8_resident_chunk(1024,1024));
    pair.current_num_patches_=1024;
    assert(pair.ordinary_w8_resident_profile());
    assert(pair.ordinary_w8_raw_chunk(1024,1024));
    assert(pair.dynamic_config({1024,1}).attention_policy == AttentionExecutionPolicy::AUTO);
    assert(!pair.ordinary_w8_raw_chunk(512,512));
    Vision triple; triple.image_batch_count_=3; triple.current_num_patches_=768;
    assert(triple.ordinary_w8_resident_chunk(768,768));
    assert(triple.ordinary_w8_raw_chunk(768,768));
    assert(triple.dynamic_config({768,1}).attention_policy == AttentionExecutionPolicy::AUTO);
    assert(!triple.ordinary_w8_raw_chunk(768,767));
    for (int64_t count : {2,3}) for (int64_t image_rows : {4,12,132,196,252}) {
        Vision v; v.image_batch_count_=count; v.current_num_patches_=count*image_rows;
        const int64_t capacity=CeilDiv(v.current_num_patches_,int64_t{16})*16;
        assert(v.ordinary_w8_split_raw_profile());
        assert(v.ordinary_w8_raw_chunk(v.current_num_patches_,v.current_num_patches_));
        assert(v.ordinary_w8_raw_chunk(capacity,v.current_num_patches_));
        assert(!v.ordinary_w8_raw_chunk(capacity+16,v.current_num_patches_));
        const auto cfg=v.dynamic_config({v.current_num_patches_,1});
        assert(cfg.chunk_mode==ChunkMode::SEQUENTIAL);
        assert(cfg.inter_layer_io==InterLayerIO::SPM_RESIDENT);
        assert(cfg.attention_policy==AttentionExecutionPolicy::SPM_KV_BY_MHA);
        assert(!sdpa_by_mha_spm_is_valid(count,image_rows,image_rows,16,16,64,8,0));
        assert(sdpa_by_mha_spm_is_valid(1,image_rows,image_rows,16,16,64,8,0));
    }
    for (int64_t image_rows : {0,1,3,6,16,192,256,260}) {
        Vision v;v.image_batch_count_=3;v.current_num_patches_=3*image_rows;
        assert(!v.ordinary_w8_split_raw_geometry());
    }
    for(bool fp16_encoder:{false,true})for(int64_t batch:{2,3}){
        Vision gr00t;gr00t.image_batch_count_=batch;gr00t.current_num_patches_=196*batch;
        gr00t.adaptive=false;gr00t.merger_m0_w_.dtype=gr00t.merger_m2_w_.dtype=at::kHalf;
        if(fp16_encoder)for(auto&w:gr00t.layer_weights_)
            w.q_w.dtype=w.k_w.dtype=w.v_w.dtype=w.o_w.dtype=w.fc1_w.dtype=w.fc2_w.dtype=at::kHalf;
        assert(gr00t.ordinary_w8_split_raw_profile());
        assert(gr00t.dynamic_config({196*batch,1}).attention_policy==AttentionExecutionPolicy::SPM_KV_BY_MHA);
        for(auto projection:{&Weights::q_w,&Weights::k_w,&Weights::v_w,&Weights::o_w,&Weights::fc1_w,&Weights::fc2_w}){
            auto bad=gr00t;(bad.layer_weights_.back().*projection).dtype=fp16_encoder?at::kChar:at::kHalf;
            assert(!bad.ordinary_w8_split_raw_profile());
        }
        auto missing=gr00t;missing.merger=false;assert(!missing.ordinary_w8_split_raw_profile());
        auto mixed=gr00t;mixed.merger_m2_w_.dtype=at::kChar;assert(!mixed.ordinary_w8_split_raw_profile());
        gr00t.current_num_patches_=256*batch;
        assert(gr00t.ordinary_w8_resident_profile());
        assert(gr00t.ordinary_w8_raw_chunk(256*batch,256*batch));
        assert(gr00t.dynamic_config({256*batch,1}).attention_policy==AttentionExecutionPolicy::AUTO);
        auto acc32=gr00t;acc32.linear_acc32_=true;
        assert(!acc32.ordinary_w8_resident_profile());
        assert(acc32.dynamic_config({256*batch,1}).attention_policy==AttentionExecutionPolicy::DDR_KV);
        auto mixed_layer=gr00t;mixed_layer.layer_weights_[12].k_w.dtype=3;
        assert(!mixed_layer.ordinary_w8_resident_profile());
        gr00t.current_num_patches_=512*batch;
        assert(!gr00t.ordinary_w8_resident_profile());
        gr00t.image_batch_count_=1;gr00t.current_num_patches_=256;
        assert(!gr00t.ordinary_w8_resident_profile());
    }
    for (int64_t count : {2,3,4}) for (int64_t per_image : {128,256,512,768}) {
        Vision v; v.image_batch_count_=count; v.current_num_patches_=count*per_image;
        const bool allowed=(count==2&&(per_image==256||per_image==512)) ||
                           (count==3&&per_image==256);
        assert(v.ordinary_w8_resident_profile()==allowed);
        assert(v.ordinary_w8_raw_chunk(v.current_num_patches_,v.current_num_patches_)==allowed);
    }
    for (auto member : {&Vision::pipeline, &Vision::delivery, &Vision::linear_acc32_}) {
        Vision v; v.*member = true;
        assert(!v.ordinary_w8_resident_profile());
        assert(!v.ordinary_w8_raw_chunk(256, 256));
    }
    for (auto member : {&Vision::image_batch_count_, &Vision::cores, &Vision::layers,
                        &Vision::hidden, &Vision::intermediate, &Vision::heads,
                        &Vision::dim, &Vision::orig_head_dim_, &Vision::merger_out_hidden_}) {
        Vision v; ++(v.*member);
        assert(!v.ordinary_w8_resident_profile());
        assert(!v.ordinary_w8_raw_chunk(256, 256));
    }
    for (auto projection : {&Weights::q_w, &Weights::k_w, &Weights::v_w,
                           &Weights::o_w, &Weights::fc1_w, &Weights::fc2_w}) {
        Vision v; (v.layer_weights_.back().*projection).dtype = 2;
        assert(!v.ordinary_w8_resident_profile());
        assert(!v.ordinary_w8_raw_chunk(256, 256));
    }
    Vision missing; missing.layer_weights_.pop_back();
    assert(!missing.ordinary_w8_resident_profile());
    Vision fp16; fp16.adaptive = false;
    assert(!fp16.ordinary_w8_resident_profile());
    assert(fp16.dynamic_config({784, 1}).chunk_mode == ChunkMode::KV_FIRST);
    assert(fp16.dynamic_config({784, 1}).inter_layer_io == InterLayerIO::DDR_PINGPONG);
    assert(fp16.dynamic_config({784, 1}).attention_policy == AttentionExecutionPolicy::DDR_KV);
    bool rejected = false;
    try { fp16.dynamic_config({256, 4}); } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    Vision delivery; delivery.delivery = true; delivery.current_num_patches_ = 576;
    assert(delivery.dynamic_config({576, 1}).chunk_mode == ChunkMode::SEQUENTIAL);
    assert(delivery.dynamic_config({576, 1}).attention_policy == AttentionExecutionPolicy::DDR_KV);
    delivery.current_num_patches_ = 784;
    rejected = false;
    try { delivery.dynamic_config({784, 1}); } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
}
'''
    cpp = tmp_path / "resident.cpp"
    exe = tmp_path / "resident"
    cpp.write_text(harness)
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    str(cpp), "-o", str(exe)], check=True, capture_output=True, text=True)
    subprocess.run([str(exe)], check=True, capture_output=True, text=True)


def test_resident_dataflow_and_complete_route_are_separate_from_delivery():
    source = SOURCE.read_text()
    kv = _method(source, "void emit_kv_first_body(")
    layer = _method(source, "void build_layer_subgraph(")
    buffers = _method(source, "std::vector<BufferDecl> declare_buffers(")
    manifest = _method(source, "FmbPhysicalExecutionManifest make_physical_manifest_for_candidate(")
    assert "if (!ctx().input_in_spm)" in kv
    assert kv.index("ChunkMode::SEQUENTIAL") < kv.index("q_ddr_slots_.count")
    assert 'if (sequential) {\n            emit_kv_first_body(layer_idx, chunk);' in layer
    assert 'addr_offset(sequential ? "q" : "q_comp").value' in layer
    assert "/*qwen3vl_tm160=*/delivery" in layer
    assert "if (!ctx().output_to_spm && !pipeline_final)" in layer
    assert "QWEN3VL_DEEPSTACK_SNAPSHOT_SITE" in layer
    assert "qwen3vl_2b_delivery_profile_active() || ordinary_resident" in buffers
    assert 'return std::string(decl.name) == "q_comp";' in buffers
    assert "ordinary_w8_resident_schedule_for_plan(plan, physical_len)" in manifest
    schedule = _method(source, "bool ordinary_w8_resident_schedule_for_plan(")
    assert "ordinary_w8_resident_chunk(plan.compute.plan.chunk_size, sequence_length)" in schedule
    assert "plan.chunk_mode == ChunkMode::SEQUENTIAL" in manifest
    assert "physical_len == logical_len" in manifest
    assert "consume_ordinary_w8_resident_schedule();" in layer
    site = "293709511256992144"
    assert f"QWEN3VL_W8_RESIDENT_SCHEDULE_SITE = {site}LL" in source
