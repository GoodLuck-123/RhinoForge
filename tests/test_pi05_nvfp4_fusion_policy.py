"""Execute the native NVFP4 producer admission against stale/cross-format routes."""
import importlib
from pathlib import Path
import json
import re
import runpy
import subprocess

from test_siglip_attention_dry_live_contract import _function

ROOT = Path(__file__).resolve().parents[1]


def test_nvfp4_geglu_rejects_incomplete_stale_or_cross_format_descriptor(tmp_path):
    source = (ROOT / "src/fused/rpu_pi05_denoise_step_model.cpp").read_text()
    header = (ROOT / "src/fused/rpu_pi05_denoise_step_model.h").read_text()
    policy = _function(header, "    FmbLinearAccumulationPolicy linear_accumulation_policy(")
    names = ("pi05_nvfp4_geglu_m50_arguments", "use_pi05_nvfp4_geglu_m50",
             "validate_pi05_nvfp4_geglu_m50_manifest")
    methods = [_function(source, "Pi05DenoiseStepModel::" + name + "(") for name in names]
    bodies = [method[method.index("{"):] for method in methods]
    constants = "\n".join(re.findall(
        r"constexpr int64_t PI05_DENOISE_(?:NVFP4_GEGLU_\w+|W4_GEGLU_M50_\w+_SITE|GEGLU_M50_\w+_SITE)\s*=.*?;",
        source, re.S))
    cpp = r'''
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while (false)
enum class KernelId { PI05_DENOISE_GATE_UP_GEGLU_NVFP4_M320N64K128,
    PI05_DENOISE_GATE_UP_GEGLU_NVFP4_ACC16_M320N64K128 };
struct KernelCache {
    bool loaded=true;
    static KernelCache& instance() { static KernelCache cache;return cache; }
    bool has_loaded(KernelId id) const {
        assert(id==KernelId::PI05_DENOISE_GATE_UP_GEGLU_NVFP4_ACC16_M320N64K128);
        return loaded;
    }
};
enum class FmbPhysicalManifestState { UNSPECIFIED, COMPLETE };
enum class FmbGraphLifecycle { UNSPECIFIED, COMPOSITE_CHILD };
enum class FmbLinearAccumulationPolicy { ACC16, ACC32, MIXED_BY_SITE };
enum class FmbRouteFamily { LINEAR, GRAPH_SCHEDULE };
__CONSTANTS__
struct Route { int64_t site_id; FmbRouteFamily family; int64_t selector, flags, invocation;
    std::vector<int64_t> arguments; };
struct FmbPhysicalExecutionManifest {
    FmbPhysicalManifestState state=FmbPhysicalManifestState::COMPLETE;
    int64_t logical_length=50,physical_length=50,execution_padding_rows=0,
            kv_logical_length=850,kv_insert_physical_rows=64;
    FmbGraphLifecycle graph_lifecycle=FmbGraphLifecycle::COMPOSITE_CHILD;
    FmbLinearAccumulationPolicy linear_accumulation=FmbLinearAccumulationPolicy::ACC16;
    std::vector<Route> routes;
};
struct Model {
    bool linear_acc32_=false;
__POLICY__
    int64_t planned_prefix_len_=800, planned_num_steps_=10;
    int cores=8,attention_cores=8,mlp_cores=8;
    int num_cores() const {return cores;} int attn_tp() const {return attention_cores;}
    int mlp_tp() const {return mlp_cores;}
    bool pi05_nvfp4_geglu_m50_opt_in_=true, owners=true, continuation=true;
    struct { bool value=true; bool enabled() const { return value; } } nvfp4_;
    bool pi05_nvfp4_weight_profile_admitted() const { return owners; }
    bool use_pi05_xor3_gated_mlp(int64_t rows) const { return continuation && rows==50; }
    std::vector<int64_t> pi05_w4_fastpath_arguments() const { return {2,16,8,32}; }
    void validate_pi05_w4_fastpath_manifest(const FmbPhysicalExecutionManifest&) const { TORCH_CHECK(owners); }
    void validate_pi05_xor3_gated_mlp_manifest(const FmbPhysicalExecutionManifest&) const { TORCH_CHECK(continuation); }
    std::vector<int64_t> pi05_nvfp4_geglu_m50_arguments() const __ARGS__
    bool use_pi05_nvfp4_geglu_m50(int64_t rows) const __USE__
    void validate_pi05_nvfp4_geglu_m50_manifest(const FmbPhysicalExecutionManifest& manifest) const __VALIDATE__
};
int main() {
    Model model;
    FmbPhysicalExecutionManifest valid;
    valid.routes={{PI05_DENOISE_NVFP4_GEGLU_POLICY_SITE,FmbRouteFamily::GRAPH_SCHEDULE,1,0,0,
                   model.pi05_nvfp4_geglu_m50_arguments()},
                  {PI05_DENOISE_NVFP4_GEGLU_LINEAR_SITE,FmbRouteFamily::LINEAR,
                   PI05_DENOISE_NVFP4_GEGLU_ACC16_M320N64K128,0,0,model.pi05_nvfp4_geglu_m50_arguments()}};
    model.validate_pi05_nvfp4_geglu_m50_manifest(valid);
    auto reject=[&](const FmbPhysicalExecutionManifest& m) {
        try { model.validate_pi05_nvfp4_geglu_m50_manifest(m);assert(false); }
        catch (const std::runtime_error&) {}
    };
    for (int mutation=0;mutation<24;++mutation) {
        auto bad=valid;
        switch(mutation) {
          case 0: bad.state=FmbPhysicalManifestState::UNSPECIFIED;break;
          case 1: bad.physical_length=64;break;
          case 2: bad.logical_length=49;break;
          case 3: bad.execution_padding_rows=14;break;
          case 4: bad.kv_logical_length=849;break;
          case 5: bad.kv_insert_physical_rows=50;break;
          case 6: bad.graph_lifecycle=FmbGraphLifecycle::UNSPECIFIED;break;
          case 7: bad.linear_accumulation=FmbLinearAccumulationPolicy::ACC32;break;
          case 8: bad.routes.erase(bad.routes.begin());break;
          case 9: bad.routes.pop_back();break;
          case 10: bad.routes.push_back(bad.routes[0]);break;
          case 11: bad.routes.push_back(bad.routes[1]);break;
          case 12: bad.routes[1].site_id=123;break;
          case 13: bad.routes[1].selector=1;break;
          case 14: bad.routes[1].flags=1;break;
          case 15: bad.routes[1].invocation=1;break;
          case 16: bad.routes[0].arguments[0]=99;break;
          case 17: bad.routes[1].arguments[28]=128;break; // INT4 K128 cannot replace NVFP4 K16
          case 18: bad.routes.push_back({PI05_DENOISE_W4_GEGLU_M50_LINEAR_SITE,FmbRouteFamily::LINEAR,1,0,0,{}});break;
          case 19: bad.routes.push_back({PI05_DENOISE_GEGLU_M50_LINEAR_SITE,FmbRouteFamily::LINEAR,1,0,0,{}});break;
          case 20: bad.routes[1].selector=PI05_DENOISE_NVFP4_GEGLU_M320N64K128;break;
          case 21: bad.routes[1].arguments[0]=1;break; // Retired ACC32 argument ABI.
          case 22: bad.routes[1].arguments[2]=static_cast<int64_t>(KernelId::PI05_DENOISE_GATE_UP_GEGLU_NVFP4_M320N64K128);break;
          case 23: bad.routes[1].arguments[34]=32;break; // ACC32 producer arithmetic.

        }
        reject(bad);
    }
    model.owners=false;reject(valid);model.owners=true;
    model.linear_acc32_=true;
    assert(model.linear_accumulation_policy()==FmbLinearAccumulationPolicy::ACC32);
    reject(valid);
    auto acc32=valid;acc32.linear_accumulation=FmbLinearAccumulationPolicy::ACC32;
    reject(acc32); // ACC32 remains on decomposed projections.
    model.linear_acc32_=false;
    KernelCache::instance().loaded=false;reject(valid);KernelCache::instance().loaded=true;
    for (int* domain : {&model.cores,&model.attention_cores,&model.mlp_cores}) {
        for (int reduced : {4,6}) {*domain=reduced;reject(valid);}
        *domain=8;
    }
    for (int steps : {1,4,9,11}) {model.planned_num_steps_=steps;reject(valid);}
    model.planned_num_steps_=10;
    model.continuation=false;reject(valid);model.continuation=true;
    model.nvfp4_.value=false;reject(valid);model.nvfp4_.value=true;
    model.pi05_nvfp4_geglu_m50_opt_in_=false;reject(valid);
    auto empty=valid;empty.routes.clear();model.validate_pi05_nvfp4_geglu_m50_manifest(empty);
    model.pi05_nvfp4_geglu_m50_opt_in_=true;
    model.planned_prefix_len_=544;
    // The Action shape stays M50/PAD64, but a P800 logical-KV descriptor
    // cannot be reused for a true two-camera prefix.
    reject(valid);
    valid.kv_logical_length=594;
    model.validate_pi05_nvfp4_geglu_m50_manifest(valid);
    valid.kv_logical_length=608;reject(valid); // physical insertion end is not attention length
    for (int prefix : {672,928}) {
        model.planned_prefix_len_=prefix;
        reject(valid); // A previous profile's KV length cannot be reused.
        valid.kv_logical_length=prefix+50;
        model.validate_pi05_nvfp4_geglu_m50_manifest(valid);
    }
    for (int prefix : {0,543,576,608,640,799,801,832,864,896}) {
        model.planned_prefix_len_=prefix;
        valid.kv_logical_length=prefix+50;
        reject(valid); // The new producer is limited to actual two/three-image T32/T160.
    }
}
'''
    cpp = cpp.replace("__POLICY__", policy)
    for key, value in zip(("ARGS", "USE", "VALIDATE"), bodies):
        cpp = cpp.replace("__" + key + "__", value)
    cpp = cpp.replace("__CONSTANTS__", constants)
    path, binary = tmp_path / "nvfp4-policy.cpp", tmp_path / "nvfp4-policy"
    path.write_text(cpp)
    compiled = subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", str(path), "-o", str(binary)],
                              capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=5)
    assert run.returncode == 0, run.stderr


def test_nvfp4_geglu_launchers_keep_distinct_ids_and_typed_ddr_abi(tmp_path):
    """Execute both production launchers through their real shared ABI writer."""
    source = (ROOT / "src/ops/rpu_linear.cpp").read_text()
    functions = "\n".join(_function(source, marker) for marker in (
        "static void launch_pi05_denoise_gate_up_geglu_nvfp4_m50(",
        "void rpu_launch_pi05_denoise_gate_up_geglu_nvfp4_m50_kernel(",
        "void rpu_launch_pi05_denoise_gate_up_geglu_nvfp4_acc16_m50_kernel("))
    cpp = r'''
#include <cassert>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while(false)
constexpr int NUM_CORES=8;
namespace c10 {using Half=uint16_t;}
namespace at {
constexpr int kPrivateUse1=1,kByte=2;
using IntArrayRef=std::vector<int64_t>;
struct Tensor {
    uint64_t address;IntArrayRef shape;int dtype=kByte;
    bool defined()const{return true;}
    struct Device {int type()const{return kPrivateUse1;}};
    Device device()const{return {};}
    int scalar_type()const{return dtype;}
    bool is_contiguous()const{return true;}
    int storage_offset()const{return 0;}
    const IntArrayRef& sizes()const{return shape;}
    int64_t nbytes()const{return shape[0]*shape[1];}
    void* data_ptr()const{return reinterpret_cast<void*>(address);}
};
}
struct SpmAllocator {
    static constexpr uint64_t SPM_USABLE=8387584;
    bool is_initialized()const{return true;}
    uint32_t addr(int,int)const{return 0x100000;}
} allocator;
#define SPM_ALLOC allocator
int flushed=0;
void rpu_ddr_flush(void*){++flushed;}
uint64_t RpuGetDevAddr(void* ptr){return reinterpret_cast<uint64_t>(ptr);}
enum class KernelId {PI05_DENOISE_GATE_UP_GEGLU_NVFP4_M320N64K128,
    PI05_DENOISE_GATE_UP_GEGLU_NVFP4_ACC16_M320N64K128};
enum class GraphDdrRegisterRole {PackedFp4ModelWeight,Fp8BlockScale};
enum class GraphDdrRegisterAccess {Read};
enum class GraphDdrRegisterEncoding {DevAddrShift8LoHi};
struct Abi {GraphDdrRegisterRole role;GraphDdrRegisterAccess access;
    GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec {Abi abi;at::Tensor owner;};
std::vector<GraphDdrRegisterOperandSpec> staged;
KernelId staged_id,looked_up_id;
struct Kernel {
    std::map<int,uint16_t> regs;
    std::vector<int> full_word_writes;
    void reset_regs(){regs.clear();full_word_writes.clear();}
    void set_regs(int reg,uint16_t value){regs[reg]=value;}
    void set_regs(int reg,uint32_t value){
        assert(reg%2==0);full_word_writes.push_back(reg);
        regs[reg]=value&0xffff;regs[reg+1]=value>>16;
    }
} kernel;
struct Writer {
    void write(Kernel& k) {
        for(const auto& operand:staged) {
            const auto encoded=operand.owner.address>>8;
            k.set_regs(operand.abi.lo,static_cast<uint16_t>(encoded&0xffff));
            k.set_regs(operand.abi.hi,static_cast<uint16_t>(encoded>>16));
        }
    }
};
struct RpuKernelGraph {
    static RpuKernelGraph& active(){static RpuKernelGraph graph;return graph;}
    Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec> operands) {
        staged_id=id;staged=std::move(operands);return {};
    }
};
Kernel* get_kernel(KernelId id){looked_up_id=id;return &kernel;}
#define GET_KERNEL(id) get_kernel(id)
struct Queue {
    bool broadcast=false;int launches=0;
    void set_broadcast_mode(bool value){broadcast=value;}
    void enqueu_kernel(Kernel&,std::vector<uint16_t> grid,std::vector<uint8_t> cores){
        assert(broadcast && staged_id==looked_up_id);
        assert(grid==std::vector<uint16_t>({8,1,1}));
        assert(cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));
        ++launches;
    }
} queue;
Queue* get_queue(int cores){assert(cores==8);queue.broadcast=false;return &queue;}
#define GET_QUEUE(cores) get_queue(cores)
__FUNCTIONS__
int main(){
    const at::Tensor gate{0x2000000,{4096,512}},up{0x2400000,{4096,512}},
        gate_scale{0x2800000,{64,4096}},up_scale{0x2900000,{64,4096}};
    const auto u32=[](int reg){return uint32_t(kernel.regs.at(reg))|(uint32_t(kernel.regs.at(reg+1))<<16);};
    for(bool acc16:{false,true,false,true}) {
        const auto launch=acc16 ? rpu_launch_pi05_denoise_gate_up_geglu_nvfp4_acc16_m50_kernel
                                : rpu_launch_pi05_denoise_gate_up_geglu_nvfp4_m50_kernel;
        staged.clear();flushed=0;
        launch(0x100000,gate,up,0x120000,gate_scale,up_scale,0x130000,0x130100,17);
        assert(staged_id==(acc16 ? KernelId::PI05_DENOISE_GATE_UP_GEGLU_NVFP4_ACC16_M320N64K128
                                : KernelId::PI05_DENOISE_GATE_UP_GEGLU_NVFP4_M320N64K128));
        assert(staged.size()==4 && flushed==4);
        const int registers[]={2,20,32,34};
        const at::Tensor* owners[]={&gate,&gate_scale,&up,&up_scale};
        for(size_t i=0;i<staged.size();++i){
            const auto& operand=staged[i];
            assert(operand.abi.role==(i%2 ? GraphDdrRegisterRole::Fp8BlockScale
                                          : GraphDdrRegisterRole::PackedFp4ModelWeight));
            assert(operand.abi.access==GraphDdrRegisterAccess::Read);
            assert(operand.abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi);
            assert(operand.abi.lo==registers[i] && operand.abi.hi==registers[i]+1);
            assert(operand.owner.address==owners[i]->address);
            assert(u32(registers[i])==owners[i]->address>>8);
        }
        assert(u32(0)==0x100000 && u32(4)==0x120000 && u32(6)==0);
        assert(kernel.regs.at(8)==50 && u32(10)==512 && kernel.regs.at(12)==1024);
        assert(u32(24)==0x130000 && u32(36)==0x130100 && kernel.regs.at(27)==17);
        if(acc16){
            // Every launch initializes both complete core counter words. A
            // halfword-only reset leaves stale high bits from earlier kernels.
            assert(kernel.full_word_writes==std::vector<int>({128,130}));
            assert(u32(128)==0 && u32(130)==0);
        }else{
            assert(kernel.full_word_writes.empty());
            for(int reg=128;reg<=255;++reg)assert(kernel.regs.count(reg)==0);
        }
        const auto launches=queue.launches;
        for(int mutation=0;mutation<4;++mutation){
            auto bad=gate;
            if(mutation==0) bad.dtype=0;
            if(mutation==1) bad.shape={4096,1024};
            bool rejected=false;
            try{launch(0x100000,bad,up,mutation==2?0x100000:0x120000,
                gate_scale,up_scale,0x130000,0x130100,mutation==3?18:0);}
            catch(const std::runtime_error&){rejected=true;}
            assert(rejected && queue.launches==launches);
        }
    }
    assert(queue.launches==4);
}
'''.replace("__FUNCTIONS__", functions)
    path, binary = tmp_path / "nvfp4-launchers.cpp", tmp_path / "nvfp4-launchers"
    path.write_text(cpp)
    built = subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", str(path), "-o", str(binary)],
                           capture_output=True, text=True, timeout=30)
    assert built.returncode == 0, built.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr


def test_nvfp4_cold_policy_preserves_uniform_arithmetic_with_optimized_flags(tmp_path):
    source = (ROOT / "src/fused/rpu_pi05_denoise_step_model.cpp").read_text()
    header = (ROOT / "src/fused/rpu_pi05_denoise_step_model.h").read_text()
    constructor = _function(source, "Pi05DenoiseStepModel::Pi05DenoiseStepModel(")
    constructor = constructor.replace("Pi05DenoiseStepModel::Pi05DenoiseStepModel", "Model")
    fields = "\n".join(re.findall(r"    const (?:bool|int64_t) \w+_;", header))
    policy = _function(header, "    FmbLinearAccumulationPolicy linear_accumulation_policy(")
    profile = vars(importlib.import_module("examples.runners.pi05.optimized"))["profile_environment"](
        "w4a16", prefill_w8a8=False, vision_owner_ln=True, cameras=3)
    environment = ",\n".join("{" + json.dumps(key) + ", " + str(value == "1").lower() + "}"
                             for key, value in profile.items())
    code = r'''
#include <cassert>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while (false)
enum class FmbLinearAccumulationPolicy { ACC16, ACC32, MIXED_BY_SITE };
std::map<std::string, bool> environment = {__ENVIRONMENT__};
bool pi05_kernel_opt_in(const char* name) { return environment[name]; }
int64_t pi05_gateup_bundle_layers() { return 0; }
struct Model {
__FIELDS__
__CONSTRUCTOR__
__POLICY__
};
int main() {
    assert(!environment["RPU_PI05_DENOISE_NVFP4_GEGLU_M50"]);
    // A legacy inherited opt-in cannot re-enable the retired mixed stack.
    environment["RPU_PI05_DENOISE_NVFP4_GEGLU_M50"] = true;
    assert(environment["RPU_PI05_DENOISE_W4_FASTPATH"]);
    for (bool new_flag : {false,true}) for (bool acc32 : {false, true}) {
        environment["RPU_PI05_DENOISE_NVFP4_GEGLU_ACC16_M50"] = new_flag;
        Model model(acc32);
        assert(model.linear_accumulation_policy() == (acc32
            ? FmbLinearAccumulationPolicy::ACC32 : FmbLinearAccumulationPolicy::ACC16));
        // Only the new ACC16 flag enables GeGLU; inherited legacy flags cannot
        // reopen the mixed Gate/Up32 + KV16 stack or change ACC32 projections.
        assert(model.pi05_nvfp4_geglu_m50_opt_in_ == (new_flag && !acc32));
        assert(model.pi05_w4_fastpath_opt_in_ == !acc32);
        assert(model.pi05_kv1_pair_owner_opt_in_ == !acc32);
        assert(model.pi05_xor3_gated_mlp_opt_in_ == !acc32);
        assert(!model.pi05_geglu_m50_opt_in_ && !model.pi05_w4_geglu_m50_opt_in_);
    }
}
'''
    for key, value in (("ENVIRONMENT", environment), ("FIELDS", fields),
                       ("CONSTRUCTOR", constructor), ("POLICY", policy)):
        code = code.replace("__" + key + "__", value)
    path, binary = tmp_path / "nvfp4-cold.cpp", tmp_path / "nvfp4-cold"
    path.write_text(code)
    compiled = subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", str(path), "-o", str(binary)],
                              capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=5)
    assert run.returncode == 0, run.stderr
