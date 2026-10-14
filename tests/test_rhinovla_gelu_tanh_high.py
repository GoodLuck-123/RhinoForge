"""Execute the vision encoder's actual precision dispatch and layout identity."""

import re

from test_high_precision_math_launch import COMMON, _block, _read, _run


def test_encoder_high_keeps_tanh_formula_and_versions_its_layout(tmp_path):
    source = _read("src/fused/rpu_qwen3vl_vision_model.cpp")
    start = source.index("if (rhinovla_high_precision_)", source.index("// Phase 5:"))
    dispatch = source[start:source.index("// Phase 6:", start)].rstrip().removesuffix("}")
    layout = _block(source, "int64_t subclass_layout_hash() const override").replace(" override", "")
    site = re.search(r"RHINOVLA_VISION_HIGH_ACTIVATION_SITE = (\d+)LL", source)[1]
    encoder_site = re.search(r"RHINOVLA_VISION_HIGH_ENCODER_ACTIVATION_SITE = (\d+)LL", source)[1]
    modes = _block(_read("src/core/rpu_kernel_decls.h"), "enum class GeluMode") + ";"
    precision = _block(_read("src/core/rpu_math_precision.h"), "enum class RpuUnaryPrecision") + ";"
    lingbot_sites = "\n".join(re.findall(r"(?:static )?constexpr int64_t QWEN3VL_LINGBOT_[A-Z0-9_]+_SITE\s*=\s*[^;]+;", source))
    program = lingbot_sites + r'''
enum class FmbRouteFamily { GRAPH_SCHEDULE };
enum class ValuOpType { ADD };
constexpr int64_t RHINOVLA_VISION_HIGH_ACTIVATION_SITE = __SITE_VALUE__;
constexpr int64_t RHINOVLA_VISION_HIGH_ENCODER_ACTIVATION_SITE = __ENCODER_SITE_VALUE__;
namespace detail {
int64_t layout_mix(int64_t a, int64_t b) { return a ^ (b + 33); }
}
struct Context {
    int calls = 0, version = 2;
    void consume_physical_route(FmbRouteFamily family, int64_t site, int64_t selector,
                                int64_t flags, std::vector<int64_t> args) {
        assert(family == FmbRouteFamily::GRAPH_SCHEDULE &&
               site == RHINOVLA_VISION_HIGH_ENCODER_ACTIVATION_SITE && selector == 1 && flags == 0);
        TORCH_CHECK(args == std::vector<int64_t>({version,1024,4096,2048,4,8}));
        ++calls;
    }
};
int launches = 0;
RpuUnaryPrecision dispatched;
void rpu_launch_eltwise_unary_spm_kernel(uint32_t x, uint32_t y, int64_t count,
        ValuOpType op, GeluMode mode, int cores, RpuUnaryPrecision precision) {
    assert(x == 0x40001000 && y == x && count == 448 * 512 && cores == 8);
    assert(op == ValuOpType::ADD && mode == GeluMode::TANH);
    dispatched = precision; ++launches;
}
struct Vision {
    bool rhinovla_high_precision_ = false;
    int lingbot_fc1_gelu_mode_=0, lingbot_qkv_joint_mode_=0, lingbot_residual_resident_mode_=0, lingbot_spm_kv_mode_=0;
    std::vector<int64_t> lingbot_qkv_joint_arguments() const{return {};}
    std::vector<int64_t> lingbot_spm_kv_arguments() const{return {};}
    Context context;
    Context& ctx() { return context; }
    int num_cores() const { return 8; }
    int attn_tp() const { return 8; }
    int mlp_tp() const { return 8; }
    uint32_t addr(int core, const char*) { assert(core == 0); return 0x40001000; }
    LAYOUT
    void dispatch() { int64_t seq_len = 448, local_inter = 512; DISPATCH }
};
int main() {
    Vision model;
    assert(model.subclass_layout_hash() == 0);
    model.dispatch();
    assert(dispatched == RpuUnaryPrecision::BASE && model.context.calls == 0);
    model.rhinovla_high_precision_ = true;
    const int64_t old_hash = detail::layout_mix(0, RHINOVLA_VISION_HIGH_ACTIVATION_SITE);
    assert(model.subclass_layout_hash() != 0 && model.subclass_layout_hash() != old_hash);
    model.dispatch(); model.dispatch();
    assert(dispatched == RpuUnaryPrecision::HIGH && model.context.calls == 2);
    const int before = launches;
    model.context.version = 1;
    rejects([&] { model.dispatch(); });
    assert(launches == before);
}
'''
    _run(tmp_path, "vision_tanh_high", COMMON + modes + precision
         + program.replace("__SITE_VALUE__", site).replace("__ENCODER_SITE_VALUE__", encoder_site)
         .replace("LAYOUT", layout).replace("DISPATCH", dispatch))
