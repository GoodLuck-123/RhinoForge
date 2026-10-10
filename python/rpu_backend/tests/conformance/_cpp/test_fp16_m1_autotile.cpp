// Dedicated conformance harness for the production M=1 FP16 auto-tile path.

#include "rpu_kernel_decls.h"
#include "rpu_spm_allocator.h"
#include "graph/graph_infra.h"
#include "graph/graph_runtime.h"

#include <ATen/ATen.h>
#include <c10/util/Half.h>
#include <c10/util/Optional.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <array>
#include <chrono>
#include <tuple>

namespace {

constexpr size_t kFp16Bytes = sizeof(c10::Half);

size_t align256(size_t value) {
    return (value + 255) & ~size_t(255);
}

void copy_input_to_spm(
    const at::Tensor& input,
    uint32_t input_off,
    int64_t k,
    int partition,
    int num_cores) {
    const int64_t local_k = (partition == 0) ? k / num_cores : k;
    auto* src = input.data_ptr<c10::Half>();
    rpu_ddr_flush_force(src);
    for (int core = 0; core < num_cores; ++core) {
        auto* dst = static_cast<c10::Half*>(SPM_ALLOC.cpu_ptr(core, input_off));
        for (int64_t row = 0; row < input.size(0); ++row) {
            const c10::Half* core_src = src + row * k +
                ((partition == 0) ? core * local_k : 0);
            std::memcpy(dst + row * local_k, core_src,
                        static_cast<size_t>(local_k) * kFp16Bytes);
        }
    }
}

void copy_bias_to_spm(
    const at::Tensor& bias,
    uint32_t bias_off,
    int64_t n,
    int partition,
    int num_cores) {
    const int64_t local_n = (partition == 1) ? n / num_cores : n;
    auto* src = bias.data_ptr<c10::Half>();
    rpu_ddr_flush_force(src);
    if (partition == 0) {
        // Match Vision Merger: clear all participating cores and load the
        // true bias on core 0 only, so summing partials counts it once.
        const uint32_t bias_addr = SPM_ALLOC.addr(0, bias_off);
        rpu_launch_fill_spm_kernel(bias_addr, n, c10::Half(0.0f), num_cores);
        rpu_launch_ddr_broadcast_spm_dma(src, n, bias_addr, /*num_cores=*/1);
        return;
    }
    for (int core = 0; core < num_cores; ++core) {
        auto* dst = static_cast<c10::Half*>(SPM_ALLOC.cpu_ptr(core, bias_off));
        std::memcpy(dst, src + core * local_n,
                    static_cast<size_t>(local_n) * kFp16Bytes);
    }
}

at::Tensor dump_output(
    uint32_t output_off,
    int64_t n,
    int partition,
    int num_cores,
    int64_t m = 1) {
    at::Tensor out = at::zeros(
        {m, n}, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
    auto* dst = out.data_ptr<float>();

    if (partition == 0) {
        for (int core = 0; core < num_cores; ++core) {
            const auto* src = static_cast<const c10::Half*>(
                SPM_ALLOC.cpu_ptr(core, output_off));
            for (int64_t col = 0; col < m * n; ++col) {
                dst[col] += static_cast<float>(src[col]);
            }
        }
        return out;
    }

    const int64_t local_n = n / num_cores;
    for (int core = 0; core < num_cores; ++core) {
        const auto* src = static_cast<const c10::Half*>(
            SPM_ALLOC.cpu_ptr(core, output_off));
        for (int64_t row = 0; row < m; ++row) {
            for (int64_t col = 0; col < local_n; ++col) {
                dst[row * n + core * local_n + col] =
                    static_cast<float>(src[row * local_n + col]);
            }
        }
    }
    return out;
}

}  // namespace

// Unlike the PASSTHROUGH projection probe, both arms own retained private queues.
// All weights and SPM addresses stay live until both caches are destroyed;
// changing input contents never changes either signature. Timings cover only
// synchronous begin/emit/end, not CPU staging or output reconstruction.
std::tuple<at::Tensor, at::Tensor, at::Tensor> rpu_test_m1_linear_graph_ab(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& scale,
    int64_t n, int64_t k, int64_t partition, int64_t num_cores,
    int64_t warmup, int64_t runs) {
    TORCH_CHECK(!RpuKernelGraph::has_active(), "M1 A/B requires no outer Graph");
    TORCH_CHECK(n > 0 && k > 0 && n <= 65535 && k <= 65535,
                "M1 A/B requires bounded positive N/K");
    TORCH_CHECK(partition == 0 || partition == 1, "M1 A/B invalid partition");
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8, "M1 A/B invalid TP");
    TORCH_CHECK(warmup >= 2 && warmup <= 32 && runs >= 2 && runs <= 64,
                "M1 A/B needs 2..32 warmups and 2..64 measured replays");
    TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1 &&
                    input.scalar_type() == at::kHalf && input.is_contiguous() &&
                    input.dim() == 2 && input.size(0) == 1 && input.size(1) == k,
                "M1 A/B input must be contiguous RPU fp16 [1,K]");
    TORCH_CHECK(weight.device().type() == c10::DeviceType::PrivateUse1 &&
                    weight.is_contiguous() && weight.dim() == 2 &&
                    weight.size(0) == n && weight.size(1) == k &&
                    (weight.scalar_type() == at::kHalf || weight.scalar_type() == at::kChar),
                "M1 A/B weight must be contiguous RPU fp16/int8 [N,K]");
    const bool int8_weight = weight.scalar_type() == at::kChar;
    const at::Tensor scales = scale.value_or(at::Tensor{});
    TORCH_CHECK(!int8_weight ||
                    (scales.defined() && scales.device() == input.device() &&
                     scales.scalar_type() == at::kHalf && scales.is_contiguous() &&
                     scales.dim() == 1 && scales.size(0) == n),
                "M1 A/B int8 weight requires RPU fp16 [N] scales");
    TORCH_CHECK(int8_weight || !scales.defined(), "M1 A/B fp16 has no scales");
    TORCH_CHECK((partition == 0 ? k : n) % num_cores == 0,
                "M1 A/B partition dimension must divide TP");
    const int64_t local_k = partition == 0 ? k / num_cores : k;
    const int64_t local_n = partition == 1 ? n / num_cores : n;
    TORCH_CHECK(local_n % 16 == 0 && local_k % (int8_weight ? 32 : 16) == 0,
                "M1 A/B local dimensions violate kernel alignment");
    const int64_t rounds = 1 + warmup + runs;
    const auto cpu = at::TensorOptions().device(at::kCPU);
    auto outputs = at::empty({rounds, 2, n}, cpu.dtype(at::kFloat));
    auto times = at::empty({rounds, 2}, cpu.dtype(at::kDouble));
    // replay, recapture, kernels, segments, queue hits, queue misses,
    // hardware submits, build generation, cache size, cache invariant.
    auto evidence = at::empty({rounds, 2, 10}, cpu.dtype(at::kLong));
    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
    SPM_ALLOC.reset_temporary();
    RpuGraphCache cache(2);
    try {
        const uint32_t input_off = SPM_ALLOC.alloc_temporary(align256(local_k * kFp16Bytes));
        const uint32_t output_off = SPM_ALLOC.alloc_temporary(align256(local_n * kFp16Bytes));
        const uint32_t input_addr = SPM_ALLOC.addr(0, input_off);
        const uint32_t output_addr = SPM_ALLOC.addr(0, output_off);
        std::array<GraphSignature, 2> signatures;
        std::array<uint64_t, 2> generations{}, topologies{};
        for (int arm = 0; arm < 2; ++arm) {
            auto& sig = signatures[arm];
            sig.op_id = 0x350ab + arm;
            sig.op_id_str = arm ? "qwen35_m1_gemv" : "qwen35_m1_auto";
            sig.shapes = {1, n, k};
            sig.dyn_dims = {partition, num_cores};
            sig.dtypes = {static_cast<uint8_t>(weight.scalar_type())};
            cache.get_or_create(sig);
        }
        for (int64_t round = 0; round < rounds; ++round) {
            for (int order = 0; order < 2; ++order) {
                const int arm = (round % 2) ? 1 - order : order;  // AB, BA, ...
                copy_input_to_spm(input, input_off, k, partition, num_cores);
                if (round % 2) {
                    for (int core = 0; core < num_cores; ++core) {
                        auto* values = static_cast<c10::Half*>(SPM_ALLOC.cpu_ptr(core, input_off));
                        for (int64_t col = 0; col < local_k; ++col) values[col] = -values[col];
                    }
                }
                // Poison output before each invocation: a missing batch launch
                // cannot pass by reading the previous arm's correct result.
                for (int core = 0; core < num_cores; ++core) {
                    std::memset(SPM_ALLOC.cpu_ptr(core, output_off), 0xff, local_n * kFp16Bytes);
                }
                const auto& sig = signatures[arm];
                auto& graph = *cache.lookup(sig);
                const auto start = std::chrono::steady_clock::now();
                graph.begin(sig);
                try {
                    TORCH_CHECK(graph.state() == (round == 0 ? RpuKernelGraph::State::RECORDING
                                                             : RpuKernelGraph::State::REPLAYING),
                                "M1 A/B expected one BUILD then stable REPLAY");
                    rpu_launch_linear_spm_to_spm_acc16_kernel(
                        input_addr, weight, output_addr, 1, n, k,
                        partition, num_cores, /*bias_spm_addr=*/0, scales,
                        /*nvfp4_tensor_scale_spm_addr=*/0, /*nvfp4_layer_id=*/0,
                        /*force_acc32=*/false, /*prefer_gemv=*/arm == 1);
                    graph.end();
                } catch (...) {
                    if (RpuKernelGraph::is_on_active_stack(graph)) graph.abort();
                    throw;
                }
                const auto stop = std::chrono::steady_clock::now();
                times.data_ptr<double>()[round * 2 + arm] =
                    std::chrono::duration<double, std::milli>(stop - start).count();
                TORCH_CHECK(graph.state() == RpuKernelGraph::State::BUILT && graph.replayable(),
                            "M1 A/B did not retain a replayable Graph");
                cache.touch_entry(sig, round != 0);
                const auto& stats = graph.debug_stats();
                if (round == 0) {
                    generations[arm] = graph.build_generation();
                    topologies[arm] = graph.build_topology_hash();
                }
                TORCH_CHECK(graph.build_generation() == generations[arm] &&
                                graph.build_topology_hash() == topologies[arm] &&
                                topologies[arm] != 0,
                            "M1 A/B graph was recaptured");
                TORCH_CHECK(stats.kernel_count == 1 && stats.segment_count == 1 &&
                                stats.prepared_segment_miss_total == 0 &&
                                stats.prepared_segment_hit_total == static_cast<size_t>(round) &&
                                stats.hw_batch_submit_total == static_cast<size_t>(round + 1),
                            "M1 A/B private prepared queue was rebuilt or not submitted");
                TORCH_CHECK(cache.size() == 2 && cache.cache_invariant_ok(),
                            "M1 A/B GraphCache grew or lost its invariant");
                for (const auto& snapshot : cache.snapshot()) {
                    if (snapshot.signature == sig) {
                        TORCH_CHECK(snapshot.replay_count == static_cast<size_t>(round) &&
                                        snapshot.recapture_count == 0,
                                    "M1 A/B replay counters drifted");
                        auto* dst = evidence.data_ptr<int64_t>() + (round * 2 + arm) * 10;
                        const int64_t values[] = {
                            static_cast<int64_t>(snapshot.replay_count),
                            static_cast<int64_t>(snapshot.recapture_count),
                            static_cast<int64_t>(snapshot.kernel_count),
                            static_cast<int64_t>(snapshot.segment_count),
                            static_cast<int64_t>(snapshot.prepared_segment_hit_total),
                            static_cast<int64_t>(snapshot.prepared_segment_miss_total),
                            static_cast<int64_t>(snapshot.hw_batch_submit_total),
                            static_cast<int64_t>(graph.build_generation()),
                            static_cast<int64_t>(cache.size()), cache.cache_invariant_ok()};
                        std::memcpy(dst, values, sizeof(values));
                    }
                }
                // CPU readback happens only AFTER synchronous graph.end().
                const auto output = dump_output(output_off, n, partition, num_cores);
                std::memcpy(outputs.data_ptr<float>() + (round * 2 + arm) * n,
                            output.data_ptr<float>(), n * sizeof(float));
            }
        }
        cache.clear();  // retire both queues before releasing their SPM addresses
        SPM_ALLOC.reset_temporary();
    } catch (...) {
        cache.clear();
        SPM_ALLOC.reset_temporary();
        throw;
    }
    return {outputs, times, evidence};
}

at::Tensor rpu_test_fp16_m1_autotile_projection(
    const at::Tensor& input,
    const at::Tensor& weight_swizzled,
    const c10::optional<at::Tensor>& bias,
    int64_t n,
    int64_t k,
    int64_t partition,
    int64_t num_cores,
    bool prefer_gemv,
    bool acc32,
    bool capture_graph) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "fp16_m1_autotile: requires no outer Graph");
    TORCH_CHECK(
        input.device().type() == c10::DeviceType::PrivateUse1,
        "fp16_m1_autotile: input must be on RPU");
    TORCH_CHECK(
        weight_swizzled.device().type() == c10::DeviceType::PrivateUse1,
        "fp16_m1_autotile: weight must be on RPU");
    TORCH_CHECK(
        input.scalar_type() == at::kHalf &&
            weight_swizzled.scalar_type() == at::kHalf,
        "fp16_m1_autotile: input and weight must be fp16");

    TORCH_CHECK(
        input.is_contiguous() && weight_swizzled.is_contiguous(),
        "fp16_m1_autotile: input and weight must be contiguous");
    TORCH_CHECK(
        input.dim() == 2 && (input.size(0) == 1 || input.size(0) == 2) &&
            input.size(1) == k,
        "fp16_m1_autotile: input must have shape [1 or 2, K]");
    TORCH_CHECK(
        partition == 0 || partition == 1,
        "fp16_m1_autotile: partition must be 0 (row) or 1 (col)");
    TORCH_CHECK(
        num_cores >= 1 && num_cores <= 8,
        "fp16_m1_autotile: num_cores must be in [1, 8]");
    TORCH_CHECK(
        (partition == 0 ? k : n) % num_cores == 0,
        "fp16_m1_autotile: partitioned dimension must be divisible by num_cores");

    const bool has_bias = bias.has_value() && bias->defined();
    if (has_bias) {
        TORCH_CHECK(
            bias->device().type() == c10::DeviceType::PrivateUse1 &&
                bias->scalar_type() == at::kHalf && bias->is_contiguous() &&
                bias->numel() == n,
            "fp16_m1_autotile: bias must be a contiguous fp16 RPU tensor with N elements");
    }

    const int64_t m = input.size(0);
    const int64_t local_k = (partition == 0) ? k / num_cores : k;
    const int64_t local_n = (partition == 1) ? n / num_cores : n;

    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
    SPM_ALLOC.reset_temporary();
    const uint32_t input_off = SPM_ALLOC.alloc_temporary(
        align256(static_cast<size_t>(m * local_k) * kFp16Bytes));
    const uint32_t output_off = SPM_ALLOC.alloc_temporary(
        align256(static_cast<size_t>(m * local_n) * kFp16Bytes));
    uint32_t bias_off = 0;
    if (has_bias) {
        bias_off = SPM_ALLOC.alloc_temporary(
            align256(static_cast<size_t>(local_n) * kFp16Bytes));
    }

    copy_input_to_spm(
        input, input_off, k, static_cast<int>(partition),
        static_cast<int>(num_cores));
    if (has_bias) {
        copy_bias_to_spm(
            *bias, bias_off, n, static_cast<int>(partition),
            static_cast<int>(num_cores));
    }

    const uint32_t input_addr = SPM_ALLOC.addr(0, input_off);
    const uint32_t output_addr = SPM_ALLOC.addr(0, output_off);
    const uint32_t bias_addr = has_bias ? SPM_ALLOC.addr(0, bias_off) : 0;
    RpuKernelGraph graph;
    GraphSignature signature;
    signature.op_id_str = "fp16_row_bias";
    signature.shapes = {m, n, k, partition, num_cores};
    signature.flags = (acc32 ? 1 : 0) | (prefer_gemv ? 2 : 0);
    at::Tensor out, first;
    size_t nodes = 0, submits = 0;
    uint64_t generation = 0, topology = 0;
    try {
        for (int pass = 0; pass < (capture_graph ? 4 : 1); ++pass) {
            // Poison output before BUILD and every REPLAY: stale correct
            // output cannot satisfy the execution or byte-equality checks.
            for (int core = 0; core < num_cores; ++core) {
                std::memset(SPM_ALLOC.cpu_ptr(core, output_off), 0xff,
                            m * local_n * kFp16Bytes);
            }
            if (capture_graph) {
                graph.begin(signature);
                TORCH_CHECK(
                    graph.state() == (pass == 0 ? RpuKernelGraph::State::RECORDING
                                               : RpuKernelGraph::State::REPLAYING),
                    "Linear bias expected one BUILD then stable REPLAY");
            }
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                input_addr, weight_swizzled, output_addr, m, n, k,
                static_cast<int>(partition), static_cast<int>(num_cores), bias_addr,
                /*scale=*/{}, /*nvfp4_tensor_scale_spm_addr=*/0,
                /*nvfp4_layer_id=*/0, acc32, prefer_gemv);
            if (capture_graph) {
                graph.end();
                TORCH_CHECK(graph.state() == RpuKernelGraph::State::BUILT &&
                                graph.replayable(),
                            "Linear bias did not retain a replayable Graph");
                if (pass == 0) {
                    nodes = graph.graph_size();
                    generation = graph.build_generation();
                    topology = graph.build_topology_hash();
                }
                TORCH_CHECK(nodes > 0 && graph.graph_size() == nodes &&
                                topology != 0 && graph.build_topology_hash() == topology &&
                                graph.build_generation() == generation,
                            "Linear bias Graph changed or recaptured");
                const auto& stats = graph.debug_stats();
                TORCH_CHECK(stats.kernel_count > 0 &&
                                stats.hw_batch_submit_total > submits &&
                                stats.prepared_segment_miss_total == 0,
                            "Linear bias Graph did not submit its retained queue");
                submits = stats.hw_batch_submit_total;
            }
            out = dump_output(output_off, n, static_cast<int>(partition),
                              static_cast<int>(num_cores), m);
            if (pass == 0) first = out;
            else TORCH_CHECK(at::equal(first, out),
                             "Linear bias BUILD/REPLAY output differs");
        }
        if (capture_graph) {
            std::fprintf(stderr,
                         "LINEAR_BIAS_GRAPH M=%lld TP=%lld acc32=%d gemv=%d "
                         "builds=1 replays=3 nodes=%zu submits=%zu exact=1\n",
                         static_cast<long long>(m), static_cast<long long>(num_cores),
                         acc32, prefer_gemv, nodes, submits);
        }
    } catch (...) {
        if (RpuKernelGraph::is_on_active_stack(graph)) graph.abort();
        graph.invalidate();
        SPM_ALLOC.reset_temporary();
        throw;
    }
    // Retire the private queue before releasing the SPM storage it captured.
    graph.invalidate();
    SPM_ALLOC.reset_temporary();
    return out;
}
