// rpu_concat.cpp - Concat operation for device backend (memcpy version)
#include "rpu_ops.h"
#include <string.h>  // for memcpy
#include <vector>
#include <limits>

using namespace at;

// ============================================================================
// rpu_cat - 原生 memcpy 实现
// 直接在 DDR 上进行内存拷贝，避免 CPU allocator 分配和设备转换开销
// ============================================================================
at::Tensor rpu_cat(const c10::IListRef<at::Tensor>& tensors, int64_t dim) {
    TORCH_CHECK(!tensors.empty(), "cat: expected a non-empty list of tensors");

    // 收集输入 tensor 并确保 contiguous
    std::vector<at::Tensor> inputs;
    inputs.reserve(tensors.size());
    c10::Device target_device = c10::kCPU;

    for (const auto& t : tensors) {
        if (t.numel() == 0) continue;  // 跳过空 tensor
        if (target_device == c10::kCPU &&
            t.device().type() == c10::DeviceType::PrivateUse1) {
            target_device = t.device();
        }
        inputs.push_back(t.contiguous());
    }

    if (inputs.empty()) {
        // 所有输入都是空的，返回空 tensor
        return at::empty({0}, tensors.front().options());
    }

    const auto& first = inputs[0];
    int64_t ndim = first.dim();

    // 处理标量 tensor (0 维)
    if (ndim == 0) {
        TORCH_CHECK(false, "cat: zero-dimensional tensors cannot be concatenated");
    }

    // 处理负数 dim
    if (dim < 0) {
        dim = dim + ndim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim,
                "cat: dimension ", dim, " out of range for tensor with ",
                ndim, " dimensions");

    // 计算输出形状
    std::vector<int64_t> output_shape = first.sizes().vec();
    int64_t cat_dim_size = first.size(dim);

    for (size_t i = 1; i < inputs.size(); i++) {
        const auto& t = inputs[i];
        TORCH_CHECK(t.dim() == ndim,
                    "cat: all tensors must have the same number of dimensions, "
                    "got ", ndim, " and ", t.dim());
        TORCH_CHECK(t.scalar_type() == first.scalar_type(),
                    "cat: all tensors must have the same dtype");

        // 检查非 cat 维度是否匹配
        for (int64_t d = 0; d < ndim; d++) {
            if (d != dim) {
                TORCH_CHECK(t.size(d) == first.size(d),
                            "cat: sizes must match except in dimension ", dim,
                            ", got ", first.size(d), " and ", t.size(d),
                            " in dimension ", d);
            }
        }
        cat_dim_size += t.size(dim);
    }
    output_shape[dim] = cat_dim_size;

    // 在 device 上分配输出
    auto output = at::empty(output_shape, first.options().device(target_device));

    // 计算 memcpy 参数
    // outer_size: dim 之前所有维度的乘积
    // inner_size: dim 之后所有维度的乘积
    int64_t outer_size = 1;
    for (int64_t d = 0; d < dim; d++) {
        outer_size *= output_shape[d];
    }

    int64_t inner_size = 1;
    for (int64_t d = dim + 1; d < ndim; d++) {
        inner_size *= output_shape[d];
    }

    size_t elem_size = first.element_size();
    // 输出在 cat 维度上每个元素占用的字节数
    int64_t out_dim_stride = inner_size * elem_size;
    // 输出在 outer 维度上的步长 (bytes)
    int64_t out_outer_stride = output_shape[dim] * out_dim_stride;

    char* out_ptr = static_cast<char*>(output.data_ptr());
    int64_t cat_offset = 0;  // 当前在 cat 维度上的偏移

    // Host memcpy crosses the CPU<->device coherency boundary.  The regular
    // rpu_ddr_flush knob is intentionally disabled for intra-device traffic, so
    // it cannot guard these reads/writes.  Flush only the live tensor ranges.
    // Batch has no SDK owner/bounds lookup. Prove every rounded read range
    // before deferring any acquire, including disjointness from the fresh output.
    std::vector<void*> acquire_ptrs;
    std::vector<size_t> acquire_sizes;
    const uintptr_t max_addr = std::numeric_limits<uintptr_t>::max();
    const uintptr_t out_begin = reinterpret_cast<uintptr_t>(out_ptr);
    const size_t out_bytes = output.nbytes();
    bool batch_acquire = g_rpu_ddr_flush_force_enabled && out_begin != 0 &&
        out_bytes != 0 && out_bytes <= max_addr - out_begin &&
        out_begin + out_bytes <= max_addr - 63;
    const uintptr_t out_lo = out_begin & ~uintptr_t(63);
    const uintptr_t out_hi = batch_acquire
        ? (out_begin + out_bytes + 63) & ~uintptr_t(63) : 0;
    for (const auto& inp : inputs) {
        if (!batch_acquire) break;
        void* ptr = inp.data_ptr();
        const size_t bytes = inp.nbytes();
        const uintptr_t begin = reinterpret_cast<uintptr_t>(ptr);
        if (begin == 0 || bytes == 0 || bytes > max_addr - begin ||
            begin + bytes > max_addr - 63) {
            batch_acquire = false;
            break;
        }
        const uintptr_t lo = begin & ~uintptr_t(63);
        const uintptr_t hi = (begin + bytes + 63) & ~uintptr_t(63);
        if (lo < out_hi && out_lo < hi) {
            batch_acquire = false;
            break;
        }
        if (inp.device().type() != c10::DeviceType::PrivateUse1) continue;
        auto* owner = rhino_lkn::RpuGetHostddr(ptr);
        if (!owner) {
            batch_acquire = false;
            break;
        }
        const uintptr_t base = reinterpret_cast<uintptr_t>(owner->get_cpu_ptr());
        const size_t extent = owner->get_memory_size();
        if (base == 0 || extent == 0 || extent > max_addr - base ||
            lo < base || hi - base > extent) {
            batch_acquire = false;
            break;
        }
        acquire_ptrs.push_back(ptr);
        acquire_sizes.push_back(bytes);
    }
    if (batch_acquire && acquire_ptrs.size() > 1) {
        rhino_lkn::RpuDdrFlushBatch(
            acquire_ptrs.data(), acquire_sizes.data(), acquire_ptrs.size());
#if defined(__aarch64__)
        __asm__ volatile("dsb sy" ::: "memory");
#endif
    } else {
        for (const auto& inp : inputs) {
            if (inp.device().type() == c10::DeviceType::PrivateUse1) {
                rpu_ddr_flush_force_sized(inp.data_ptr(), inp.nbytes());
            }
        }
    }

    for (const auto& inp : inputs) {
        const char* inp_ptr = static_cast<const char*>(inp.data_ptr());
        int64_t inp_cat_size = inp.size(dim);
        // 输入在 outer 维度上的步长 (bytes)
        int64_t inp_outer_stride = inp_cat_size * out_dim_stride;
        // 每次拷贝的字节数
        int64_t copy_size = inp_cat_size * out_dim_stride;

        for (int64_t o = 0; o < outer_size; o++) {
            char* dst = out_ptr + o * out_outer_stride + cat_offset * out_dim_stride;
            const char* src = inp_ptr + o * inp_outer_stride;
            memcpy(dst, src, copy_size);
        }
        cat_offset += inp_cat_size;
    }

    if (target_device.type() == c10::DeviceType::PrivateUse1) {
        // Publish the host-written output before the next device DMA reads it.
        rpu_ddr_flush_force_sized(out_ptr, output.nbytes());
    }

    return output;
}

// ============================================================================
// rpu_cat_out - 带输出参数的版本 (用于 inplace 场景)
// ============================================================================
at::Tensor& rpu_cat_out(const c10::IListRef<at::Tensor>& tensors,
                         int64_t dim,
                         at::Tensor& out) {
    auto result = rpu_cat(tensors, dim);
    out.resize_(result.sizes());
    out.copy_(result);
    return out;
}
