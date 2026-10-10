#pragma once

// Private HyVLA expansion ABI. No DDR operands or scratch allocation.
// Caller supplies two read-only row masks and two disjoint activation owners.
// Result may replace text; the inactive act temporary is dead at every caller.
inline void launch_hyvla_row_merge_spm(
    uint32_t text, uint32_t act, uint32_t out,
    uint32_t text_mask, uint32_t act_mask,
    int64_t rows, int64_t cols, int cores) {
    TORCH_CHECK((rows >= 128 && rows <= 240 && rows % 16 == 0) &&
                    (cols == 128 || cols == 256 || cols == 2048) &&
                    cores == 8 && SPM_ALLOC.is_initialized(),
                "HyVLA row merge requires S128..240 aligned16/C128,256,2048/TP8 and live SPM");
    const uint64_t bytes = uint64_t(rows) * cols * 2, mask_bytes = rows * 2;
    const uint64_t base = SPM_ALLOC.addr(0, 0);
    const auto checked = [base](uint32_t address, uint64_t size) {
        if (address % 256 || address < base ||
            uint64_t(address) - base + size > SpmAllocator::SPM_USABLE) return false;
        for (int core = 0; core < 8; ++core) {
            const uint64_t end = uint64_t(SPM_ALLOC.addr(core, 0)) +
                                 (uint64_t(address) - base) + size;
            if (end > uint64_t(UINT32_MAX) + 1) return false;
        }
        return true;
    };
    const auto disjoint = [](uint32_t a, uint64_t as, uint32_t b, uint64_t bs) {
        return uint64_t(a) + as <= b || uint64_t(b) + bs <= a;
    };
    TORCH_CHECK(checked(text, bytes) && checked(act, bytes) && checked(out, bytes) &&
                    checked(text_mask, mask_bytes) && checked(act_mask, mask_bytes),
                "HyVLA row merge requires complete aligned absolute SPM operands");
    TORCH_CHECK(disjoint(text, bytes, act, bytes) &&
                    (out == text || disjoint(out, bytes, text, bytes)) &&
                    disjoint(out, bytes, act, bytes) &&
                    disjoint(text_mask, mask_bytes, act_mask, mask_bytes),
                "HyVLA row merge rejects partial aliases and overlapping inputs");
    for (auto data : {text, act, out}) {
        TORCH_CHECK(disjoint(data, bytes, text_mask, mask_bytes) &&
                        disjoint(data, bytes, act_mask, mask_bytes),
                    "HyVLA row merge rejects activation/mask overlap");
    }
    const uint16_t row_tile = cols == 2048 ? 16 : 32;
    const uint16_t blocks = (rows + row_tile - 1) / row_tile;
    const uint16_t normal = row_tile / 16;
    const uint16_t last = (rows - (blocks - 1) * row_tile) / 16;
    const uint16_t vectors = cols / 16;
    auto* kernel = RpuKernelGraph::active().get_kernel_reset("hyvla_mot_row_merge_s240");
    TORCH_CHECK(kernel, "HyVLA row merge expansion kernel unavailable");
    kernel->reset_regs();
    kernel->set_regs(0, normal);
    kernel->set_regs(1, last);
    for (auto entry : {std::pair<uint32_t, uint32_t>{4, text_mask},
                       {6, text}, {8, out}, {14, act_mask}, {16, act}}) {
        kernel->set_regs(entry.first, uint16_t(entry.second));
        kernel->set_regs(entry.first + 1, uint16_t(entry.second >> 16));
    }
    kernel->set_regs(10, uint16_t(row_tile * 2));
    kernel->set_regs(12, vectors);
    kernel->set_regs(13, normal);
    kernel->set_regs(18, uint16_t(2 * normal));
    kernel->set_regs(19, uint16_t(2 * normal + normal * vectors));
    auto* queue = GET_QUEUE(cores);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel, {blocks, uint16_t(1), uint16_t(1)},
                        {0, 1, 2, 3, 4, 5, 6, 7});
}
