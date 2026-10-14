#pragma once

// Layer metadata has already passed the model's full shape/device validation.
// Weight-independent fusions admit uniform W8 with its original FP16 scales.
// Private cold capability prevents silently changing existing W8 owners.
template <class Layer>
bool hyvla_linear_precision(const Layer& w, bool allow_w8, bool allow_nvfp4 = false) {
    const auto dtype = w.q_w.scalar_type();
    if (dtype != at::kHalf && !(allow_w8 && dtype == at::kChar) &&
        !(allow_nvfp4 && dtype == at::kByte)) return false;
    const std::array<std::pair<const at::Tensor*, const at::Tensor*>, 7> pairs{{
        {&w.q_w, &w.q_ws}, {&w.k_w, &w.k_ws}, {&w.v_w, &w.v_ws},
        {&w.o_w, &w.o_ws}, {&w.gate_w, &w.gate_ws}, {&w.up_w, &w.up_ws},
        {&w.down_w, &w.down_ws}}};
    for (const auto& pair : pairs) {
        if (!pair.first->defined() || pair.first->scalar_type() != dtype) return false;
        if (dtype == at::kHalf) {
            if (pair.second->defined()) return false;
        } else if (dtype == at::kByte) {
            if (!pair.second->defined() || pair.second->scalar_type() != at::kByte ||
                !pair.second->is_contiguous() || pair.second->dim() != 2) return false;
        } else if (!pair.second->defined() || pair.second->scalar_type() != at::kHalf ||
                   !pair.second->is_contiguous() || pair.second->dim() != 1) return false;
    }
    return true;
}
