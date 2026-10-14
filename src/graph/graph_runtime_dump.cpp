// Read-only public Graph structure; no addresses, register words or payloads.
#include "graph/graph_runtime.h"
#include <iomanip>
#include <sstream>
#include <cstdio>
#include <cctype>

std::string RpuKernelGraph::dump_resources() const {
    std::ostringstream os;
    const size_t n_persist  = host_callback_persistent_owners_.size();
    const size_t n_zerocopy = host_callback_zero_copy_holders_.size();
    const size_t n_t3xform  = tier3_oneshot_transient_owners_.size();
    const char* st = "Unknown";
    switch (state_) {
        case State::PASSTHROUGH: st = "PASSTHROUGH"; break;
        case State::RECORDING:   st = "RECORDING";   break;
        case State::REPLAYING:   st = "REPLAYING";   break;
        case State::BUILT:       st = "BUILT";       break;
    }
    os << "GraphResources state=" << st
       << " total=" << (n_persist + n_zerocopy + n_t3xform)
       << " StableOutput=" << n_persist
       << " ZeroCopyStorageHolder=" << n_zerocopy
       << " TransientTier3Buffer=" << n_t3xform << "\n";
    auto render = [&](size_t idx, const GraphOwnedResource& r) {
        os << "  [" << std::setw(3) << std::setfill('0') << idx
           << std::setfill(' ') << "] " << resource_kind_name(r.kind)
           << " owner=" << r.owner_node_id
           << " nbytes=" << r.nbytes
           << " state=" << lifetime_state_name(r.state);
        if (r.pool_slot_id != kInvalidPoolSlotId) {
            os << " pool_slot=" << r.pool_slot_id;
        }
        if (r.allocator_block_id != 0) {
            os << " block_id=" << r.allocator_block_id;
        }
        os << "\n";
    };
    size_t idx = 0;
    for (const auto& r : host_callback_persistent_owners_)  render(idx++, r);
    for (const auto& r : host_callback_zero_copy_holders_)  render(idx++, r);
    for (const auto& r : tier3_oneshot_transient_owners_)   render(idx++, r);
    return os.str();
}

namespace {

const char* graph_state_name(RpuKernelGraph::State state) {
    switch (state) {
    case RpuKernelGraph::State::PASSTHROUGH: return "PASSTHROUGH";
    case RpuKernelGraph::State::RECORDING: return "RECORDING";
    case RpuKernelGraph::State::BUILT: return "BUILT";
    case RpuKernelGraph::State::REPLAYING: return "REPLAYING";
    }
    return "UNKNOWN";
}

const char* graph_node_kind_name(GraphNodeKind kind) {
    switch (kind) {
    case GraphNodeKind::Kernel: return "Kernel";
    case GraphNodeKind::Memcpy: return "Memcpy";
    case GraphNodeKind::Memset: return "Memset";
    case GraphNodeKind::ChildGraph: return "ChildGraph";
    case GraphNodeKind::Branch: return "Branch";
    case GraphNodeKind::HostCallback: return "HostCallback";
    case GraphNodeKind::Tier3Oneshot: return "Tier3Oneshot";
    case GraphNodeKind::Dma: return "Dma";
    case GraphNodeKind::Barrier: return "Barrier";
    }
    return "Unknown";
}

const char* copy_kind_name(CopyKind kind) {
    switch (kind) {
    case CopyKind::HOST_MEMCPY: return "host";
    case CopyKind::DDR_TO_DDR: return "ddr_to_ddr";
    case CopyKind::DDR_TO_SPM: return "ddr_to_spm";
    case CopyKind::SPM_TO_DDR: return "spm_to_ddr";
    case CopyKind::SPM_TO_SPM: return "spm_to_spm";
    }
    return "unknown";
}

const char* dma_variant_name(DmaNodeData::Variant variant) {
    switch (variant) {
    case DmaNodeData::Variant::Fixed: return "fixed";
    case DmaNodeData::Variant::MutableSrc: return "mutable_src";
    case DmaNodeData::Variant::MutableDst: return "mutable_dst";
    }
    return "unknown";
}

template <typename T>
void append_values(std::ostringstream& out, const std::vector<T>& values) {
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out << ",";
        out << static_cast<uint64_t>(values[i]);
    }
    out << "]";
}

std::string graph_kernel_name(const KernelNodeData& kernel) {
    if (kernel.kernel_id.has_value()) {
        const size_t id = static_cast<size_t>(*kernel.kernel_id);
        if (id < static_cast<size_t>(KernelId::_COUNT)) {
            return KERNEL_ID_NAMES[id];
        }
        return "kernel_id#" + std::to_string(id);
    }
    return kernel.kernel_name.empty() ? "<unnamed>" : kernel.kernel_name;
}

std::string safe_node_summary(
        size_t node_id, const GraphNode& node, int64_t segment_id) {
    std::ostringstream out;
    out << "node[" << node_id << "] kind=" << graph_node_kind_name(node.kind)
        << " segment=" << segment_id;
    switch (node.kind) {
    case GraphNodeKind::Kernel: {
        const auto& kernel = node.as_kernel();
        out << " kernel=" << graph_kernel_name(kernel) << " cores=";
        append_values(out, kernel.core_ids);
        out << " grid=";
        append_values(out, kernel.grid_dims);
        break;
    }
    case GraphNodeKind::Memcpy: {
        const auto& copy = node.as_memcpy();
        out << " copy=" << copy_kind_name(copy.kind)
            << " bytes=" << copy.bytes;
        break;
    }
    case GraphNodeKind::Memset: {
        const auto& fill = std::get<MemsetNodeData>(node.data);
        out << " bytes=" << fill.bytes
            << " value=" << static_cast<uint32_t>(fill.value);
        break;
    }
    case GraphNodeKind::Branch:
        out << " branch_key=0x" << std::hex << node.as_branch().branch_key;
        break;
    case GraphNodeKind::HostCallback: {
        const auto& callback = node.as_host_callback();
        out << " tier=" << static_cast<uint32_t>(callback.tier)
            << " outputs=" << callback.output_pool.size();
        break;
    }
    case GraphNodeKind::Tier3Oneshot: {
        const auto& oneshot = node.as_tier3_oneshot();
        out << " input_deps=" << oneshot.input_dep_node_ids.size()
            << " output_deps=" << oneshot.output_dep_node_ids.size();
        break;
    }
    case GraphNodeKind::Dma: {
        const auto& dma = node.as_dma();
        out << " variant=" << dma_variant_name(dma.variant)
            << " bytes=" << dma.bytes
            << " channel=" << static_cast<uint32_t>(dma.channel);
        break;
    }
    case GraphNodeKind::Barrier: {
        const auto& barrier = node.as_barrier_node();
        out << " self_stream=" << static_cast<uint32_t>(barrier.self_stream)
            << " target_stream="
            << static_cast<uint32_t>(barrier.target_stream);
        break;
    }
    case GraphNodeKind::ChildGraph:
        break;
    }
    return out.str();
}

std::vector<int64_t> graph_segment_ids(
        size_t node_count, const std::vector<Segment>& segments) {
    std::vector<int64_t> result(node_count, -1);
    for (const auto& segment : segments) {
        for (size_t i = segment.start_idx; i < segment.end_idx; ++i) {
            result[i] = static_cast<int64_t>(segment.segment_id);
        }
    }
    return result;
}

void append_segment_summary(std::ostringstream& out, const Segment& segment) {
    out << "segment[" << segment.segment_id << "] nodes=["
        << segment.start_idx << "," << segment.end_idx << ") cores=";
    append_values(out, segment.core_ids);
    out << " broadcast=" << (segment.queue_state.broadcast_mode ? 1 : 0)
        << " flush_icache=" << (segment.queue_state.flush_icache ? 1 : 0)
        << " replay_count=" << segment.replay_count;
}

const char* tier_name(HostCallbackTier t) {
    switch (t) {
        case HostCallbackTier::Tier1:  return "T1";
        case HostCallbackTier::Tier2:  return "T2";
        case HostCallbackTier::Tier3:  return "T3";
        case HostCallbackTier::Reject: return "Reject";
    }
    return "?";
}

std::string host_callback_op_name(const HostCallbackNodeData& d) {
    if (d.op_handle.has_value()) {
        return d.op_handle->schema().operator_name().name;
    }
    return "<no_op>";
}

std::string tier3_oneshot_op_name(const Tier3OneshotNodeData& d) {
    if (d.op_handle.has_value()) {
        return d.op_handle->schema().operator_name().name;
    }
    return "<no_op>";
}

// 一行节点摘要(用于 dump_replay_plan 文本)。

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x",
                                  static_cast<unsigned char>(c));
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

std::string graph_reason_without_addresses(const std::string& reason) {
    std::string out;
    for (size_t i = 0; i < reason.size();) {
        if (i + 2 < reason.size() && reason[i] == '0' &&
                (reason[i + 1] == 'x' || reason[i + 1] == 'X') &&
                std::isxdigit(static_cast<unsigned char>(reason[i + 2]))) {
            i += 2;
            while (i < reason.size() &&
                    std::isxdigit(static_cast<unsigned char>(reason[i]))) ++i;
            out += "<address>";
        } else {
            out += reason[i++];
        }
    }
    return out;
}

std::string render_node_jsonl(size_t idx,
                              const GraphNode& n,
                              int64_t seg_id) {
    std::ostringstream o;
    o << "{\"node_id\":" << idx
      << ",\"kind\":\"" << graph_node_kind_name(n.kind)
      << "\",\"segment_id\":" << seg_id;
    switch (n.kind) {
    case GraphNodeKind::Kernel: {
        const auto& k = n.as_kernel();
        o << ",\"kernel\":\"" << json_escape(graph_kernel_name(k)) << "\""
          << ",\"cores\":";
        append_values(o, k.core_ids);
        o << ",\"grid\":";
        append_values(o, k.grid_dims);
        o << ",\"broadcast\":"
          << (k.queue_state.broadcast_mode ? "true" : "false");
        break;
    }
    case GraphNodeKind::Memcpy: {
        const auto& m = std::get<MemcpyNodeData>(n.data);
        o << ",\"copy_kind\":\"" << (m.kind == CopyKind::HOST_MEMCPY ? "host_memcpy" : copy_kind_name(m.kind)) << "\""
          << ",\"bytes\":" << m.bytes;
        break;
    }
    case GraphNodeKind::Memset: {
        const auto& m = std::get<MemsetNodeData>(n.data);
        o << ",\"bytes\":" << m.bytes
          << ",\"value\":" << static_cast<int>(m.value);
        break;
    }
    case GraphNodeKind::ChildGraph: {
        o << ",\"child\":true";
        break;
    }
    case GraphNodeKind::Branch: {
        const auto& b = n.as_branch();
        o << ",\"branch_key\":" << b.branch_key;
        break;
    }
    case GraphNodeKind::HostCallback: {
        const auto& h = n.as_host_callback();
        o << ",\"op\":\"" << json_escape(host_callback_op_name(h)) << "\""
          << ",\"tier\":\"" << tier_name(h.tier) << "\""
          << ",\"output_count\":" << h.output_pool.size()
          << ",\"output_bytes\":[";
        for (size_t i = 0; i < h.output_pool_bytes.size(); ++i) {
            if (i) o << ",";
            o << h.output_pool_bytes[i];
        }
        o << "]";
        break;
    }
    case GraphNodeKind::Tier3Oneshot: {
        const auto& t = n.as_tier3_oneshot();
        o << ",\"op\":\"" << json_escape(tier3_oneshot_op_name(t)) << "\""
          << ",\"input_dep_node_ids\":[";
        for (size_t i = 0; i < t.input_dep_node_ids.size(); ++i) {
            if (i) o << ",";
            o << t.input_dep_node_ids[i];
        }
        o << "],\"output_dep_node_ids\":[";
        for (size_t i = 0; i < t.output_dep_node_ids.size(); ++i) {
            if (i) o << ",";
            o << t.output_dep_node_ids[i];
        }
        o << "],\"transient_resource_ids\":[";
        for (size_t i = 0; i < t.transient_resource_ids.size(); ++i) {
            if (i) o << ",";
            o << t.transient_resource_ids[i];
        }
        o << "]";
        break;
    }
    case GraphNodeKind::Dma: {
        const auto& d = n.as_dma();
        const char* vname = (d.variant == DmaNodeData::Variant::Fixed)      ? "fixed"
                          : (d.variant == DmaNodeData::Variant::MutableSrc) ? "msrc"
                                                                            : "mdst";
        o << ",\"variant\":\""  << vname << "\""
          << ",\"bytes\":"      << d.bytes
          << ",\"channel\":"    << static_cast<int>(d.channel);
        if (d.semantic_endpoint_id != 0) {
            o << ",\"semantic_endpoint_id\":"
              << d.semantic_endpoint_id;
            o << ",\"semantic_canonical_member_emission\":"
              << (d.semantic_canonical_member_emission ? "true" : "false");
            o << ",\"semantic_spm_peer_bound\":"
              << (d.semantic_spm_peer_id != 0 ? "true" : "false");
        }
        break;
    }
    case GraphNodeKind::Barrier: {
        const auto& b = n.as_barrier_node();
        o << ",\"self_stream\":"   << static_cast<int>(b.self_stream)
          << ",\"target_stream\":" << static_cast<int>(b.target_stream);
        break;
    }
    }
    o << "}";
    return o.str();
}

}  // namespace

std::string RpuKernelGraph::dump_replay_plan() const {
    std::ostringstream out;
    out << "GraphReplayPlan state=" << graph_state_name(state_)
        << " replayable=" << (replayable_ ? "true" : "false")
        << " graph_size=" << nodes_.size()
        << " segments=" << segments_.size()
        << " boundary_flush_count=" << boundary_flush_ptrs_.size();
    if (has_built_signature_) out << " has_built_sig=true";
    if (!non_replayable_reason_.empty()) {
        out << " non_replayable_reason=\""
            << json_escape(graph_reason_without_addresses(non_replayable_reason_)) << "\"";
    }
    out << "\n";
    for (const auto& segment : segments_) {
        out << "  ";
        append_segment_summary(out, segment);
        out << "\n";
    }
    const auto segment_ids = graph_segment_ids(nodes_.size(), segments_);
    for (size_t i = 0; i < nodes_.size(); ++i) {
        out << "  " << safe_node_summary(i, nodes_[i], segment_ids[i]) << "\n";
    }
    return out.str();
}

std::string RpuKernelGraph::dump_tree() const {
    std::ostringstream out;
    out << "GraphTree state=" << graph_state_name(state_)
        << " replayable=" << (replayable_ ? "true" : "false")
        << " graph_size=" << nodes_.size()
        << " segments=" << segments_.size()
        << " boundary_flush_count=" << boundary_flush_ptrs_.size() << "\n";
    const auto segment_ids = graph_segment_ids(nodes_.size(), segments_);
    for (const auto& segment : segments_) {
        out << "  ";
        append_segment_summary(out, segment);
        out << "\n";
        for (size_t i = segment.start_idx; i < segment.end_idx; ++i) {
            out << "    " << safe_node_summary(i, nodes_[i], segment_ids[i])
                << "\n";
        }
    }
    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (segment_ids[i] < 0) {
            out << "  inter_segment "
                << safe_node_summary(i, nodes_[i], -1) << "\n";
        }
    }
    return out.str();
}

std::string RpuKernelGraph::dump_nodes_jsonl() const {
    std::ostringstream out;
    const auto ids = graph_segment_ids(nodes_.size(), segments_);
    for (size_t i = 0; i < nodes_.size(); ++i) {
        out << render_node_jsonl(i, nodes_[i], ids[i]) << "\n";
    }
    return out.str();
}

std::string RpuKernelGraph::dump_kernel_args_jsonl() const {
    std::ostringstream o;
    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (nodes_[i].kind != GraphNodeKind::Kernel) {
            continue;
        }
        const auto& kn = nodes_[i].as_kernel();

        ::rhino_lkn::KernelLaunchFootprint footprint{};
        const uint32_t status =
            kernels_[kn.kernel_idx]->query_launch_footprint(&footprint);
        o << "{\"node_id\":" << i
          << ",\"kernel_idx\":" << kn.kernel_idx
          << ",\"kernel\":\"" << json_escape(graph_kernel_name(kn)) << "\""
          << ",\"raw_state\":\"unavailable_in_release\""
          << ",\"query_status\":" << status
          << ",\"command_reservation_bytes\":"
          << footprint.command_reservation_bytes
          << ",\"code_reservation_bytes\":"
          << footprint.code_reservation_bytes << "}\n";

    }
    return o.str();
}

std::string RpuKernelGraph::dump_segments_jsonl() const {
    std::ostringstream o;
    for (const auto& seg : segments_) {
        o << "{\"segment_id\":" << seg.segment_id
          << ",\"start_idx\":" << seg.start_idx
          << ",\"end_idx\":" << seg.end_idx
          << ",\"cores\":[";
        for (size_t i = 0; i < seg.core_ids.size(); ++i) {
            if (i) o << ",";
            o << static_cast<int>(seg.core_ids[i]);
        }
        o << "],\"replay_count\":" << seg.replay_count
          << ",\"signature_layer_hash\":" << seg.signature_layer_hash
          << ",\"node_indices\":[";
        for (size_t i = seg.start_idx; i < seg.end_idx; ++i) {
            if (i != seg.start_idx) o << ",";
            o << i;
        }
        o << "]}\n";
    }
    return o.str();
}
