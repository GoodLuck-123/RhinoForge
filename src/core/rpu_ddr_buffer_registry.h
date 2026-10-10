// rpu_ddr_buffer_registry.h — shape-keyed at::Tensor registry for DMA-baked
// DDR buffers in FusedModelBase. See spec §3 and §9.
//
// Why this exists: DMA src/dst addresses are baked into kd_buf at
// Queue_t::build_batch and have no sync_mutable_params equivalent. Any DMA
// inside a cached GraphSlot must reference a pointer that is stable for the
// slot's lifetime. The registry holds one (a, b, out) tensor triple per
// distinct ShapeKey on the FusedModelBase::Impl, indexed by
// (batch, seq, hidden, dtype, device). Same shape → same data_ptr.
// Multi-key replay (A→B→A) returns the original A pointers while A's graph
// remains live. Opted-in owners retire entries after all referencing graphs
// retire; an evicted A gets fresh buffers for its next BUILD.
#pragma once

#include <unordered_map>
#include <algorithm>
#include <memory>
#include <ATen/ATen.h>
#include <c10/core/Device.h>

#include "rpu_bounded_shape_map.h"

struct ShapeKey {
    int64_t       batch;
    int64_t       seq;
    int64_t       hidden;
    at::ScalarType dtype;
    at::DeviceType device;

    bool operator==(const ShapeKey& o) const {
        return batch == o.batch && seq == o.seq && hidden == o.hidden
            && dtype == o.dtype && device == o.device;
    }

    struct Hash {
        size_t operator()(const ShapeKey& k) const {
            // Simple bitwise mix — collisions impact only registry probe cost.
            size_t h = std::hash<int64_t>()(k.batch);
            h ^= std::hash<int64_t>()(k.seq)    + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= std::hash<int64_t>()(k.hidden) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(static_cast<int>(k.dtype))  + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(static_cast<int>(k.device)) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            return h;
        }
    };
};

class DdrBufferRegistry {
public:
    struct Entry {
        at::Tensor        a;
        at::Tensor        b;
        at::Tensor        out;
        at::TensorOptions opts;  // captured at first insert
        bool graph_scoped = false;
        std::vector<std::weak_ptr<const uint8_t>> graph_lifetimes;
    };

    // Returns the entry for `key`, allocating missing fields lazily.
    // Asserts caller-supplied opts match the stored entry's opts on reuse.
    //
    // Layer-0 input no longer needs a registry-owned staging slot — it uses
    // a mutable DMA referencing the caller's per-forward `hidden_states`
    // directly through add_dma_kernel_mutable.
    const Entry& get(const ShapeKey& key, at::TensorOptions opts,
                     bool need_ab, bool need_out,
                     bool graph_scoped = false,
                     std::weak_ptr<const uint8_t> graph_lifetime = {});

    size_t size() const noexcept { return store_.size(); }

    // Bound on distinct shapes per handle. Entries cannot be evicted: output_tensor_
    // and the DDR ping-pong bases are baked into DMA nodes at BUILD. Releasing an
    // entry while a cached graph exists could let REPLAY access reissued storage.
    // The bound limits growth from requests that continually introduce new shapes.
    static constexpr size_t kMaxShapes = 64;

private:
    rpu::BoundedShapeMap<ShapeKey, Entry, ShapeKey::Hash> store_{
        kMaxShapes, rpu::ShapeMapPolicy::Reject, "DdrBufferRegistry"};
};

inline const DdrBufferRegistry::Entry&
DdrBufferRegistry::get(const ShapeKey& key, at::TensorOptions opts,
                        bool need_ab, bool need_out, bool graph_scoped,
                        std::weak_ptr<const uint8_t> graph_lifetime) {
    if (graph_scoped) {
        // Graph destruction/invalidation retires its token only after execution
        // and recorded resources are finished. Keep every still-live graph's
        // buffers, including when several signatures share one ShapeKey.
        store_.erase_if([](Entry& e) {
            auto& owners = e.graph_lifetimes;
            owners.erase(std::remove_if(owners.begin(), owners.end(),
                [](const auto& owner) { return owner.expired(); }), owners.end());
            return e.graph_scoped && owners.empty();
        });
    }
    bool inserted = false;
    Entry& e = store_.touch(key, &inserted);
    if (inserted) {
        e.opts = opts;
        e.graph_scoped = graph_scoped;
    } else {
        TORCH_CHECK(e.graph_scoped == graph_scoped,
                    "DdrBufferRegistry: cannot change a live entry's lifetime policy");
        TORCH_CHECK(e.opts.dtype()  == opts.dtype()  &&
                    e.opts.device() == opts.device() &&
                    e.opts.layout() == opts.layout(),
                    "DdrBufferRegistry: TensorOptions mismatch on reuse for "
                    "(batch=", key.batch, " seq=", key.seq,
                    " hidden=", key.hidden, ")");
    }
    if (graph_scoped && !graph_lifetime.expired()) {
        const bool recorded = std::any_of(
            e.graph_lifetimes.begin(), e.graph_lifetimes.end(),
            [&](const auto& owner) {
                return !owner.owner_before(graph_lifetime) &&
                       !graph_lifetime.owner_before(owner);
            });
        if (!recorded) e.graph_lifetimes.push_back(std::move(graph_lifetime));
    }
    std::vector<int64_t> shape_vec{key.batch, key.seq, key.hidden};

    if (need_ab) {
        if (!e.a.defined()) e.a = at::empty(shape_vec, e.opts);
        if (!e.b.defined()) e.b = at::empty(shape_vec, e.opts);
    }
    if (need_out && !e.out.defined()) {
        e.out = at::empty(shape_vec, e.opts);
    }
    return e;
}
