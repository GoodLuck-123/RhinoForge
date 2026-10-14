// Adapted from optimization_moe commit 451d4a534; same test source serves
// the ATen-only host executable and actual Graph conformance registration.
#include <ATen/ATen.h>
#include <iostream>
#include "core/rpu_ddr_buffer_registry.h"
#ifndef RPU_DDR_REGISTRY_HOST_ONLY
#include <torch/library.h>
#include "graph/graph_runtime.h"
#endif

namespace {

int64_t test_ddr_registry_lifetime() {
    const auto opts = at::TensorOptions().dtype(at::kHalf).device(at::kCPU);
    auto key = [](int64_t n) { return ShapeKey{1, n, 16, at::kHalf, at::kCPU}; };
    int64_t checks = 0;
    auto check = [&](bool ok, const char* label) {
        TORCH_CHECK(ok, "DDR graph lifetime: ", label);
        ++checks;
    };

    // Preserve the default policy used by all non-opted-in fused models.
    DdrBufferRegistry legacy;
    for (int64_t n = 1; n <= 64; ++n) legacy.get(key(n), opts, true, true);
    bool rejected = false;
    try { legacy.get(key(65), opts, true, true); }
    catch (const c10::Error& e) {
        rejected = std::string(e.what()).find("exceeded its bound of 64") != std::string::npos;
    }
    check(rejected && legacy.size() == 64, "legacy 64-shape rejection unchanged");

    DdrBufferRegistry cache;
    auto a = std::make_shared<const uint8_t>(0);
    auto b = std::make_shared<const uint8_t>(0);
    const auto& ea = cache.get(key(4), opts, true, true, true, a);
    auto* a_ptr = ea.out.data_ptr();
    ea.out.fill_(3);
    cache.get(key(8), opts, true, true, true, b).out.fill_(7);
    const auto& again = cache.get(key(4), opts, true, true, true, a);
    check(again.out.data_ptr() == a_ptr && again.out.eq(3).all().item<bool>(),
          "live A survives A-B-A with original data/address");
    check(again.graph_lifetimes.size() == 1, "repeated registration is deduplicated");

    // Two graph signatures may point at the same ShapeKey.
    cache.get(key(4), opts, true, true, true, b);
    a.reset();
    const auto& shared = cache.get(key(4), opts, true, true, true, b);
    check(shared.out.data_ptr() == a_ptr && shared.graph_lifetimes.size() == 1,
          "last live graph retains a shared entry");
    b.reset();
    auto current = std::make_shared<const uint8_t>(0);
    cache.get(key(12), opts, true, true, true, current);
    check(cache.size() == 1, "both retired graphs release their entries");

    for (int64_t n = 1; n <= 100; ++n) {
        current = std::make_shared<const uint8_t>(0);
        const auto& e = cache.get(key(n), opts, true, true, true, current);
        check(cache.size() == 1 && e.out.size(1) == n,
              "100 successive retired shapes stay at one entry");
    }
    current.reset();
    for (int64_t n = 1; n <= 100; ++n) {
        cache.get(key(n), opts, true, true, true);
        check(cache.size() == 1, "PASSTHROUGH also bounds retired entries");
    }


    // A surviving tensor value owns its storage independently of the registry.
    // Entry& itself is not promised to survive retirement of its own key.
    DdrBufferRegistry immediate;
    auto expired_owner = std::make_shared<const uint8_t>(0);
    std::weak_ptr<const uint8_t> expired = expired_owner;
    expired_owner.reset();
    const auto& first = immediate.get(key(2), opts, false, true, true, expired);
    at::Tensor returned = first.out;
    returned.fill_(19);
    void* returned_ptr = returned.data_ptr();
    check(first.graph_lifetimes.empty(), "expired incoming token is not retained");
    immediate.get(key(3), opts, true, true, true, expired);
    check(immediate.size() == 1 && returned.data_ptr() == returned_ptr &&
              returned.eq(19).all().item<bool>(),
          "returned tensor survives registry entry retirement");
    const auto& fresh = immediate.get(key(2), opts, true, true, true, expired);
    check(immediate.size() == 1 && fresh.a.defined() && fresh.b.defined() &&
              fresh.out.data_ptr() != returned_ptr,
          "expired-token same-key access creates a fresh immediate entry");
    fresh.out.fill_(23);
    check(returned.eq(19).all().item<bool>(), "new immediate output cannot overwrite returned tensor");

    // The opt-in permits retirement, never eviction of any live graph.
    DdrBufferRegistry pinned;
    std::vector<std::shared_ptr<const uint8_t>> owners;
    for (int64_t n = 1; n <= 64; ++n) {
        owners.push_back(std::make_shared<const uint8_t>(0));
        pinned.get(key(n), opts, true, true, true, owners.back());
    }
    auto next = std::make_shared<const uint8_t>(0);
    rejected = false;
    try { pinned.get(key(65), opts, true, true, true, next); }
    catch (const c10::Error& error) {
        rejected = std::string(error.what()).find("exceeded its bound of 64") != std::string::npos;
    }
    check(rejected && pinned.size() == 64, "65 simultaneously live graph shapes still reject");
    owners.front().reset();
    pinned.get(key(65), opts, true, true, true, next);
    check(pinned.size() == 64, "only an expired owner makes room at the original bound");

    DdrBufferRegistry mixed;
    auto live = std::make_shared<const uint8_t>(0);
    mixed.get(key(1), opts, true, true);
    rejected = false;
    try { mixed.get(key(1), opts, true, true, true, live); }
    catch (const c10::Error&) { rejected = true; }
    check(rejected && mixed.size() == 1, "legacy entry cannot silently change lifetime policy");
    return checks;
}

#ifndef RPU_DDR_REGISTRY_HOST_ONLY
int64_t test_ddr_graph_lifetime() {
    int64_t checks = test_ddr_registry_lifetime();
    auto check = [&](bool ok, const char* label) {
        TORCH_CHECK(ok, "DDR graph lifetime: ", label);
        ++checks;
    };
    const auto opts = at::TensorOptions().dtype(at::kHalf).device(at::kCPU);
    auto key = [](int64_t n) { return ShapeKey{1, n, 16, at::kHalf, at::kCPU}; };
    DdrBufferRegistry cache;
    // Exercise actual Graph retirement, including reusing a raw Graph object.
    auto graph = std::make_unique<RpuKernelGraph>();
    graph->begin();
    auto token = graph->ddr_buffer_lifetime();
    cache.get(key(4), opts, true, true, true, token);
    graph->end();
    check(!token.expired(), "end alone retains the graph resource witness");
    graph->invalidate();
    check(token.expired(), "invalidate retires the graph resource witness");
    graph->begin();
    token = graph->ddr_buffer_lifetime();
    graph->abort();
    check(token.expired(), "abort retires the graph resource witness");
    graph->begin();
    token = graph->ddr_buffer_lifetime();
    graph->end();
    graph->begin();
    check(token.expired(), "new raw capture retires the old build");
    token = graph->ddr_buffer_lifetime();
    graph->end();
    graph.reset();
    check(token.expired(), "destruction retires the graph resource witness");
    cache.get(key(16), opts, true, true, true);
    check(cache.size() == 1, "actual graph retirement permits DDR reclamation");

    // An empty graph cannot become BUILT. A CPU data node exercises the real
    // retained-signature executor without requiring any model/device kernel.
    uint8_t replay_value = 0;
    RpuKernelGraph retained;
    GraphSignature signature;
    signature.op_id_str = "ddr_lifetime_retained_memset";
    signature.shapes = {1};
    retained.begin(signature);
    auto retained_token = retained.ddr_buffer_lifetime();
    retained.capture_data(MemsetNodeData{&replay_value, 17, 1});
    retained.end();
    check(retained.state() == RpuKernelGraph::State::BUILT && replay_value == 17,
          "signed nonempty graph executes BUILD and is retained");
    check(!retained_token.expired(), "retained BUILD end preserves token");
    retained.begin(signature);
    check(retained.state() == RpuKernelGraph::State::REPLAYING,
          "same signature selects actual REPLAY");
    auto replay_token = retained.ddr_buffer_lifetime();
    check(!retained_token.owner_before(replay_token) &&
              !replay_token.owner_before(retained_token) && !replay_token.expired(),
          "REPLAY returns the original lifetime control block");
    retained.capture_data(MemsetNodeData{&replay_value, 29, 1});
    retained.end();
    check(retained.state() == RpuKernelGraph::State::BUILT && replay_value == 29 &&
              !retained_token.expired(),
          "REPLAY end executes new payload and keeps the original token alive");
    retained.invalidate();
    check(retained_token.expired() && replay_token.expired(),
          "retained graph invalidation retires both token observations");
    return checks;
}

TORCH_LIBRARY_FRAGMENT(rpu, m) {
    m.def("test_ddr_graph_lifetime() -> int", TORCH_FN(test_ddr_graph_lifetime));
}

#endif
}  // namespace

#ifdef RPU_DDR_REGISTRY_HOST_ONLY
int main() {
    const auto checks = test_ddr_registry_lifetime();
    std::cout << "{\"completed\":true,\"scope\":\"ATen CPU registry; Graph lifecycle excluded\",\"checks\":"
              << checks << "}\n";
    return 0;
}
#endif
