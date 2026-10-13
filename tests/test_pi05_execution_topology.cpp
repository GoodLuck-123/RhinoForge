#include "pi05_execution_topology.h"
#include <iostream>
#include <stdexcept>

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template <class F> void rejects(F&& function) {
    try { function(); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid Pi0.5 profile was accepted");
}
}

int main() {
    using namespace v3;
    for (const bool expert : {false, true}) {
        const int64_t h = expert ? 1024 : 2048;
        const int64_t logical = expert ? 4096 : 16384;
        for (const bool w8 : {false, true}) {
            for (const int cores : {4, 6}) {
                const int64_t physical = validate_pi05_reduced_geometry(
                    cores, 8, 4, 256, h, logical, 18, expert, w8);
                const int64_t granule = cores * (w8 ? 32 : 16);
                check(physical >= logical && physical % granule == 0,
                      "row partition must contain all channels and align K");
                check(physical - logical < granule,
                      "padding must be the minimum cold packing allowance");
                check(physical / cores % 16 == 0,
                      "column partition must preserve the vector width");
                const auto topology = resolve_pi05_reduced_physical_topology(
                    cores, 8, 4, 256, h, physical, expert);
                check(topology.num_cores == cores && topology.attention_tp == 4 &&
                          topology.mlp_tp == cores && topology.physical_kv_cores == 8,
                      "root, attention, MLP and physical KV lanes differ");
                std::array<std::array<int64_t, 2>, 7> shapes{{
                    {2048, h}, {1024, h}, {1024, h}, {h, 2048},
                    {physical, h}, {physical, h}, {h, physical}}};
                validate_pi05_reduced_projection_shapes(shapes, h, physical);
                // An old unpadded payload cannot masquerade as physical MLP6.
                if (cores == 6) {
                    shapes[6][1] = logical;
                    rejects([&] { validate_pi05_reduced_projection_shapes(shapes, h, physical); });
                }
                // Checkpoint geometry is logical, never a padded model width.
                rejects([&] { validate_pi05_reduced_geometry(
                    cores, 8, 4, 256, h, logical + 16, 18, expert, w8); });
                rejects([&] { validate_pi05_reduced_geometry(
                    cores, 8, 4, 256, h, logical, 17, expert, w8); });
                rejects([&] { validate_pi05_reduced_geometry(
                    cores, 8, 1, 256, h, logical, 18, expert, w8); });
                rejects([&] { validate_pi05_reduced_geometry(
                    cores, 8, 8, 256, h, logical, 18, expert, w8); });
                rejects([&] { validate_pi05_reduced_geometry(
                    cores, 6, 4, 256, h, logical, 18, expert, w8); });
                rejects([&] { resolve_pi05_reduced_physical_topology(
                    cores, 8, 4, 256, h, physical + 2 * granule, expert); });
                // Pi0.5 must not expand unrelated owners' generic admission.
                rejects([&] { resolve_decoder_execution_topology(
                    cores, 8, 4, 256, h, logical); });
            }
            check(pi05_mlp_physical_size(logical, 8, w8) == logical,
                  "existing eight-core width changed");
        }
    }
    check(pi05_mlp_physical_size(16384, 6, false) == 16416, "FP16 prefix width");
    check(pi05_mlp_physical_size(4096, 6, false) == 4128, "FP16 expert width");
    check(pi05_mlp_physical_size(16384, 6, true) == 16512, "W8 prefix width");
    check(pi05_mlp_physical_size(4096, 6, true) == 4224, "W8 expert width");
    for (const int invalid : {0, 1, 2, 3, 5, 7, 9})
        rejects([&] { pi05_mlp_physical_size(4096, invalid, false); });
    std::cout << "Pi0.5 reduced-core native geometry checks passed\n";
}
