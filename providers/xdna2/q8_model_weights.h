// Bounds-checked extraction of trained DSpark Q8_0 shared-expert weights.
//
// GGUF keeps one dense shared expert per draft layer.  This seam validates the
// model identity, tensor type, shape, and complete file extent before reading
// the three projections, then performs the one-time compensated Q8_0 AIE pack.
// Raw projections remain available for an independent quantized reference.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ember::xdna2 {

constexpr int kQ8ModelLayers = 3;
constexpr uint32_t kQ8GgmlType = 8;

struct Q8ModelSharedExpert {
    int layer = -1;
    std::vector<uint8_t> gate;
    std::vector<uint8_t> up;
    std::vector<uint8_t> down;
    std::vector<uint8_t> packed;
};

bool load_q8_model_shared_expert(const char * path,
                                 int layer,
                                 Q8ModelSharedExpert & expert,
                                 std::string * error = nullptr);

}  // namespace ember::xdna2
