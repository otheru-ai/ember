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

struct Q8ModelProjection {
    std::string name;
    int k = 0;
    int n = 0;
    std::vector<uint8_t> raw;
    std::vector<uint8_t> packed;
};

// Load one named DSpark Q8_0 matrix and put it in the generic projection
// graph's group/column/K-tile/row order.  This is the measurement seam for
// attention and output projections; it deliberately does not guess tensor
// shapes from untrusted GGUF metadata.
bool load_q8_model_projection(const char * path,
                              const char * tensor_name,
                              int k,
                              int n,
                              Q8ModelProjection & projection,
                              std::string * error = nullptr);

bool load_q8_model_shared_expert(const char * path,
                                 int layer,
                                 Q8ModelSharedExpert & expert,
                                 std::string * error = nullptr);

}  // namespace ember::xdna2
