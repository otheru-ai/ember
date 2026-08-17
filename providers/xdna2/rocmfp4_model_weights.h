// Bounds-checked extraction of trained DSpark routed-expert weights.
//
// The XDNA graph consumes one pre-tiled expert at a time, while GGUF stores
// each projection as [input, output, expert].  This seam reads only requested
// expert slices and losslessly packs them; it never mmap's or copies an entire
// 256-expert tensor.  Keeping the raw slices alongside the packed image gives
// the hardware validator an independent scalar reference.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ember::xdna2 {

constexpr int kRocmfp4ModelLayers = 3;
constexpr int kRocmfp4ModelExperts = 256;
constexpr uint32_t kRocmfp4FastGgmlType = 101;

struct Rocmfp4ModelExpert {
    int expert_id = -1;
    std::vector<uint8_t> gate;
    std::vector<uint8_t> up;
    std::vector<uint8_t> down;
    std::vector<uint8_t> packed;
};

bool load_rocmfp4_model_experts(const char * path,
                                int layer,
                                const std::vector<int> & expert_ids,
                                std::vector<Rocmfp4ModelExpert> & experts,
                                std::string * error = nullptr);

}  // namespace ember::xdna2
