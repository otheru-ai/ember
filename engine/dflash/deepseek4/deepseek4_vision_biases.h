// Metadata-only inventory gate for DeepSeek-V4-Flash-Vision-Exp router bias.
//
// The language GGUF must carry one F32 expert row per decoder layer under the
// exact converter spelling. This scanner intentionally reads only GGUF
// metadata: it can gate a full artifact without allocating or reading model
// weights, and shares the suffix predicate used by the production loaders.

#pragma once

#include <string>
#include <vector>

namespace dflash {

struct Deepseek4VisionBiasInventory {
    std::vector<int> layer_ids;
};

// Operator-supplied model path only. Request media never supplies this path.
bool deepseek4_inspect_vision_router_biases(
    const std::string & model_path, int expected_layers, int expected_experts,
    Deepseek4VisionBiasInventory & out, std::string & error);

}  // namespace dflash
