// DeepSeek-V4 target metadata contracts that must fail before allocation.
//
// Kept dependency-free so strict GPU-free tests can prove that the diagnostic
// exception remains narrower than the production architecture contract.

#pragma once

#include <cstdint>

namespace dflash::common {

inline bool deepseek4_layer_contract_supported(
        uint32_t n_layer, uint32_t n_hash_layer,
        bool allow_single_layer_control) {
    return (n_layer == 43U && n_hash_layer == 3U) ||
           (allow_single_layer_control && n_layer == 1U &&
            n_hash_layer == 1U);
}

}  // namespace dflash::common
