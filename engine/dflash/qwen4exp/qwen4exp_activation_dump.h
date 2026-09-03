// Qwen3.8-Flash-Next activation-record shape.
//
// The writer itself is shared with the other backends; see
// engine/dflash/common/activation_dump.h. Only the record geometry is
// architecture-specific and lives here.

#pragma once

#include <cstddef>

namespace dflash::common {

constexpr size_t kQwen4ExpActivationLayers = 48;
constexpr size_t kQwen4ExpActivationWidth = 2560;
constexpr size_t kQwen4ExpActivationFloats =
    kQwen4ExpActivationLayers * kQwen4ExpActivationWidth;

} // namespace dflash::common
