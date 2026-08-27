// Transactional activation-dump writer for Qwen3.8-Flash-Next.
//
// Direction extraction follows OtherU's architecture-change contract: one raw
// little-endian F32 record per prompt, with 48 layer rows of width 2560 in
// numeric layer order.  Rewriting through a same-directory temporary file is
// intentional.  A killed process may lose the newest record, but it must never
// leave a short record that a later extractor mistakes for evidence.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

struct Qwen4ExpActivationDumpResult {
    uint64_t ordinal = 0;
    uint64_t byte_offset = 0;
};

constexpr size_t kQwen4ExpActivationLayers = 48;
constexpr size_t kQwen4ExpActivationWidth = 2560;
constexpr size_t kQwen4ExpActivationFloats =
    kQwen4ExpActivationLayers * kQwen4ExpActivationWidth;

// Append one complete prompt record. `path` must be absolute. Existing output
// is accepted only when it is a regular file containing whole records.
bool qwen4exp_append_activation_dump(
    const std::string & path, const std::vector<float> & activations,
    Qwen4ExpActivationDumpResult & result, std::string & error);

} // namespace dflash::common
