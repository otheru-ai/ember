// Transactional activation-dump writer, shared by the model backends.
//
// One raw little-endian F32 record per prompt, `expected_floats` values in
// captured-layer-major order. That is OtherU's architecture-change contract and
// what tools/abliterate/extract_direction.py consumes.
//
// Rewriting through a same-directory temporary file is intentional. A killed
// process may lose the newest record, but it must never leave a SHORT record
// that a later extractor mistakes for evidence.
//
// The record size is the caller's contract and is passed in rather than being a
// compile-time constant: Qwen3.8-Flash-Next captures 48x2560, DeepSeek-V4
// captures 43x4096. A file holds records of ONE size; appending a differently
// shaped record to an existing dump is rejected, because the format is
// headerless and a size change would otherwise be silently unrecoverable.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

struct ActivationDumpResult {
    uint64_t ordinal = 0;
    uint64_t byte_offset = 0;
};

// Append one complete prompt record. `path` must be absolute. Existing output
// is accepted only when it is a regular file containing whole records of
// exactly `expected_floats` floats.
bool append_activation_dump(
    const std::string & path, const std::vector<float> & activations,
    size_t expected_floats, const char * model_label,
    ActivationDumpResult & result, std::string & error);

} // namespace dflash::common
