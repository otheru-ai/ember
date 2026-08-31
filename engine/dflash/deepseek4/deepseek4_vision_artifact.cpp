#include "deepseek4_vision_artifact.h"

#include <cstddef>
#include <cstring>
#include <limits>

namespace dflash {
namespace {

constexpr uint8_t kMagic[8] = {'D', 'S', '4', 'V', 'I', 'M', 'G', 0};
constexpr uint32_t kVersion = 2;
constexpr size_t kHeaderBytes = 32;
constexpr size_t kMaxImageRows = 4096;

uint32_t read_u32(const uint8_t * p) {
    return static_cast<uint32_t>(p[0]) |
           static_cast<uint32_t>(p[1]) << 8 |
           static_cast<uint32_t>(p[2]) << 16 |
           static_cast<uint32_t>(p[3]) << 24;
}

uint64_t semantic_digest(const Deepseek4VisionArtifact & artifact) {
    uint64_t hash = 1469598103934665603ULL;
    const auto mix = [&hash](const void * data, size_t size) {
        const auto * bytes = static_cast<const uint8_t *>(data);
        for (size_t i = 0; i < size; ++i) {
            hash ^= bytes[i];
            hash *= 1099511628211ULL;
        }
    };
    mix(&artifact.n_embd, sizeof(artifact.n_embd));
    mix(&artifact.n_llm_h, sizeof(artifact.n_llm_h));
    mix(&artifact.n_llm_w, sizeof(artifact.n_llm_w));
    if (!artifact.image_embeddings.empty()) {
        mix(artifact.image_embeddings.data(),
            artifact.image_embeddings.size() * sizeof(float));
    }
    return hash == 0 ? 1 : hash;
}

}  // namespace

bool deepseek4_parse_vision_artifact(
        const std::vector<uint8_t> & bytes, int32_t model_n_embd,
        Deepseek4VisionArtifact & out, std::string & error) {
    out = {};
    error.clear();
    if (bytes.size() < kHeaderBytes) {
        error = "truncated DeepSeek4 vision artifact header";
        return false;
    }
    if (std::memcmp(bytes.data(), kMagic, sizeof(kMagic)) != 0) {
        error = "bad DeepSeek4 vision artifact magic";
        return false;
    }
    const uint32_t version = read_u32(bytes.data() + 8);
    if (version != kVersion) {
        error = "unsupported DeepSeek4 vision artifact version";
        return false;
    }
    const int32_t n_embd = static_cast<int32_t>(read_u32(bytes.data() + 12));
    const int32_t n_llm_h = static_cast<int32_t>(read_u32(bytes.data() + 16));
    const int32_t n_llm_w = static_cast<int32_t>(read_u32(bytes.data() + 20));
    const uint32_t flags = read_u32(bytes.data() + 24);
    const uint32_t reserved = read_u32(bytes.data() + 28);
    if (model_n_embd <= 0 || n_embd != model_n_embd) {
        error = "vision artifact embedding width does not match the model";
        return false;
    }
    if (n_llm_h <= 0 || n_llm_w <= 0) {
        error = "vision artifact requires a nonzero language grid";
        return false;
    }
    if (flags != 0 || reserved != 0) {
        error = "vision artifact has unsupported flags or reserved fields";
        return false;
    }

    const size_t h = static_cast<size_t>(n_llm_h);
    const size_t w = static_cast<size_t>(n_llm_w);
    if (h > kMaxImageRows || w > kMaxImageRows / h) {
        error = "vision artifact language grid is too large";
        return false;
    }
    const size_t rows = h * w;
    const size_t width = static_cast<size_t>(n_embd);
    if (rows > std::numeric_limits<size_t>::max() / width) {
        error = "vision artifact embedding matrix overflows";
        return false;
    }
    const size_t values = rows * width;
    if (values > (std::numeric_limits<size_t>::max() - kHeaderBytes) /
                     sizeof(float)) {
        error = "vision artifact byte size overflows";
        return false;
    }
    const size_t expected_bytes = kHeaderBytes + values * sizeof(float);
    if (bytes.size() != expected_bytes) {
        error = bytes.size() < expected_bytes
            ? "truncated vision artifact embedding data"
            : "trailing bytes after vision artifact embedding data";
        return false;
    }

    out.n_embd = n_embd;
    out.n_llm_h = n_llm_h;
    out.n_llm_w = n_llm_w;
    out.image_embeddings.resize(values);
    for (size_t i = 0; i < values; ++i) {
        const uint32_t bits = read_u32(
            bytes.data() + kHeaderBytes + i * sizeof(float));
        std::memcpy(&out.image_embeddings[i], &bits, sizeof(bits));
    }
    out.digest = semantic_digest(out);
    return true;
}

}  // namespace dflash
