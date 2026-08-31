#include "deepseek4_vision_artifact.h"

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sys/stat.h>
#include <unistd.h>

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

bool deepseek4_load_vision_artifact(
        const std::string & path, int32_t model_n_embd,
        Deepseek4VisionArtifact & out, std::string & error) {
    out = {};
    error.clear();
    if (path.empty() || model_n_embd <= 0) {
        error = "invalid DeepSeek4 vision artifact path or model width";
        return false;
    }
    const size_t width = static_cast<size_t>(model_n_embd);
    if (width > (std::numeric_limits<size_t>::max() - kHeaderBytes) /
                    (kMaxImageRows * sizeof(float))) {
        error = "DeepSeek4 vision artifact size bound overflows";
        return false;
    }
    const size_t max_bytes = kHeaderBytes +
                             kMaxImageRows * width * sizeof(float);
    const int fd = open(path.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) {
        error = "cannot open DeepSeek4 vision artifact";
        return false;
    }
    struct stat st {};
    if (fstat(fd, &st) != 0) {
        close(fd);
        error = "cannot stat DeepSeek4 vision artifact";
        return false;
    }
    if (!S_ISREG(st.st_mode)) {
        close(fd);
        error = "DeepSeek4 vision artifact is not a regular file";
        return false;
    }
    if (st.st_size < 0 ||
        static_cast<uintmax_t>(st.st_size) >
            static_cast<uintmax_t>(max_bytes)) {
        close(fd);
        error = "DeepSeek4 vision artifact file is too large";
        return false;
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(st.st_size));
    size_t offset = 0;
    while (offset < bytes.size()) {
        const ssize_t n = read(fd, bytes.data() + offset,
                               bytes.size() - offset);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            close(fd);
            error = "cannot read complete DeepSeek4 vision artifact";
            return false;
        }
        offset += static_cast<size_t>(n);
    }
    uint8_t trailing = 0;
    ssize_t trailing_read;
    do {
        trailing_read = read(fd, &trailing, 1);
    } while (trailing_read < 0 && errno == EINTR);
    if (trailing_read != 0) {
        close(fd);
        error = trailing_read > 0
            ? "DeepSeek4 vision artifact grew while being read"
            : "cannot verify DeepSeek4 vision artifact length";
        return false;
    }
    if (close(fd) != 0) {
        error = "cannot close DeepSeek4 vision artifact";
        return false;
    }
    return deepseek4_parse_vision_artifact(
        bytes, model_n_embd, out, error);
}

}  // namespace dflash
