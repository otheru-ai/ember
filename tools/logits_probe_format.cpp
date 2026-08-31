#include "logits_probe_format.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>

namespace {

class Sha256 {
public:
    void update(const uint8_t *data, size_t size) {
        constexpr uint64_t maximum_bytes =
            std::numeric_limits<uint64_t>::max() / 8U;
        if (size > maximum_bytes - byte_count_) {
            overflow_ = true;
            return;
        }
        byte_count_ += static_cast<uint64_t>(size);
        while (size > 0U) {
            const size_t count = std::min(size, block_.size() - used_);
            std::memcpy(block_.data() + used_, data, count);
            data += count;
            size -= count;
            used_ += count;
            if (used_ == block_.size()) {
                transform(block_.data());
                used_ = 0U;
            }
        }
    }

    bool finish(std::string &digest) {
        if (overflow_) return false;
        const uint64_t bit_count = byte_count_ * 8U;
        block_[used_++] = 0x80U;
        if (used_ > 56U) {
            std::fill(block_.begin() + static_cast<ptrdiff_t>(used_),
                      block_.end(), 0U);
            transform(block_.data());
            used_ = 0U;
        }
        std::fill(block_.begin() + static_cast<ptrdiff_t>(used_),
                  block_.begin() + 56, 0U);
        for (unsigned shift = 0U; shift < 64U; shift += 8U) {
            block_[63U - shift / 8U] =
                static_cast<uint8_t>(bit_count >> shift);
        }
        transform(block_.data());

        std::ostringstream output;
        output << std::hex << std::setfill('0');
        for (uint32_t word : state_) output << std::setw(8) << word;
        digest = output.str();
        return true;
    }

private:
    static uint32_t rotate_right(uint32_t value, unsigned shift) {
        return (value >> shift) | (value << (32U - shift));
    }

    void transform(const uint8_t *data) {
        static constexpr std::array<uint32_t, 64> constants = {
            0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
            0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
            0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
            0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
            0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
            0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
            0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
            0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
            0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
            0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
            0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
            0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
            0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
            0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
            0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
            0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
        };
        std::array<uint32_t, 64> words{};
        for (size_t i = 0; i < 16U; ++i) {
            const size_t p = i * 4U;
            words[i] = (static_cast<uint32_t>(data[p]) << 24U) |
                       (static_cast<uint32_t>(data[p + 1U]) << 16U) |
                       (static_cast<uint32_t>(data[p + 2U]) << 8U) |
                       static_cast<uint32_t>(data[p + 3U]);
        }
        for (size_t i = 16U; i < words.size(); ++i) {
            const uint32_t s0 = rotate_right(words[i - 15U], 7U) ^
                                rotate_right(words[i - 15U], 18U) ^
                                (words[i - 15U] >> 3U);
            const uint32_t s1 = rotate_right(words[i - 2U], 17U) ^
                                rotate_right(words[i - 2U], 19U) ^
                                (words[i - 2U] >> 10U);
            words[i] = words[i - 16U] + s0 + words[i - 7U] + s1;
        }
        uint32_t a = state_[0];
        uint32_t b = state_[1];
        uint32_t c = state_[2];
        uint32_t d = state_[3];
        uint32_t e = state_[4];
        uint32_t f = state_[5];
        uint32_t g = state_[6];
        uint32_t h = state_[7];
        for (size_t i = 0; i < words.size(); ++i) {
            const uint32_t s1 = rotate_right(e, 6U) ^ rotate_right(e, 11U) ^
                                rotate_right(e, 25U);
            const uint32_t choice = (e & f) ^ ((~e) & g);
            const uint32_t temp1 = h + s1 + choice + constants[i] + words[i];
            const uint32_t s0 = rotate_right(a, 2U) ^ rotate_right(a, 13U) ^
                                rotate_right(a, 22U);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = s0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<uint32_t, 8> state_ = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
    };
    std::array<uint8_t, 64> block_{};
    size_t used_ = 0U;
    uint64_t byte_count_ = 0U;
    bool overflow_ = false;
};

bool write_all(int fd, const uint8_t *data, size_t size, std::string &error) {
    size_t offset = 0U;
    while (offset < size) {
        const ssize_t count = write(fd, data + offset, size - offset);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            error = std::string("cannot write evidence file: ") +
                    std::strerror(count < 0 ? errno : EIO);
            return false;
        }
        offset += static_cast<size_t>(count);
    }
    return true;
}

bool write_exclusive_file(const std::string &path, const uint8_t *data,
                          size_t size, std::string &error) {
    const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
                        S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if (fd < 0) {
        error = "cannot create " + path + ": " + std::strerror(errno);
        return false;
    }
    bool ok = write_all(fd, data, size, error);
    if (ok && fsync(fd) != 0) {
        error = "cannot sync " + path + ": " + std::strerror(errno);
        ok = false;
    }
    if (close(fd) != 0 && ok) {
        error = "cannot close " + path + ": " + std::strerror(errno);
        ok = false;
    }
    if (!ok) (void)unlink(path.c_str());
    return ok;
}

std::string json_string(std::string_view input) {
    std::ostringstream output;
    output << '"';
    for (char raw : input) {
        const unsigned char c = static_cast<unsigned char>(raw);
        switch (c) {
        case '"': output << "\\\""; break;
        case '\\': output << "\\\\"; break;
        case '\b': output << "\\b"; break;
        case '\f': output << "\\f"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default:
            if (c < 0x20U) {
                output << "\\u" << std::hex << std::setw(4)
                       << std::setfill('0') << static_cast<unsigned>(c)
                       << std::dec;
            } else {
                output << static_cast<char>(c);
            }
        }
    }
    output << '"';
    return output.str();
}

std::vector<uint8_t> pack_logits(const std::vector<float> &logits,
                                 std::string &error) {
    static_assert(sizeof(float) == 4U, "logit evidence requires 32-bit float");
    if (logits.empty() ||
        logits.size() > std::numeric_limits<size_t>::max() / 4U) {
        error = "logit row is empty or too large";
        return {};
    }
    std::vector<uint8_t> bytes(logits.size() * 4U);
    for (size_t i = 0; i < logits.size(); ++i) {
        if (!std::isfinite(logits[i])) {
            error = "logit row contains a non-finite value";
            return {};
        }
        uint32_t word = 0U;
        std::memcpy(&word, &logits[i], sizeof(word));
        bytes[4U * i] = static_cast<uint8_t>(word);
        bytes[4U * i + 1U] = static_cast<uint8_t>(word >> 8U);
        bytes[4U * i + 2U] = static_cast<uint8_t>(word >> 16U);
        bytes[4U * i + 3U] = static_cast<uint8_t>(word >> 24U);
    }
    return bytes;
}

std::string sha256_bytes(const std::vector<uint8_t> &bytes) {
    Sha256 sha;
    sha.update(bytes.data(), bytes.size());
    std::string digest;
    if (!sha.finish(digest)) return {};
    return digest;
}

bool valid_lower_hex(const std::string &value, size_t length) {
    if (value.size() != length) return false;
    for (char c : value) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
            return false;
        }
    }
    return true;
}

bool valid_prefill_contract(const EmberLogitsProbeBundle &bundle) {
    return (bundle.prefill_mode == "exact-q1" && bundle.prefill_chunk == 1 &&
            bundle.comparison_role == "authority") ||
           (bundle.prefill_mode == "exact-q4" && bundle.prefill_chunk == 4 &&
            bundle.comparison_role == "exact-batching-control") ||
           (bundle.prefill_mode == "dense-q8" && bundle.prefill_chunk == 8 &&
            bundle.comparison_role ==
                "shape-matched-approximate-diagnostic");
}

}  // namespace

bool ember_parse_logits_probe_token(const char *text, int32_t &token,
                                    std::string &error) {
    if (!text || !text[0]) {
        error = "token id is empty";
        return false;
    }
    errno = 0;
    char *end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (errno == ERANGE || !end || *end != '\0' || value < 0L ||
        value > std::numeric_limits<int32_t>::max()) {
        error = "invalid non-negative token id: " + std::string(text);
        return false;
    }
    token = static_cast<int32_t>(value);
    return true;
}

bool ember_sha256_regular_file(const std::string &path, std::string &digest,
                               std::string &error) {
    digest.clear();
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) {
        error = "cannot open " + path + ": " + std::strerror(errno);
        return false;
    }
    struct stat status {};
    if (fstat(fd, &status) != 0 || !S_ISREG(status.st_mode)) {
        const int saved_errno = errno;
        close(fd);
        error = "SHA-256 input is not a regular file: " + path;
        if (saved_errno != 0) error += ": " + std::string(std::strerror(saved_errno));
        return false;
    }
    Sha256 sha;
    std::array<uint8_t, 1024U * 1024U> buffer{};
    for (;;) {
        const ssize_t count = read(fd, buffer.data(), buffer.size());
        if (count < 0 && errno == EINTR) continue;
        if (count < 0) {
            error = "cannot hash " + path + ": " + std::strerror(errno);
            close(fd);
            return false;
        }
        if (count == 0) break;
        sha.update(buffer.data(), static_cast<size_t>(count));
    }
#if defined(POSIX_FADV_DONTNEED)
    (void)posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
    if (close(fd) != 0) {
        error = "cannot close " + path + ": " + std::strerror(errno);
        return false;
    }
    if (!sha.finish(digest)) {
        error = "SHA-256 input is too large";
        return false;
    }
    return true;
}

bool ember_write_logits_probe_bundle(const std::string &directory,
                                     const EmberLogitsProbeBundle &bundle,
                                     std::string &error) {
    error.clear();
    if (directory.empty() || directory[0] != '/') {
        error = "output directory must be absolute";
        return false;
    }
    if (!valid_lower_hex(bundle.model_sha256, 64U) ||
        !valid_lower_hex(bundle.binary_sha256, 64U) ||
        !valid_lower_hex(bundle.ember_revision, 40U) ||
        !valid_prefill_contract(bundle) ||
        bundle.token_ids.empty()) {
        error = "bundle identity is incomplete";
        return false;
    }
    std::vector<uint8_t> payload = pack_logits(bundle.logits, error);
    if (payload.empty()) return false;
    const std::string payload_digest = sha256_bytes(payload);
    if (payload_digest.empty()) {
        error = "cannot hash logit payload";
        return false;
    }
    if (mkdir(directory.c_str(), S_IRWXU | S_IRGRP | S_IXGRP |
                                 S_IROTH | S_IXOTH) != 0) {
        error = "cannot create output directory " + directory + ": " +
                std::strerror(errno);
        return false;
    }

    const std::string logits_path = directory + "/logits.f32";
    const std::string manifest_path = directory + "/manifest.json";
    if (!write_exclusive_file(logits_path, payload.data(), payload.size(), error)) {
        (void)rmdir(directory.c_str());
        return false;
    }

    std::ostringstream manifest;
    manifest << "{\"schema\":\"ember-ds4-logits-v1\""
             << ",\"semantics\":\"next_token_after_final_input\""
             << ",\"format\":\"little-endian-f32\""
             << ",\"model_path\":" << json_string(bundle.model_path)
             << ",\"model_sha256\":" << json_string(bundle.model_sha256)
             << ",\"binary_sha256\":" << json_string(bundle.binary_sha256)
             << ",\"ember_revision\":" << json_string(bundle.ember_revision)
             << ",\"prefill\":" << json_string(bundle.prefill_mode)
             << ",\"prefill_chunk\":" << bundle.prefill_chunk
             << ",\"skip_intermediate_logits\":true"
             << ",\"comparison_role\":"
             << json_string(bundle.comparison_role)
             << ",\"placement\":\"monolithic-gpu\""
             << ",\"row_position\":" << (bundle.token_ids.size() - 1U)
             << ",\"vocab_width\":" << bundle.logits.size()
             << ",\"payload_sha256\":" << json_string(payload_digest)
             << ",\"token_ids\":[";
    for (size_t i = 0; i < bundle.token_ids.size(); ++i) {
        if (i != 0U) manifest << ',';
        manifest << bundle.token_ids[i];
    }
    manifest << "]}\n";
    const std::string manifest_text = manifest.str();
    if (!write_exclusive_file(
            manifest_path,
            reinterpret_cast<const uint8_t *>(manifest_text.data()),
            manifest_text.size(), error)) {
        (void)unlink(logits_path.c_str());
        (void)rmdir(directory.c_str());
        return false;
    }
    const int directory_fd = open(directory.c_str(), O_RDONLY | O_DIRECTORY |
                                                     O_CLOEXEC);
    if (directory_fd < 0 || fsync(directory_fd) != 0) {
        error = "cannot sync output directory " + directory;
        if (directory_fd >= 0) close(directory_fd);
        return false;
    }
    if (close(directory_fd) != 0) {
        error = "cannot close output directory " + directory;
        return false;
    }
    return true;
}
