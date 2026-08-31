#include "deepseek4_vision_oracle.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <set>
#include <sys/stat.h>
#include <unistd.h>

namespace dflash {
namespace {

constexpr uint8_t kMagic[8] = {'D', 'S', '4', 'O', 'R', 'C', 0, 0};
constexpr uint8_t kBlockMagic[8] = {'D', 'S', '4', 'B', 'L', 'K', 0, 0};
constexpr uint32_t kVersion = 1;
constexpr uint32_t kRecordCount = 4;
constexpr uint32_t kBlockRecordCount = 13;
constexpr size_t kMaxMetadataBytes = 64U * 1024U;
constexpr size_t kMaxOracleBytes = 512U * 1024U * 1024U;

class Reader {
public:
    explicit Reader(const std::vector<uint8_t> & bytes) : bytes_(bytes) {}

    bool take(size_t count, const uint8_t *& value) {
        if (count > bytes_.size() - offset_) return false;
        value = bytes_.data() + offset_;
        offset_ += count;
        return true;
    }

    bool u32(uint32_t & value) {
        const uint8_t * p = nullptr;
        if (!take(4, p)) return false;
        value = static_cast<uint32_t>(p[0]) |
                static_cast<uint32_t>(p[1]) << 8 |
                static_cast<uint32_t>(p[2]) << 16 |
                static_cast<uint32_t>(p[3]) << 24;
        return true;
    }

    bool u64(uint64_t & value) {
        const uint8_t * p = nullptr;
        if (!take(8, p)) return false;
        value = 0;
        for (unsigned shift = 0; shift < 64; shift += 8) {
            value |= static_cast<uint64_t>(p[shift / 8]) << shift;
        }
        return true;
    }

    bool i64(int64_t & value) {
        uint64_t bits = 0;
        if (!u64(bits)) return false;
        std::memcpy(&value, &bits, sizeof(value));
        return true;
    }

    size_t offset() const { return offset_; }
    bool complete() const { return offset_ == bytes_.size(); }

private:
    const std::vector<uint8_t> & bytes_;
    size_t offset_ = 0;
};

uint64_t fnv1a(const uint8_t * data, size_t bytes) {
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= data[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

bool json_string(const nlohmann::json & object, const char * key,
                 std::string & value) {
    const auto found = object.find(key);
    if (found == object.end() || !found->is_string()) return false;
    value = found->get<std::string>();
    return !value.empty();
}

bool json_positive_int(const nlohmann::json & object, const char * key,
                       int & value) {
    const auto found = object.find(key);
    if (found == object.end() ||
        (!found->is_number_integer() && !found->is_number_unsigned())) {
        return false;
    }
    if (found->is_number_unsigned()) {
        const uint64_t parsed = found->get<uint64_t>();
        if (parsed == 0 || parsed > static_cast<uint64_t>(INT32_MAX)) {
            return false;
        }
        value = static_cast<int>(parsed);
        return true;
    }
    const int64_t parsed = found->get<int64_t>();
    if (parsed <= 0 || parsed > INT32_MAX) return false;
    value = static_cast<int>(parsed);
    return true;
}

bool json_zero_int(const nlohmann::json & object, const char * key) {
    const auto found = object.find(key);
    if (found == object.end() ||
        (!found->is_number_integer() && !found->is_number_unsigned())) {
        return false;
    }
    return found->is_number_unsigned()
        ? found->get<uint64_t>() == 0
        : found->get<int64_t>() == 0;
}

bool is_lower_hex(const std::string & value, size_t length) {
    return value.size() == length && std::all_of(
        value.begin(), value.end(), [](char c) {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        });
}

bool checked_elements(const std::vector<int64_t> & shape,
                      uint64_t & result) {
    result = 1;
    if (shape.empty() || shape.size() > 4) return false;
    for (int64_t extent : shape) {
        if (extent <= 0 || static_cast<uint64_t>(extent) >
                std::numeric_limits<uint64_t>::max() / result) {
            return false;
        }
        result *= static_cast<uint64_t>(extent);
    }
    return true;
}

struct Sha256 {
    std::array<uint32_t, 8> state {{
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
    }};
    std::array<uint8_t, 64> block {};
    size_t used = 0;
    uint64_t total = 0;

    static uint32_t rotate(uint32_t value, unsigned bits) {
        return (value >> bits) | (value << (32U - bits));
    }

    void compress(const uint8_t * input) {
        static constexpr uint32_t constants[64] = {
            0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,
            0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
            0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,
            0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
            0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,
            0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
            0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,
            0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
            0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,
            0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
            0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,
            0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
            0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,
            0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
            0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,
            0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U,
        };
        uint32_t words[64] = {};
        for (int i = 0; i < 16; ++i) {
            const size_t p = static_cast<size_t>(i) * 4;
            words[i] = static_cast<uint32_t>(input[p]) << 24 |
                       static_cast<uint32_t>(input[p + 1]) << 16 |
                       static_cast<uint32_t>(input[p + 2]) << 8 |
                       static_cast<uint32_t>(input[p + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotate(words[i - 15], 7) ^
                                rotate(words[i - 15], 18) ^
                                (words[i - 15] >> 3);
            const uint32_t s1 = rotate(words[i - 2], 17) ^
                                rotate(words[i - 2], 19) ^
                                (words[i - 2] >> 10);
            words[i] = words[i - 16] + s0 + words[i - 7] + s1;
        }
        uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
        uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t s1 = rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25);
            const uint32_t choice = (e & f) ^ (~e & g);
            const uint32_t temp1 = h + s1 + choice + constants[i] + words[i];
            const uint32_t s0 = rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = s0 + majority;
            h = g; g = f; f = e; e = d + temp1;
            d = c; c = b; b = a; a = temp1 + temp2;
        }
        state[0] += a; state[1] += b; state[2] += c; state[3] += d;
        state[4] += e; state[5] += f; state[6] += g; state[7] += h;
    }

    void update(const uint8_t * data, size_t size) {
        total += static_cast<uint64_t>(size);
        while (size > 0) {
            const size_t take = std::min(size, block.size() - used);
            std::memcpy(block.data() + used, data, take);
            used += take;
            data += take;
            size -= take;
            if (used == block.size()) {
                compress(block.data());
                used = 0;
            }
        }
    }

    std::array<uint8_t, 32> finish() {
        const uint64_t bits = total * 8U;
        block[used++] = 0x80;
        if (used > 56) {
            std::fill(block.begin() + static_cast<ptrdiff_t>(used),
                      block.end(), 0);
            compress(block.data());
            used = 0;
        }
        std::fill(block.begin() + static_cast<ptrdiff_t>(used),
                  block.begin() + 56, 0);
        for (unsigned i = 0; i < 8; ++i) {
            block[63U - i] = static_cast<uint8_t>(bits >> (i * 8));
        }
        compress(block.data());
        std::array<uint8_t, 32> digest {};
        for (size_t i = 0; i < state.size(); ++i) {
            digest[i * 4] = static_cast<uint8_t>(state[i] >> 24);
            digest[i * 4 + 1] = static_cast<uint8_t>(state[i] >> 16);
            digest[i * 4 + 2] = static_cast<uint8_t>(state[i] >> 8);
            digest[i * 4 + 3] = static_cast<uint8_t>(state[i]);
        }
        return digest;
    }
};

}  // namespace

static bool parse_vision_oracle_impl(
        const std::vector<uint8_t> & bytes,
        bool block_oracle, Deepseek4VisionOracle & out,
        std::string & error) {
    out = {};
    error.clear();
    Reader reader(bytes);
    const uint8_t * magic = nullptr;
    uint32_t version = 0;
    uint32_t record_count = 0;
    uint32_t metadata_length = 0;
    const uint8_t * expected_magic = block_oracle ? kBlockMagic : kMagic;
    const uint32_t expected_records = block_oracle
        ? kBlockRecordCount : kRecordCount;
    if (!reader.take(sizeof(kMagic), magic) ||
        std::memcmp(magic, expected_magic, sizeof(kMagic)) != 0 ||
        !reader.u32(version) || version != kVersion ||
        !reader.u32(record_count) || record_count != expected_records ||
        !reader.u32(metadata_length) || metadata_length == 0 ||
        metadata_length > kMaxMetadataBytes) {
        error = "invalid DeepSeek4 vision oracle header";
        return false;
    }
    const uint8_t * metadata = nullptr;
    if (!reader.take(metadata_length, metadata)) {
        error = "truncated DeepSeek4 vision oracle metadata";
        return false;
    }
    out.version = version;
    out.metadata_json.assign(
        reinterpret_cast<const char *>(metadata), metadata_length);
    nlohmann::json metadata_object;
    try {
        metadata_object = nlohmann::json::parse(out.metadata_json);
    } catch (const nlohmann::json::exception &) {
        error = "invalid DeepSeek4 vision oracle metadata JSON";
        out = {};
        return false;
    }
    if (!metadata_object.is_object()) {
        error = "invalid DeepSeek4 vision oracle metadata JSON";
        out = {};
        return false;
    }
    std::string lane;
    if (!json_string(metadata_object, "image", out.image) ||
        !json_string(metadata_object, "lane", lane) ||
        !json_string(metadata_object, "patch_digest", out.patch_digest) ||
        !json_string(metadata_object, "source_revision", out.source_revision) ||
        !json_string(metadata_object, "weight_digest", out.weight_digest) ||
        !json_positive_int(metadata_object, "n_vit_h", out.n_vit_h) ||
        !json_positive_int(metadata_object, "n_vit_w", out.n_vit_w) ||
        !json_positive_int(metadata_object, "n_llm_h", out.n_llm_h) ||
        !json_positive_int(metadata_object, "n_llm_w", out.n_llm_w) ||
        (block_oracle && !json_zero_int(metadata_object, "block")) ||
        !is_lower_hex(out.patch_digest, 32) ||
        !is_lower_hex(out.weight_digest, 32) ||
        (lane != "bf16" && lane != "f32")) {
        error = "invalid DeepSeek4 vision oracle metadata contract";
        out = {};
        return false;
    }
    out.lane = lane == "bf16"
        ? Deepseek4VisionOracleLane::BF16 : Deepseek4VisionOracleLane::F32;
    out.block = block_oracle ? 0 : -1;

    std::set<std::string> names;
    out.records.reserve(record_count);
    for (uint32_t record_index = 0; record_index < record_count;
         ++record_index) {
        uint32_t name_length = 0;
        uint32_t raw_lane = 0;
        uint32_t rank = 0;
        if (!reader.u32(name_length) || name_length == 0 ||
            name_length > 128) {
            error = "invalid DeepSeek4 vision oracle record name";
            out = {};
            return false;
        }
        const uint8_t * raw_name = nullptr;
        if (!reader.take(name_length, raw_name) ||
            !reader.u32(raw_lane) || raw_lane > 1 ||
            !reader.u32(rank) || rank == 0 || rank > 4) {
            error = "truncated DeepSeek4 vision oracle record header";
            out = {};
            return false;
        }
        Deepseek4VisionOracleRecord record;
        record.name.assign(
            reinterpret_cast<const char *>(raw_name), name_length);
        record.lane = static_cast<Deepseek4VisionOracleLane>(raw_lane);
        record.shape.resize(rank);
        for (uint32_t dim = 0; dim < rank; ++dim) {
            if (!reader.i64(record.shape[dim])) {
                error = "truncated DeepSeek4 vision oracle shape";
                out = {};
                return false;
            }
        }
        uint64_t element_count = 0;
        uint64_t shape_elements = 0;
        if (!reader.u64(element_count) || !reader.u64(record.checksum) ||
            !checked_elements(record.shape, shape_elements) ||
            element_count != shape_elements ||
            element_count > static_cast<uint64_t>(
                std::numeric_limits<size_t>::max() / sizeof(float))) {
            error = "invalid DeepSeek4 vision oracle element count";
            out = {};
            return false;
        }
        const size_t payload_bytes =
            static_cast<size_t>(element_count) * sizeof(float);
        const uint8_t * payload = nullptr;
        if (!reader.take(payload_bytes, payload) ||
            fnv1a(payload, payload_bytes) != record.checksum) {
            error = "DeepSeek4 vision oracle payload checksum differs";
            out = {};
            return false;
        }
        record.values.resize(static_cast<size_t>(element_count));
        for (size_t i = 0; i < record.values.size(); ++i) {
            uint32_t bits = static_cast<uint32_t>(payload[i * 4]) |
                            static_cast<uint32_t>(payload[i * 4 + 1]) << 8 |
                            static_cast<uint32_t>(payload[i * 4 + 2]) << 16 |
                            static_cast<uint32_t>(payload[i * 4 + 3]) << 24;
            std::memcpy(&record.values[i], &bits, sizeof(bits));
        }
        if (record.lane != out.lane || !names.insert(record.name).second) {
            error = "DeepSeek4 vision oracle lane or record name differs";
            out = {};
            return false;
        }
        out.records.push_back(std::move(record));
    }
    int vit_rows = 0;
    int llm_rows = 0;
    if (out.n_vit_h > INT32_MAX / out.n_vit_w ||
        out.n_llm_h > INT32_MAX / out.n_llm_w) {
        error = "DeepSeek4 vision oracle grids overflow";
        out = {};
        return false;
    }
    vit_rows = out.n_vit_h * out.n_vit_w;
    llm_rows = out.n_llm_h * out.n_llm_w;
    struct Expected {
        const char * name;
        std::vector<int64_t> shape;
    };
    std::vector<Expected> expected;
    if (block_oracle) {
        expected = {
            {"norm1_out", {vit_rows, 1024}},
            {"qkv_biased", {vit_rows, 3072}},
            {"q_roped", {vit_rows, 16, 64}},
            {"k_roped", {vit_rows, 16, 64}},
            {"v_in", {vit_rows, 16, 64}},
            {"sdpa_out", {vit_rows, 1024}},
            {"wo_biased", {vit_rows, 1024}},
            {"post_attn_residual", {vit_rows, 1024}},
            {"norm2_out", {vit_rows, 1024}},
            {"mlp_gate", {vit_rows, 2816}},
            {"mlp_up", {vit_rows, 2816}},
            {"mlp_silu_act", {vit_rows, 2816}},
            {"mlp_down_out", {vit_rows, 1024}},
        };
    } else {
        expected = {
            {"post_patch_projection", {vit_rows, 1024}},
            {"post_block_0", {vit_rows, 1024}},
            {"post_vit", {vit_rows, 1024}},
            {"post_aligner", {llm_rows, 4096}},
        };
    }
    for (const auto & item : expected) {
        const auto * record = deepseek4_vision_oracle_record(out, item.name);
        if (!record || record->shape != item.shape) {
            error = "DeepSeek4 vision oracle checkpoint shape differs: " +
                    std::string(item.name);
            out = {};
            return false;
        }
    }
    if (!reader.complete()) {
        error = "trailing bytes after DeepSeek4 vision oracle";
        out = {};
        return false;
    }
    return true;
}

bool deepseek4_parse_vision_oracle(
        const std::vector<uint8_t> & bytes,
        Deepseek4VisionOracle & out, std::string & error) {
    return parse_vision_oracle_impl(bytes, false, out, error);
}

bool deepseek4_parse_vision_block_oracle(
        const std::vector<uint8_t> & bytes,
        Deepseek4VisionOracle & out, std::string & error) {
    return parse_vision_oracle_impl(bytes, true, out, error);
}

static bool load_vision_oracle_impl(
        const std::string & path, Deepseek4VisionOracle & out,
        bool block_oracle, std::string & error) {
    out = {};
    error.clear();
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        error = "cannot open DeepSeek4 vision oracle";
        return false;
    }
    struct stat st {};
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size <= 0 ||
        static_cast<uintmax_t>(st.st_size) > kMaxOracleBytes) {
        close(fd);
        error = "invalid DeepSeek4 vision oracle file";
        return false;
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(st.st_size));
    size_t offset = 0;
    while (offset < bytes.size()) {
        const ssize_t count = read(fd, bytes.data() + offset,
                                   bytes.size() - offset);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            close(fd);
            error = "cannot read complete DeepSeek4 vision oracle";
            return false;
        }
        offset += static_cast<size_t>(count);
    }
    if (close(fd) != 0) {
        error = "cannot close DeepSeek4 vision oracle";
        return false;
    }
    return parse_vision_oracle_impl(bytes, block_oracle, out, error);
}

bool deepseek4_load_vision_oracle(
        const std::string & path, Deepseek4VisionOracle & out,
        std::string & error) {
    return load_vision_oracle_impl(path, out, false, error);
}

bool deepseek4_load_vision_block_oracle(
        const std::string & path, Deepseek4VisionOracle & out,
        std::string & error) {
    return load_vision_oracle_impl(path, out, true, error);
}

const Deepseek4VisionOracleRecord * deepseek4_vision_oracle_record(
        const Deepseek4VisionOracle & oracle, const std::string & name) {
    for (const auto & record : oracle.records) {
        if (record.name == name) return &record;
    }
    return nullptr;
}

std::string deepseek4_vision_patch_digest(
        const std::vector<uint16_t> & bf16_patches) {
    Sha256 hash;
    std::array<uint8_t, 4096> bytes {};
    size_t offset = 0;
    while (offset < bf16_patches.size()) {
        const size_t count = std::min(
            bf16_patches.size() - offset, bytes.size() / 2);
        for (size_t i = 0; i < count; ++i) {
            const uint16_t value = bf16_patches[offset + i];
            bytes[i * 2] = static_cast<uint8_t>(value);
            bytes[i * 2 + 1] = static_cast<uint8_t>(value >> 8);
        }
        hash.update(bytes.data(), count * 2);
        offset += count;
    }
    const auto digest = hash.finish();
    static constexpr char hex[] = "0123456789abcdef";
    std::string result(32, '0');
    for (size_t i = 0; i < 16; ++i) {
        result[i * 2] = hex[digest[i] >> 4];
        result[i * 2 + 1] = hex[digest[i] & 0x0fU];
    }
    return result;
}

bool deepseek4_compare_vision_checkpoint(
        const std::vector<float> & actual,
        const Deepseek4VisionOracleRecord & expected,
        Deepseek4VisionOracleComparison & out,
        std::string & error) {
    out = {};
    error.clear();
    if (actual.size() != expected.values.size() || actual.empty()) {
        error = "DeepSeek4 vision checkpoint element count differs";
        return false;
    }
    double squared_error = 0.0;
    double squared_expected = 0.0;
    double squared_actual = 0.0;
    double dot = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double a = actual[i];
        const double e = expected.values[i];
        if (!std::isfinite(a) || !std::isfinite(e)) {
            error = "DeepSeek4 vision checkpoint contains a non-finite value";
            return false;
        }
        const double delta = a - e;
        squared_error += delta * delta;
        squared_expected += e * e;
        squared_actual += a * a;
        dot += a * e;
        const float absolute = static_cast<float>(std::fabs(delta));
        if (absolute > out.max_abs) {
            out.max_abs = absolute;
            out.max_abs_index = i;
        }
    }
    if (squared_expected == 0.0 || squared_actual == 0.0) {
        error = "DeepSeek4 vision checkpoint has zero norm";
        return false;
    }
    out.relative_l2 = std::sqrt(squared_error / squared_expected);
    out.cosine = dot / std::sqrt(squared_actual * squared_expected);
    return true;
}

bool deepseek4_vision_ratio_gate(
        const Deepseek4VisionOracleComparison & engine_vs_bf16,
        const Deepseek4VisionOracleComparison & f32_vs_bf16,
        double maximum_ratio, double & ratio) {
    ratio = std::numeric_limits<double>::infinity();
    if (!std::isfinite(engine_vs_bf16.relative_l2) ||
        !std::isfinite(engine_vs_bf16.cosine) ||
        !std::isfinite(engine_vs_bf16.max_abs) ||
        !std::isfinite(f32_vs_bf16.relative_l2) ||
        !std::isfinite(f32_vs_bf16.cosine) ||
        !std::isfinite(f32_vs_bf16.max_abs) ||
        engine_vs_bf16.relative_l2 < 0.0 ||
        f32_vs_bf16.relative_l2 <= 0.0 ||
        !std::isfinite(maximum_ratio) || maximum_ratio <= 0.0) {
        return false;
    }
    ratio = engine_vs_bf16.relative_l2 / f32_vs_bf16.relative_l2;
    return std::isfinite(ratio) && ratio <= maximum_ratio;
}

bool deepseek4_vision_budget_gate(
        const Deepseek4VisionOracleComparison & engine_vs_bf16,
        double maximum_relative_l2, double & consumption) {
    if (!(maximum_relative_l2 > 0.0) ||
        !std::isfinite(maximum_relative_l2) ||
        !std::isfinite(engine_vs_bf16.relative_l2) ||
        engine_vs_bf16.relative_l2 < 0.0) {
        consumption = std::numeric_limits<double>::infinity();
        return false;
    }
    consumption = engine_vs_bf16.relative_l2 / maximum_relative_l2;
    return std::isfinite(consumption) && consumption <= 1.0;
}

}  // namespace dflash
