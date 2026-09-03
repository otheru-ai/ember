// Streaming ROCMI4 GGUF quantizer used by the pinned release pipeline.
//
// The implementation is architecture-neutral: GGUF metadata supplies the
// tensor inventory and shapes, while a conservative name/rank policy chooses
// ordinary weights. Explicit, non-overlapping --tensor-type regexes are
// evaluated first; the released control forces the PLE table through type 108,
// while bakeoff artifacts may select Q6_K, ROCmFP4 FAST, or the compatible
// ROCmFPX FP3 row-gather encoding per tensor.

#include "gguf_quantize_common.h"

#include "ggml.h"
#include "gguf.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <unordered_map>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef EMBER_CONFIGURED_GIT_HEAD
#error "EMBER_CONFIGURED_GIT_HEAD must be supplied by CMake"
#endif

namespace quant = ember::gguf_quantize;

namespace {

constexpr std::size_t kWorkingSetBytes = 64U * 1024U * 1024U;
constexpr std::size_t kAlignment = GGUF_DEFAULT_ALIGNMENT;

using GgufContext = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
using GgmlContext = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using File = std::unique_ptr<FILE, int (*)(FILE *)>;

struct InterventionDirection {
    std::string id;
    std::vector<float> values;
};

struct InterventionTarget {
    std::string tensor_name;
    std::size_t direction_index = 0;
    float scale = 0.0f;
    bool preserve_row_norm = false;
    std::size_t expected_columns = 0;
    std::size_t expected_rows = 0;
};

struct InterventionPlan {
    std::string manifest_sha256;
    std::string target_names_sha256;
    std::vector<InterventionDirection> directions;
    std::vector<InterventionTarget> targets;
    std::unordered_map<std::string, std::size_t> target_by_name;
};

struct TensorPlan {
    const ggml_tensor * source = nullptr;
    std::string name;
    std::size_t source_offset = 0;
    std::size_t source_size = 0;
    std::size_t output_size = 0;
    bool quantize = false;
    bool manual = false;
    enum ggml_type output_type = GGML_TYPE_F32;
    const InterventionTarget * intervention = nullptr;
    const InterventionDirection * intervention_direction = nullptr;
};

struct SplitInfo {
    std::uint16_t number = 0;
    std::uint16_t count = 1;
    std::int32_t tensor_count = 0;
};

struct ShardPlan {
    ShardPlan() : input(nullptr, &std::fclose),
                  input_gguf(nullptr, &gguf_free),
                  input_meta(nullptr, &ggml_free),
                  output_gguf(nullptr, &gguf_free) {}
    ShardPlan(const ShardPlan &) = delete;
    ShardPlan & operator=(const ShardPlan &) = delete;

    std::filesystem::path input_path;
    std::filesystem::path output_path;
    File input;
    GgufContext input_gguf;
    GgmlContext input_meta;
    GgufContext output_gguf;
    std::vector<TensorPlan> tensors;
    SplitInfo split;
    std::uint64_t output_size = 0;
};

class OwnedPartial {
public:
    explicit OwnedPartial(const std::filesystem::path & output) {
        struct stat status {};
        if (::lstat(output.c_str(), &status) == 0) {
            throw std::runtime_error("output already exists: " + output.string());
        }
        if (errno != ENOENT) {
            throw std::runtime_error("cannot inspect output path: " +
                                     std::string(std::strerror(errno)));
        }
        for (unsigned int attempt = 0; attempt < 1000; ++attempt) {
            path_ = output.string() + ".partial." +
                    std::to_string(static_cast<unsigned long long>(::getpid())) +
                    "." + std::to_string(attempt);
            const int descriptor = ::open(path_.c_str(),
                                          O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
                                          0644);
            if (descriptor >= 0) {
                FILE * stream = ::fdopen(descriptor, "wb");
                if (!stream) {
                    const int saved_errno = errno;
                    (void)::close(descriptor);
                    (void)::unlink(path_.c_str());
                    throw std::runtime_error("fdopen failed: " +
                                             std::string(std::strerror(saved_errno)));
                }
                file_ = File(stream, &std::fclose);
                return;
            }
            if (errno != EEXIST) {
                throw std::runtime_error("cannot create partial output: " +
                                         std::string(std::strerror(errno)));
            }
        }
        throw std::runtime_error("cannot allocate a unique partial output name");
    }

    ~OwnedPartial() {
        file_.reset();
        if (!path_.empty()) {
            (void)::unlink(path_.c_str());
        }
    }

    OwnedPartial(const OwnedPartial &) = delete;
    OwnedPartial & operator=(const OwnedPartial &) = delete;

    FILE * file() const { return file_.get(); }
    const std::string & path() const { return path_; }

    void retain_for_recovery() { path_.clear(); }

    void sync_and_close() {
        if (std::fflush(file_.get()) != 0 || ::fsync(::fileno(file_.get())) != 0) {
            throw std::runtime_error("failed to sync partial output: " +
                                     std::string(std::strerror(errno)));
        }
        if (std::fclose(file_.release()) != 0) {
            throw std::runtime_error("failed to close partial output: " +
                                     std::string(std::strerror(errno)));
        }
    }

    void promote_no_clobber(const std::filesystem::path & output) {
        if (file_) {
            throw std::logic_error("partial output must be closed before promotion");
        }
        // link(2) is an atomic, same-filesystem no-clobber promotion: unlike
        // rename(2), it can never replace a destination created concurrently.
        if (::link(path_.c_str(), output.c_str()) != 0) {
            throw std::runtime_error("cannot promote output without clobbering: " +
                                     std::string(std::strerror(errno)));
        }
        // Keep the partial hard link until the transaction marker records
        // COMPLETE. On failure the partial is retained for explicit recovery;
        // Ember never attempts a racy conditional unlink of final names.

        const std::filesystem::path parent = output.has_parent_path()
            ? output.parent_path() : std::filesystem::path(".");
        const int directory = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (directory >= 0) {
            if (::fsync(directory) != 0) {
                std::fprintf(stderr, "warning: could not sync output directory: %s\n",
                             std::strerror(errno));
            }
            (void)::close(directory);
        }
    }

private:
    std::string path_;
    File file_{nullptr, &std::fclose};
};

class TransactionMarker {
public:
    explicit TransactionMarker(const std::filesystem::path & output) {
        path_ = output.string() + ".transaction." +
                std::to_string(static_cast<unsigned long long>(::getpid())) +
                ".marker";
        descriptor_ = ::open(path_.c_str(),
                             O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
        if (descriptor_ < 0) {
            throw std::runtime_error("cannot create quantization transaction marker: " +
                                     std::string(std::strerror(errno)));
        }
        struct stat status {};
        if (::fstat(descriptor_, &status) != 0) {
            const int saved_errno = errno;
            (void)::close(descriptor_);
            descriptor_ = -1;
            throw std::runtime_error("cannot identify quantization transaction marker: " +
                                     std::string(std::strerror(saved_errno)));
        }
        device_ = status.st_dev;
        inode_ = status.st_ino;
        static constexpr char message[] =
            "incomplete multi-shard publication; retain .partial files for recovery\n";
        const ssize_t written = ::write(descriptor_, message, sizeof(message) - 1U);
        if (written != static_cast<ssize_t>(sizeof(message) - 1U) ||
            ::fsync(descriptor_) != 0) {
            const int saved_errno = errno;
            (void)::close(descriptor_);
            descriptor_ = -1;
            throw std::runtime_error("cannot sync quantization transaction marker: " +
                                     std::string(std::strerror(saved_errno)));
        }
        try {
            sync_parent();
        } catch (...) {
            (void)::close(descriptor_);
            descriptor_ = -1;
            throw;
        }
    }

    ~TransactionMarker() {
        if (descriptor_ >= 0) {
            (void)::close(descriptor_);
        }
    }

    TransactionMarker(const TransactionMarker &) = delete;
    TransactionMarker & operator=(const TransactionMarker &) = delete;

    const std::string & path() const { return path_; }

    void commit() {
        static constexpr char complete[] = "COMPLETE\n";
        if (::ftruncate(descriptor_, 0) != 0 ||
            ::pwrite(descriptor_, complete, sizeof(complete) - 1U, 0) !=
                static_cast<ssize_t>(sizeof(complete) - 1U) ||
            ::fsync(descriptor_) != 0) {
            throw std::runtime_error("cannot commit quantization transaction marker: " +
                                     std::string(std::strerror(errno)));
        }
        struct stat status {};
        if (::lstat(path_.c_str(), &status) != 0 ||
            status.st_dev != device_ || status.st_ino != inode_) {
            throw std::runtime_error("quantization transaction marker was replaced before commit");
        }
        if (::close(descriptor_) != 0) {
            descriptor_ = -1;
            throw std::runtime_error("cannot close completed transaction marker: " +
                                     std::string(std::strerror(errno)));
        }
        descriptor_ = -1;
    }

private:
    void sync_parent() const {
        const std::filesystem::path marker(path_);
        const std::filesystem::path parent = marker.has_parent_path()
            ? marker.parent_path() : std::filesystem::path(".");
        const int directory = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (directory < 0 || ::fsync(directory) != 0) {
            const int saved_errno = errno;
            if (directory >= 0) {
                (void)::close(directory);
            }
            throw std::runtime_error("cannot sync transaction marker directory: " +
                                     std::string(std::strerror(saved_errno)));
        }
        (void)::close(directory);
    }

    std::string path_;
    int descriptor_ = -1;
    dev_t device_ = 0;
    ino_t inode_ = 0;
};

std::uint32_t rotate_right(std::uint32_t value, unsigned int shift) {
    return (value >> shift) | (value << (32U - shift));
}

std::string sha256_bytes(const std::vector<std::uint8_t> & input) {
    static constexpr std::array<std::uint32_t, 64> constants = {
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
    std::array<std::uint32_t, 8> hash = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
    };
    if (input.size() > std::numeric_limits<std::uint64_t>::max() / 8U) {
        throw std::runtime_error("SHA-256 input is too large");
    }
    const std::uint64_t bit_length = static_cast<std::uint64_t>(input.size()) * 8U;
    std::vector<std::uint8_t> padded(input);
    padded.push_back(0x80U);
    while (padded.size() % 64U != 56U) padded.push_back(0U);
    for (int shift = 56; shift >= 0; shift -= 8) {
        padded.push_back(static_cast<std::uint8_t>(bit_length >> shift));
    }
    for (std::size_t offset = 0; offset < padded.size(); offset += 64U) {
        std::array<std::uint32_t, 64> words{};
        for (std::size_t index = 0; index < 16U; ++index) {
            const std::size_t cursor = offset + index * 4U;
            words[index] =
                (static_cast<std::uint32_t>(padded[cursor]) << 24U) |
                (static_cast<std::uint32_t>(padded[cursor + 1U]) << 16U) |
                (static_cast<std::uint32_t>(padded[cursor + 2U]) << 8U) |
                static_cast<std::uint32_t>(padded[cursor + 3U]);
        }
        for (std::size_t index = 16U; index < words.size(); ++index) {
            const std::uint32_t s0 = rotate_right(words[index - 15U], 7U) ^
                                     rotate_right(words[index - 15U], 18U) ^
                                     (words[index - 15U] >> 3U);
            const std::uint32_t s1 = rotate_right(words[index - 2U], 17U) ^
                                     rotate_right(words[index - 2U], 19U) ^
                                     (words[index - 2U] >> 10U);
            words[index] = words[index - 16U] + s0 + words[index - 7U] + s1;
        }
        std::uint32_t a = hash[0];
        std::uint32_t b = hash[1];
        std::uint32_t c = hash[2];
        std::uint32_t d = hash[3];
        std::uint32_t e = hash[4];
        std::uint32_t f = hash[5];
        std::uint32_t g = hash[6];
        std::uint32_t h = hash[7];
        for (std::size_t index = 0; index < words.size(); ++index) {
            const std::uint32_t s1 = rotate_right(e, 6U) ^ rotate_right(e, 11U) ^
                                     rotate_right(e, 25U);
            const std::uint32_t choice = (e & f) ^ ((~e) & g);
            const std::uint32_t temp1 = h + s1 + choice + constants[index] + words[index];
            const std::uint32_t s0 = rotate_right(a, 2U) ^ rotate_right(a, 13U) ^
                                     rotate_right(a, 22U);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temp2 = s0 + majority;
            h = g; g = f; f = e; e = d + temp1;
            d = c; c = b; b = a; a = temp1 + temp2;
        }
        hash[0] += a; hash[1] += b; hash[2] += c; hash[3] += d;
        hash[4] += e; hash[5] += f; hash[6] += g; hash[7] += h;
    }
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const std::uint32_t word : hash) output << std::setw(8) << word;
    return output.str();
}

std::vector<std::uint8_t> read_regular_file(const std::filesystem::path & path,
                                            std::size_t maximum_bytes) {
    const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (descriptor < 0) {
        throw std::runtime_error("cannot open intervention manifest: " +
                                 std::string(std::strerror(errno)));
    }
    struct stat status {};
    if (::fstat(descriptor, &status) != 0 || !S_ISREG(status.st_mode) ||
        status.st_size < 0 || static_cast<std::uint64_t>(status.st_size) > maximum_bytes) {
        const int saved_errno = errno;
        (void)::close(descriptor);
        throw std::runtime_error("intervention manifest must be a bounded regular file: " +
                                 std::string(std::strerror(saved_errno)));
    }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(status.st_size));
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const ssize_t count = ::read(descriptor, bytes.data() + offset, bytes.size() - offset);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            const int saved_errno = errno;
            (void)::close(descriptor);
            throw std::runtime_error("short read from intervention manifest: " +
                                     std::string(std::strerror(saved_errno)));
        }
        offset += static_cast<std::size_t>(count);
    }
    if (::close(descriptor) != 0) {
        throw std::runtime_error("cannot close intervention manifest");
    }
    return bytes;
}

std::vector<std::uint8_t> packed_f32_le(const std::vector<float> & values) {
    static_assert(sizeof(float) == 4, "intervention manifest requires IEEE-754 F32");
    std::vector<std::uint8_t> bytes(values.size() * sizeof(float));
    for (std::size_t index = 0; index < values.size(); ++index) {
        std::uint32_t word = 0;
        std::memcpy(&word, &values[index], sizeof(word));
        bytes[index * 4U] = static_cast<std::uint8_t>(word);
        bytes[index * 4U + 1U] = static_cast<std::uint8_t>(word >> 8U);
        bytes[index * 4U + 2U] = static_cast<std::uint8_t>(word >> 16U);
        bytes[index * 4U + 3U] = static_cast<std::uint8_t>(word >> 24U);
    }
    return bytes;
}

std::string target_names_digest(std::vector<std::string> names) {
    std::sort(names.begin(), names.end());
    std::vector<std::uint8_t> bytes;
    for (std::size_t index = 0; index < names.size(); ++index) {
        const std::string & name = names[index];
        if (index != 0U) bytes.push_back(static_cast<std::uint8_t>('\n'));
        bytes.insert(bytes.end(), name.begin(), name.end());
    }
    return sha256_bytes(bytes);
}

InterventionPlan load_intervention_manifest(const std::filesystem::path & path) {
    constexpr std::size_t maximum_manifest_bytes = 32U * 1024U * 1024U;
    const std::vector<std::uint8_t> bytes = read_regular_file(path, maximum_manifest_bytes);
    nlohmann::json root;
    try {
        root = nlohmann::json::parse(bytes.begin(), bytes.end());
    } catch (const nlohmann::json::exception & error) {
        throw std::runtime_error("invalid intervention manifest JSON: " +
                                 std::string(error.what()));
    }
    auto required = [&](const char * key) -> const nlohmann::json & {
        if (!root.contains(key)) {
            throw std::runtime_error("intervention manifest is missing " + std::string(key));
        }
        return root.at(key);
    };
    if (!root.is_object() || required("schema_version") != 1 ||
        required("kind") != "directional_ablation" ||
        required("status") != "complete" ||
        required("weight_intervention") != true || required("prompt_only") != false ||
        required("application_stage") != "pre_quantization_encoding" ||
        !required("source").is_object() || !required("tooling").is_object() ||
        !required("corpora").is_array()) {
        throw std::runtime_error("intervention manifest identity/provenance contract is invalid");
    }

    InterventionPlan plan;
    plan.manifest_sha256 = sha256_bytes(bytes);
    const nlohmann::json & directions = required("directions");
    if (!directions.is_array() || directions.empty() || directions.size() > 128U) {
        throw std::runtime_error("intervention manifest directions must be a nonempty bounded array");
    }
    std::unordered_map<std::string, std::size_t> direction_by_id;
    for (const nlohmann::json & entry : directions) {
        if (!entry.is_object() || !entry.contains("id") || !entry.at("id").is_string() ||
            !entry.contains("dtype") || entry.at("dtype") != "F32" ||
            !entry.contains("values") || !entry.at("values").is_array() ||
            !entry.contains("sha256") || !entry.at("sha256").is_string()) {
            throw std::runtime_error("intervention direction contract is invalid");
        }
        InterventionDirection direction;
        direction.id = entry.at("id").get<std::string>();
        if (direction.id.empty() || direction.id.size() > 128U ||
            direction_by_id.count(direction.id) != 0U) {
            throw std::runtime_error("intervention direction id is empty, duplicate, or too long");
        }
        const nlohmann::json & values = entry.at("values");
        if (values.empty() || values.size() > 16384U) {
            throw std::runtime_error("intervention direction has an invalid length");
        }
        direction.values.reserve(values.size());
        double squared_norm = 0.0;
        for (const nlohmann::json & value : values) {
            if (!value.is_number()) {
                throw std::runtime_error("intervention direction contains a non-number");
            }
            const double parsed = value.get<double>();
            if (!std::isfinite(parsed) || parsed < -1.0e6 || parsed > 1.0e6) {
                throw std::runtime_error("intervention direction contains a non-finite or extreme value");
            }
            const float narrowed = static_cast<float>(parsed);
            if (!std::isfinite(narrowed)) {
                throw std::runtime_error("intervention direction cannot be represented as F32");
            }
            direction.values.push_back(narrowed);
            squared_norm += static_cast<double>(narrowed) * static_cast<double>(narrowed);
        }
        if (std::abs(std::sqrt(squared_norm) - 1.0) > 1.0e-4) {
            throw std::runtime_error("intervention direction must have unit L2 norm");
        }
        const std::string expected_sha = entry.at("sha256").get<std::string>();
        if (expected_sha.size() != 64U || sha256_bytes(packed_f32_le(direction.values)) != expected_sha) {
            throw std::runtime_error("intervention direction packed-F32 SHA-256 mismatch");
        }
        direction_by_id.emplace(direction.id, plan.directions.size());
        plan.directions.push_back(std::move(direction));
    }

    const nlohmann::json & targets = required("targets");
    if (!targets.is_array() || targets.empty() || targets.size() > 96U) {
        throw std::runtime_error("intervention targets must be a nonempty bounded array");
    }
    const std::regex allowed_name(
        R"(^blk\.([0-9]|[1-3][0-9]|4[0-7])\.(attn_output|ssm_out)\.weight$)",
        std::regex::ECMAScript);
    std::vector<std::string> names;
    for (const nlohmann::json & entry : targets) {
        if (!entry.is_object() || !entry.contains("tensor_name") ||
            !entry.at("tensor_name").is_string() || !entry.contains("direction_id") ||
            !entry.at("direction_id").is_string() || !entry.contains("scale") ||
            !entry.at("scale").is_number() || !entry.contains("normalization") ||
            entry.at("normalization") != "row_norm_preserve" ||
            !entry.contains("expected_shape") || !entry.at("expected_shape").is_array() ||
            entry.at("expected_shape").size() != 2U) {
            throw std::runtime_error("intervention target contract is invalid");
        }
        InterventionTarget target;
        target.tensor_name = entry.at("tensor_name").get<std::string>();
        std::smatch name_match;
        if (!std::regex_match(target.tensor_name, name_match, allowed_name) ||
            plan.target_by_name.count(target.tensor_name) != 0U) {
            throw std::runtime_error("intervention target is unsafe or duplicated: " + target.tensor_name);
        }
        const unsigned long layer = std::stoul(name_match[1].str());
        const bool is_qsa_layer = layer % 4UL == 3UL;
        const std::string projection = name_match[2].str();
        if ((is_qsa_layer && projection != "attn_output") ||
            (!is_qsa_layer && projection != "ssm_out")) {
            throw std::runtime_error(
                "intervention target does not match the hybrid layer map: " +
                target.tensor_name);
        }
        const std::string direction_id = entry.at("direction_id").get<std::string>();
        const auto direction = direction_by_id.find(direction_id);
        if (direction == direction_by_id.end()) {
            throw std::runtime_error("intervention target names an unknown direction: " + direction_id);
        }
        target.direction_index = direction->second;
        const double scale = entry.at("scale").get<double>();
        if (!std::isfinite(scale) || scale == 0.0 || std::abs(scale) > 16.0) {
            throw std::runtime_error("intervention target scale is invalid");
        }
        target.scale = static_cast<float>(scale);
        target.preserve_row_norm = true;
        const nlohmann::json & shape = entry.at("expected_shape");
        if (!shape[0].is_number_unsigned() || !shape[1].is_number_unsigned()) {
            throw std::runtime_error("intervention expected_shape must contain unsigned integers");
        }
        const std::uint64_t columns = shape[0].get<std::uint64_t>();
        const std::uint64_t rows = shape[1].get<std::uint64_t>();
        if (columns == 0U || rows == 0U || columns > std::numeric_limits<std::size_t>::max() ||
            rows > std::numeric_limits<std::size_t>::max() ||
            plan.directions[target.direction_index].values.size() != rows) {
            throw std::runtime_error("intervention target shape/direction length is invalid");
        }
        target.expected_columns = static_cast<std::size_t>(columns);
        target.expected_rows = static_cast<std::size_t>(rows);
        plan.target_by_name.emplace(target.tensor_name, plan.targets.size());
        names.push_back(target.tensor_name);
        plan.targets.push_back(std::move(target));
    }
    const nlohmann::json & tensor_map = required("tensor_map");
    if (!tensor_map.is_object() || tensor_map.value("kind", "") != "exact_tensor_names" ||
        !tensor_map.contains("target_count") ||
        tensor_map.at("target_count").get<std::size_t>() != plan.targets.size() ||
        !tensor_map.contains("target_names_sha256") ||
        !tensor_map.at("target_names_sha256").is_string()) {
        throw std::runtime_error("intervention tensor_map contract is invalid");
    }
    plan.target_names_sha256 = target_names_digest(names);
    if (tensor_map.at("target_names_sha256").get<std::string>() !=
        plan.target_names_sha256) {
        throw std::runtime_error("intervention target-name SHA-256 mismatch");
    }
    return plan;
}

void usage(const char * executable) {
    std::fprintf(stderr,
                 "usage: %s --build-info-json\n"
                 "       %s --tensor-type REGEX={Q4_0_ROCMI4,Q6_K,Q4_0_ROCMFP4_FAST,Q3_0_ROCMFPX} "
                 "[--intervention-manifest JSON] "
                 "[--keep-split] [--dry-size-json] "
                 "[--device-budget-bytes N --runtime-reserve-bytes N] "
                 "INPUT OUTPUT Q4_0_ROCMI4 THREADS\n",
                 executable, executable);
}

void seek_input(FILE * file, std::size_t offset) {
    if (offset > static_cast<std::size_t>(std::numeric_limits<off_t>::max()) ||
        ::fseeko(file, static_cast<off_t>(offset), SEEK_SET) != 0) {
        throw std::runtime_error("input seek failed");
    }
}

void read_exact(FILE * file, void * destination, std::size_t size) {
    if (size != 0 && std::fread(destination, 1, size, file) != size) {
        throw std::runtime_error("short read from input GGUF");
    }
}

void write_exact(FILE * file, const void * source, std::size_t size) {
    if (size != 0 && std::fwrite(source, 1, size, file) != size) {
        throw std::runtime_error("short write to output GGUF");
    }
}

void write_zeros(FILE * file, std::size_t size) {
    static const std::vector<std::uint8_t> zeros(4096, 0);
    while (size != 0) {
        const std::size_t chunk = std::min(size, zeros.size());
        write_exact(file, zeros.data(), chunk);
        size -= chunk;
    }
}

std::size_t checked_product(std::size_t left, std::size_t right,
                            const char * description) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::runtime_error(std::string("size overflow for ") + description);
    }
    return left * right;
}

std::size_t checked_sum(std::size_t left, std::size_t right,
                        const char * description) {
    if (left > std::numeric_limits<std::size_t>::max() - right) {
        throw std::runtime_error(std::string("size overflow for ") + description);
    }
    return left + right;
}

std::size_t checked_aligned_size(std::size_t size, const char * description) {
    return checked_sum(size, kAlignment - 1, description) & ~(kAlignment - 1);
}

SplitInfo split_info(const gguf_context * context) {
    const std::int64_t number_key = gguf_find_key(context, "split.no");
    const std::int64_t count_key = gguf_find_key(context, "split.count");
    const std::int64_t tensors_key = gguf_find_key(context, "split.tensors.count");
    if (number_key < 0 && count_key < 0 && tensors_key < 0) {
        return {};
    }
    if (number_key < 0 || count_key < 0 || tensors_key < 0) {
        throw std::runtime_error("incomplete GGUF split metadata");
    }
    if (gguf_get_kv_type(context, number_key) != GGUF_TYPE_UINT16 ||
        gguf_get_kv_type(context, count_key) != GGUF_TYPE_UINT16 ||
        gguf_get_kv_type(context, tensors_key) != GGUF_TYPE_INT32) {
        throw std::runtime_error(
            "split.no/count must be UINT16 and split.tensors.count must be INT32");
    }
    SplitInfo result;
    result.number = gguf_get_val_u16(context, number_key);
    result.count = gguf_get_val_u16(context, count_key);
    result.tensor_count = gguf_get_val_i32(context, tensors_key);
    if (result.count == 0 || result.number >= result.count || result.tensor_count < 0) {
        throw std::runtime_error("invalid GGUF split metadata values");
    }
    return result;
}

struct PathSet {
    std::vector<std::filesystem::path> inputs;
    std::vector<std::filesystem::path> outputs;
};

std::string shard_filename(const std::string & stem, std::size_t number,
                           std::size_t count) {
    std::ostringstream name;
    name << stem << '-' << std::setfill('0') << std::setw(5) << number
         << "-of-" << std::setw(5) << count << ".gguf";
    return name.str();
}

PathSet discover_paths(const quant::Options & options) {
    static const std::regex split_name(
        R"(^(.+)-([0-9]{5})-of-([0-9]{5})\.gguf$)", std::regex::ECMAScript);
    const std::filesystem::path input(options.input_path);
    const std::filesystem::path output(options.output_path);
    std::smatch input_match;
    const std::string input_name = input.filename().string();
    if (!std::regex_match(input_name, input_match, split_name)) {
        return {{input}, {output}};
    }
    const unsigned long input_number = std::stoul(input_match[2].str());
    const unsigned long count = std::stoul(input_match[3].str());
    if (input_number != 1 || count < 2 || count > 99999) {
        throw std::runtime_error(
            "split input must be the first -00001-of-N.gguf shard with N >= 2");
    }
    if (!options.keep_split) {
        throw std::runtime_error("multi-shard input requires --keep-split");
    }

    std::string output_stem;
    std::smatch output_match;
    const std::string output_name = output.filename().string();
    if (std::regex_match(output_name, output_match, split_name)) {
        if (std::stoul(output_match[2].str()) != 1 ||
            std::stoul(output_match[3].str()) != count) {
            throw std::runtime_error(
                "split output filename must be the first shard with the same count");
        }
        output_stem = output_match[1].str();
    } else {
        if (output.extension() != ".gguf") {
            throw std::runtime_error("split output base must end in .gguf");
        }
        output_stem = output.stem().string();
    }

    PathSet result;
    result.inputs.reserve(count);
    result.outputs.reserve(count);
    const std::string input_stem = input_match[1].str();
    for (unsigned long index = 1; index <= count; ++index) {
        result.inputs.push_back(input.parent_path() /
                                shard_filename(input_stem, index, count));
        result.outputs.push_back(output.parent_path() /
                                 shard_filename(output_stem, index, count));
    }
    return result;
}

std::size_t matching_override(const std::vector<std::regex> & patterns,
                              std::vector<bool> & matched,
                              const std::string & name) {
    std::size_t result = patterns.size();
    for (std::size_t index = 0; index < patterns.size(); ++index) {
        if (std::regex_search(name, patterns[index])) {
            if (result != patterns.size()) {
                throw std::runtime_error(
                    "multiple --tensor-type regexes match tensor: " + name);
            }
            matched[index] = true;
            result = index;
        }
    }
    return result;
}

enum ggml_type ggml_type_for_format(quant::TensorFormat format) {
    switch (format) {
        case quant::TensorFormat::rocmi4: return GGML_TYPE_Q4_0_ROCMI4;
        case quant::TensorFormat::q6_k: return GGML_TYPE_Q6_K;
        case quant::TensorFormat::rocmfp4_fast:
            return GGML_TYPE_Q4_0_ROCMFP4_FAST;
        case quant::TensorFormat::rocmfpx_fp3:
            return GGML_TYPE_Q3_0_ROCMFPX;
    }
    throw std::runtime_error("unsupported tensor format");
}

template <typename Function>
void parallel_rows(std::size_t rows, std::size_t requested_threads,
                   Function function) {
    const std::size_t workers = std::min(rows, requested_threads);
    if (workers <= 1) {
        function(0, rows);
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(workers);
    std::exception_ptr failure;
    std::mutex failure_mutex;
    for (std::size_t worker = 0; worker < workers; ++worker) {
        const std::size_t begin = rows * worker / workers;
        const std::size_t end = rows * (worker + 1) / workers;
        threads.emplace_back([&, begin, end]() {
            try {
                function(begin, end - begin);
            } catch (...) {
                std::lock_guard<std::mutex> guard(failure_mutex);
                if (!failure) {
                    failure = std::current_exception();
                }
            }
        });
    }
    for (std::thread & thread : threads) {
        thread.join();
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
}

void convert_rows_to_float(enum ggml_type source_type,
                           const std::uint8_t * source,
                           float * destination,
                           std::size_t rows,
                           std::size_t columns,
                           std::size_t source_row_size) {
    if (columns > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::runtime_error("tensor row exceeds ggml conversion range");
    }
    if (source_type == GGML_TYPE_F32) {
        std::memcpy(destination, source,
                    checked_product(checked_product(rows, columns, "F32 rows"),
                                    sizeof(float), "F32 bytes"));
        return;
    }
    const ggml_type_traits * traits = ggml_get_type_traits(source_type);
    if (!traits || !traits->to_float) {
        throw std::runtime_error("source tensor type cannot convert to F32");
    }
    for (std::size_t row = 0; row < rows; ++row) {
        traits->to_float(source + row * source_row_size,
                         destination + row * columns,
                         static_cast<std::int64_t>(columns));
    }
}

void copy_tensor(FILE * input, FILE * output, const TensorPlan & plan) {
    std::vector<std::uint8_t> buffer(4U * 1024U * 1024U);
    seek_input(input, plan.source_offset);
    std::size_t remaining = plan.source_size;
    while (remaining != 0) {
        const std::size_t chunk = std::min(remaining, buffer.size());
        read_exact(input, buffer.data(), chunk);
        write_exact(output, buffer.data(), chunk);
        remaining -= chunk;
    }
}

void quantize_tensor(FILE * input, FILE * output, const TensorPlan & plan,
                     std::size_t threads,
                     quant::InterventionMetric * intervention_metric) {
    const enum ggml_type source_type = plan.source->type;
    if (source_type != GGML_TYPE_F32 && source_type != GGML_TYPE_F16 &&
        source_type != GGML_TYPE_BF16) {
        throw std::runtime_error("selected tensor " + plan.name +
                                 " must be F32, F16, or BF16");
    }
    const std::int64_t columns_i64 = plan.source->ne[0];
    const std::int64_t rows_i64 = ggml_nrows(plan.source);
    if (columns_i64 <= 0 || rows_i64 <= 0) {
        throw std::runtime_error("selected tensor has an empty dimension: " + plan.name);
    }
    const std::size_t columns = static_cast<std::size_t>(columns_i64);
    const std::size_t total_rows = static_cast<std::size_t>(rows_i64);
    const std::size_t source_row_size = ggml_row_size(source_type, columns_i64);
    const std::size_t output_row_size = ggml_row_size(plan.output_type, columns_i64);
    const std::size_t float_row_size = checked_product(columns, sizeof(float), "F32 row");
    const std::size_t combined_row_size =
        source_row_size + output_row_size + float_row_size;
    const std::size_t chunk_rows = std::max<std::size_t>(
        1, kWorkingSetBytes / std::max<std::size_t>(1, combined_row_size));

    std::vector<double> projection;
    std::vector<double> stored_projection;
    double source_frobenius_squared = 0.0;
    double stored_delta_squared = 0.0;
    double row_norm_relative_squared = 0.0;
    double row_norm_relative_max = 0.0;
    std::size_t row_norm_count = 0;
    if (plan.intervention) {
        if (!plan.intervention_direction ||
            plan.intervention_direction->values.size() != total_rows) {
            throw std::runtime_error("intervention direction/target mismatch: " + plan.name);
        }
        projection.assign(columns, 0.0);
        std::size_t projection_row = 0;
        while (projection_row < total_rows) {
            const std::size_t count = std::min(chunk_rows, total_rows - projection_row);
            std::vector<std::uint8_t> source(
                checked_product(count, source_row_size, "intervention source chunk"));
            std::vector<float> converted(
                checked_product(count, columns, "intervention F32 chunk"));
            seek_input(input, checked_sum(
                plan.source_offset,
                checked_product(projection_row, source_row_size,
                                "intervention source row offset"),
                "absolute intervention source row offset"));
            read_exact(input, source.data(), source.size());
            parallel_rows(count, threads, [&](std::size_t begin, std::size_t worker_rows) {
                convert_rows_to_float(source_type,
                                      source.data() + begin * source_row_size,
                                      converted.data() + begin * columns,
                                      worker_rows, columns, source_row_size);
            });
            // Deliberately reduce in a fixed row-major order. The direction
            // projection is tiny relative to the weight conversion, and a
            // deterministic double accumulator is more valuable than a racy
            // or thread-count-dependent reduction here.
            for (std::size_t local_row = 0; local_row < count; ++local_row) {
                const double coefficient = static_cast<double>(
                    plan.intervention_direction->values[projection_row + local_row]);
                const float * values = converted.data() + local_row * columns;
                for (std::size_t column = 0; column < columns; ++column) {
                    projection[column] += coefficient * static_cast<double>(values[column]);
                }
            }
            projection_row += count;
        }
        stored_projection.assign(columns, 0.0);
    }

    std::size_t row = 0;
    while (row < total_rows) {
        const std::size_t count = std::min(chunk_rows, total_rows - row);
        std::vector<std::uint8_t> source(
            checked_product(count, source_row_size, "source chunk"));
        std::vector<float> converted(checked_product(count, columns, "F32 chunk"));
        std::vector<std::uint8_t> quantized(
            checked_product(count, output_row_size, "quantized chunk"));

        seek_input(input, checked_sum(
            plan.source_offset,
            checked_product(row, source_row_size, "source row offset"),
            "absolute source row offset"));
        read_exact(input, source.data(), source.size());
        parallel_rows(count, threads, [&](std::size_t begin, std::size_t worker_rows) {
            convert_rows_to_float(source_type,
                                  source.data() + begin * source_row_size,
                                  converted.data() + begin * columns,
                                  worker_rows, columns, source_row_size);
            if (plan.intervention) {
                for (std::size_t local = 0; local < worker_rows; ++local) {
                    float * values = converted.data() + (begin + local) * columns;
                    const std::size_t absolute_row = row + begin + local;
                    const double coefficient =
                        static_cast<double>(plan.intervention->scale) *
                        static_cast<double>(
                            plan.intervention_direction->values[absolute_row]);
                    double original_squared = 0.0;
                    double modified_squared = 0.0;
                    for (std::size_t column = 0; column < columns; ++column) {
                        const double original = static_cast<double>(values[column]);
                        const double modified = original - coefficient * projection[column];
                        original_squared += original * original;
                        modified_squared += modified * modified;
                        values[column] = static_cast<float>(modified);
                    }
                    if (plan.intervention->preserve_row_norm &&
                        original_squared > 0.0) {
                        if (modified_squared <= original_squared * 1.0e-24) {
                            throw std::runtime_error(
                                "intervention produced a zero or unstable row norm: " +
                                plan.name);
                        }
                        const float norm_scale = static_cast<float>(
                            std::sqrt(original_squared / modified_squared));
                        for (std::size_t column = 0; column < columns; ++column) {
                            values[column] *= norm_scale;
                        }
                    }
                }
            }
            if (plan.output_type == GGML_TYPE_Q4_0_ROCMI4) {
                const std::size_t written = ggml_quantize_chunk(
                    plan.output_type, converted.data() + begin * columns,
                    quantized.data() + begin * output_row_size,
                    0, static_cast<std::int64_t>(worker_rows), columns_i64, nullptr);
                if (written != worker_rows * output_row_size) {
                    throw std::runtime_error("ggml returned an unexpected ROCMI4 chunk size");
                }
            } else {
                const ggml_type_traits * traits = ggml_get_type_traits(plan.output_type);
                if (!traits || !traits->from_float_ref) {
                    throw std::runtime_error("selected format has no reference encoder: " +
                                             std::string(ggml_type_name(plan.output_type)));
                }
                for (std::size_t local = 0; local < worker_rows; ++local) {
                    traits->from_float_ref(
                        converted.data() + (begin + local) * columns,
                        quantized.data() + (begin + local) * output_row_size,
                        columns_i64);
                }
            }
        });
        if (plan.intervention) {
            const ggml_type_traits * stored_traits =
                ggml_get_type_traits(plan.output_type);
            if (!stored_traits || !stored_traits->to_float) {
                throw std::runtime_error(
                    "selected format has no dequantizer for intervention audit: " +
                    std::string(ggml_type_name(plan.output_type)));
            }
            std::vector<float> original(columns);
            std::vector<float> stored(columns);
            for (std::size_t local_row = 0; local_row < count; ++local_row) {
                convert_rows_to_float(
                    source_type, source.data() + local_row * source_row_size,
                    original.data(), 1, columns, source_row_size);
                stored_traits->to_float(
                    quantized.data() + local_row * output_row_size,
                    stored.data(), columns_i64);
                const double direction_value = static_cast<double>(
                    plan.intervention_direction->values[row + local_row]);
                double original_row_squared = 0.0;
                double stored_row_squared = 0.0;
                for (std::size_t column = 0; column < columns; ++column) {
                    const double source_value = static_cast<double>(original[column]);
                    const double stored_value = static_cast<double>(stored[column]);
                    const double difference = stored_value - source_value;
                    source_frobenius_squared += source_value * source_value;
                    stored_delta_squared += difference * difference;
                    original_row_squared += source_value * source_value;
                    stored_row_squared += stored_value * stored_value;
                    stored_projection[column] += direction_value * stored_value;
                }
                if (original_row_squared > 0.0) {
                    const double relative =
                        std::sqrt(stored_row_squared / original_row_squared) - 1.0;
                    row_norm_relative_squared += relative * relative;
                    row_norm_relative_max =
                        std::max(row_norm_relative_max, std::abs(relative));
                    ++row_norm_count;
                } else if (stored_row_squared != 0.0) {
                    throw std::runtime_error(
                        "quantization changed a zero intervention target row: " + plan.name);
                }
            }
        }
        write_exact(output, quantized.data(), quantized.size());
        row += count;
    }
    if (plan.intervention) {
        if (!intervention_metric || source_frobenius_squared <= 0.0 ||
            row_norm_count == 0U) {
            throw std::runtime_error("intervention audit could not measure target: " + plan.name);
        }
        double source_projection_squared = 0.0;
        double stored_projection_squared = 0.0;
        double signed_projection_numerator = 0.0;
        for (std::size_t column = 0; column < columns; ++column) {
            source_projection_squared += projection[column] * projection[column];
            stored_projection_squared +=
                stored_projection[column] * stored_projection[column];
            signed_projection_numerator +=
                stored_projection[column] * projection[column];
        }
        if (source_projection_squared <=
            source_frobenius_squared * 1.0e-30) {
            throw std::runtime_error(
                "intervention direction has no measurable source projection: " + plan.name);
        }
        intervention_metric->tensor_name = plan.name;
        intervention_metric->source_projection_l2 =
            std::sqrt(source_projection_squared);
        intervention_metric->stored_projection_l2 =
            std::sqrt(stored_projection_squared);
        intervention_metric->stored_projection_ratio =
            std::sqrt(stored_projection_squared / source_projection_squared);
        intervention_metric->signed_projection_coefficient =
            signed_projection_numerator / source_projection_squared;
        intervention_metric->relative_frobenius_delta =
            std::sqrt(stored_delta_squared / source_frobenius_squared);
        intervention_metric->row_norm_relative_rmse =
            std::sqrt(row_norm_relative_squared /
                      static_cast<double>(row_norm_count));
        intervention_metric->row_norm_relative_max = row_norm_relative_max;
    }
}

std::unique_ptr<ShardPlan> plan_shard(
        const std::filesystem::path & input_path,
        const std::filesystem::path & output_path,
        const quant::Options & options,
        const std::vector<std::regex> & patterns,
        const std::vector<quant::TensorFormat> & formats,
        std::vector<bool> & pattern_matched,
        const InterventionPlan * intervention,
        std::vector<bool> & intervention_matched) {
    auto shard = std::make_unique<ShardPlan>();
    shard->input_path = input_path;
    shard->output_path = output_path;
    shard->input = File(std::fopen(input_path.c_str(), "rb"), &std::fclose);
    if (!shard->input) {
        throw std::runtime_error("cannot open input GGUF shard " + input_path.string() +
                                 ": " + std::string(std::strerror(errno)));
    }
    struct stat input_status {};
    if (::fstat(::fileno(shard->input.get()), &input_status) != 0 ||
        input_status.st_size < 0) {
        throw std::runtime_error("cannot determine input GGUF shard size");
    }
    const std::uint64_t input_file_size =
        static_cast<std::uint64_t>(input_status.st_size);
    ggml_context * raw_meta = nullptr;
    const gguf_init_params init = {/*.no_alloc=*/true, /*.ctx=*/&raw_meta};
    shard->input_gguf = GgufContext(
        gguf_init_from_file_ptr(shard->input.get(), init), &gguf_free);
    shard->input_meta = GgmlContext(raw_meta, &ggml_free);
    if (!shard->input_gguf || !shard->input_meta) {
        throw std::runtime_error("input shard is not a readable GGUF file: " +
                                 input_path.string());
    }
    if (gguf_get_alignment(shard->input_gguf.get()) != GGUF_DEFAULT_ALIGNMENT) {
        throw std::runtime_error("non-default GGUF alignment is unsupported");
    }
    shard->split = split_info(shard->input_gguf.get());

    shard->output_gguf = GgufContext(gguf_init_empty(), &gguf_free);
    if (!shard->output_gguf) {
        throw std::runtime_error("cannot allocate output GGUF metadata");
    }
    gguf_set_kv(shard->output_gguf.get(), shard->input_gguf.get());
    gguf_set_val_u32(shard->output_gguf.get(), "general.quantization_version",
                     GGML_QNT_VERSION);
    gguf_set_val_u32(shard->output_gguf.get(), "general.file_type",
                     GGML_FTYPE_MOSTLY_Q4_0_ROCMI4);
    if (intervention) {
        gguf_set_val_str(shard->output_gguf.get(), "ember.intervention.kind",
                         "directional_ablation");
        gguf_set_val_str(shard->output_gguf.get(),
                         "ember.intervention.application_stage",
                         "pre_quantization_encoding");
        gguf_set_val_str(shard->output_gguf.get(),
                         "ember.intervention.manifest_sha256",
                         intervention->manifest_sha256.c_str());
        gguf_set_val_str(shard->output_gguf.get(),
                         "ember.intervention.target_names_sha256",
                         intervention->target_names_sha256.c_str());
        gguf_set_val_u32(shard->output_gguf.get(),
                         "ember.intervention.target_count",
                         static_cast<std::uint32_t>(intervention->targets.size()));
    } else {
        // Bootstrap controls are loadable inputs for activation-direction
        // extraction, but this label keeps them distinct from release-eligible
        // intervened artifacts. Packaging retains its manifest requirement.
        gguf_set_val_str(shard->output_gguf.get(), "ember.intervention.kind",
                         "none_control");
        gguf_set_val_str(shard->output_gguf.get(),
                         "ember.intervention.release_eligibility",
                         "control_only_requires_manifest_for_release");
    }
    if (!options.keep_split) {
        (void)gguf_remove_key(shard->output_gguf.get(), "split.no");
        (void)gguf_remove_key(shard->output_gguf.get(), "split.count");
        (void)gguf_remove_key(shard->output_gguf.get(), "split.tensors.count");
    }

    const std::int64_t tensor_count = gguf_get_n_tensors(shard->input_gguf.get());
    shard->tensors.reserve(static_cast<std::size_t>(tensor_count));
    for (std::int64_t index = 0; index < tensor_count; ++index) {
        const char * tensor_name = gguf_get_tensor_name(shard->input_gguf.get(), index);
        ggml_tensor * tensor = ggml_get_tensor(shard->input_meta.get(), tensor_name);
        if (!tensor) {
            throw std::runtime_error("GGUF tensor metadata is missing: " +
                                     std::string(tensor_name));
        }
        const std::size_t override_index =
            matching_override(patterns, pattern_matched, tensor_name);
        const bool manual = override_index != patterns.size();
        enum ggml_type output_type = manual
            ? ggml_type_for_format(formats[override_index])
            : GGML_TYPE_Q4_0_ROCMI4;
        bool selected = manual || quant::tensor_allows_quantization(
            tensor_name, static_cast<std::size_t>(ggml_n_dims(tensor)));
        const InterventionTarget * target = nullptr;
        const InterventionDirection * direction = nullptr;
        if (intervention) {
            const auto found = intervention->target_by_name.find(tensor_name);
            if (found != intervention->target_by_name.end()) {
                const std::size_t target_index = found->second;
                target = &intervention->targets[target_index];
                direction = &intervention->directions[target->direction_index];
                intervention_matched[target_index] = true;
                if (ggml_n_dims(tensor) != 2 || tensor->ne[0] <= 0 || tensor->ne[1] <= 0 ||
                    static_cast<std::size_t>(tensor->ne[0]) != target->expected_columns ||
                    static_cast<std::size_t>(tensor->ne[1]) != target->expected_rows) {
                    throw std::runtime_error(
                        "intervention target shape differs from manifest: " +
                        std::string(tensor_name));
                }
                if (!selected) {
                    throw std::runtime_error(
                        "intervention target is not selected for quantized encoding: " +
                        std::string(tensor_name));
                }
                if (output_type != GGML_TYPE_Q4_0_ROCMI4 &&
                    output_type != GGML_TYPE_Q4_0_ROCMFP4_FAST) {
                    throw std::runtime_error(
                        "intervention target must use Q4_0_ROCMI4 or "
                        "Q4_0_ROCMFP4_FAST encoding: " +
                        std::string(tensor_name));
                }
            }
        }
        const std::int64_t block_size = ggml_blck_size(output_type);
        if (selected && tensor->ne[0] % block_size != 0) {
            if (manual) {
                throw std::runtime_error("manual " +
                    std::string(ggml_type_name(output_type)) +
                    " tensor has an incompatible row width: " + tensor_name);
            }
            std::fprintf(stderr,
                         "warning: keeping %s because its row width is not divisible by %lld\n",
                         tensor_name, static_cast<long long>(block_size));
            selected = false;
        }
        if (selected && tensor->type != GGML_TYPE_F32 &&
            tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_BF16) {
            throw std::runtime_error("selected tensor " + std::string(tensor_name) +
                                     " must be F32, F16, or BF16");
        }
        gguf_add_tensor(shard->output_gguf.get(), tensor);
        if (selected) {
            gguf_set_tensor_type(shard->output_gguf.get(), tensor_name, output_type);
        }
        const std::size_t data_offset = gguf_get_data_offset(shard->input_gguf.get());
        const std::size_t tensor_offset =
            gguf_get_tensor_offset(shard->input_gguf.get(), index);
        const std::size_t source_offset =
            checked_sum(data_offset, tensor_offset, "input tensor offset");
        const std::size_t source_size =
            gguf_get_tensor_size(shard->input_gguf.get(), index);
        const std::size_t source_end =
            checked_sum(source_offset, source_size, "input tensor extent");
        if (static_cast<std::uint64_t>(source_end) > input_file_size) {
            throw std::runtime_error("input tensor exceeds GGUF file: " +
                                     std::string(tensor_name));
        }
        shard->tensors.push_back({
            tensor,
            tensor_name,
            source_offset,
            source_size,
            0,
            selected,
            manual,
            output_type,
            target,
            direction,
        });
    }
    for (std::int64_t index = 0; index < tensor_count; ++index) {
        shard->tensors[static_cast<std::size_t>(index)].output_size =
            gguf_get_tensor_size(shard->output_gguf.get(), index);
    }

    const std::size_t metadata_size = gguf_get_meta_size(shard->output_gguf.get());
    std::size_t data_size = 0;
    if (tensor_count > 0) {
        const std::size_t last = static_cast<std::size_t>(tensor_count - 1);
        const std::size_t last_offset =
            gguf_get_tensor_offset(shard->output_gguf.get(), tensor_count - 1);
        const std::size_t padded_last = checked_aligned_size(
            shard->tensors[last].output_size, "last output tensor alignment");
        data_size = checked_sum(last_offset, padded_last, "output GGUF data size");
    }
    if (metadata_size > std::numeric_limits<std::size_t>::max() - data_size) {
        throw std::runtime_error("output GGUF artifact size overflows size_t");
    }
    shard->output_size = static_cast<std::uint64_t>(metadata_size + data_size);
    return shard;
}

void write_shard(const ShardPlan & shard, OwnedPartial & partial,
                 std::size_t threads,
                 std::vector<quant::InterventionMetric> & intervention_metrics) {
    if (!gguf_write_to_file_ptr(shard.output_gguf.get(), partial.file(), true)) {
        throw std::runtime_error("failed to write output GGUF metadata");
    }
    std::size_t data_written = 0;
    for (std::size_t index = 0; index < shard.tensors.size(); ++index) {
        const TensorPlan & plan = shard.tensors[index];
        const std::size_t expected_offset =
            gguf_get_tensor_offset(shard.output_gguf.get(),
                                   static_cast<std::int64_t>(index));
        if (data_written > expected_offset) {
            throw std::runtime_error("output tensor offsets overlap");
        }
        write_zeros(partial.file(), expected_offset - data_written);
        data_written = expected_offset;
        std::fprintf(stderr, "%s: %s -> %s\n", plan.name.c_str(),
                     ggml_type_name(plan.source->type),
                     plan.quantize ? ggml_type_name(plan.output_type) :
                                     ggml_type_name(plan.source->type));
        if (plan.quantize) {
            quant::InterventionMetric metric;
            quant::InterventionMetric * metric_output =
                plan.intervention ? &metric : nullptr;
            quantize_tensor(shard.input.get(), partial.file(), plan, threads,
                            metric_output);
            if (metric_output) {
                intervention_metrics.push_back(std::move(metric));
            }
        } else {
            if (plan.output_size != plan.source_size) {
                throw std::runtime_error("copied tensor size changed unexpectedly: " + plan.name);
            }
            copy_tensor(shard.input.get(), partial.file(), plan);
        }
        data_written += plan.output_size;
        const std::size_t padded = checked_aligned_size(
            plan.output_size, "output tensor alignment");
        write_zeros(partial.file(), padded - plan.output_size);
        data_written += padded - plan.output_size;
    }
}

quant::SizeReport run_quantization(const quant::Options & options) {
    const PathSet paths = discover_paths(options);
    if (paths.inputs.size() != paths.outputs.size() || paths.inputs.empty()) {
        throw std::runtime_error("internal shard path discovery failure");
    }

    std::vector<std::regex> patterns;
    std::vector<quant::TensorFormat> formats;
    patterns.reserve(options.tensor_type_overrides.size());
    formats.reserve(options.tensor_type_overrides.size());
    try {
        for (const quant::TensorTypeOverride & override : options.tensor_type_overrides) {
            patterns.emplace_back(override.pattern, std::regex::ECMAScript);
            formats.push_back(override.format);
        }
    } catch (const std::regex_error & error) {
        throw std::runtime_error("invalid --tensor-type regex: " +
                                 std::string(error.what()));
    }
    std::vector<bool> pattern_matched(patterns.size(), false);
    std::unique_ptr<InterventionPlan> intervention;
    if (!options.intervention_manifest_path.empty()) {
        intervention = std::make_unique<InterventionPlan>(
            load_intervention_manifest(options.intervention_manifest_path));
    }
    std::vector<bool> intervention_matched(
        intervention ? intervention->targets.size() : 0U, false);
    std::vector<std::unique_ptr<ShardPlan>> shards;
    shards.reserve(paths.inputs.size());
    for (std::size_t index = 0; index < paths.inputs.size(); ++index) {
        const std::filesystem::path normalized_input =
            std::filesystem::absolute(paths.inputs[index]).lexically_normal();
        const std::filesystem::path normalized_output =
            std::filesystem::absolute(paths.outputs[index]).lexically_normal();
        if (normalized_input == normalized_output) {
            throw std::runtime_error("input and output shard paths must differ");
        }
        shards.push_back(plan_shard(paths.inputs[index], paths.outputs[index],
                                    options, patterns, formats, pattern_matched,
                                    intervention.get(), intervention_matched));
    }
    for (std::size_t index = 0; index < pattern_matched.size(); ++index) {
        if (!pattern_matched[index]) {
            throw std::runtime_error("--tensor-type regex matched no tensor in shard set: " +
                                     options.tensor_type_overrides[index].pattern);
        }
    }
    for (std::size_t index = 0; index < intervention_matched.size(); ++index) {
        if (!intervention_matched[index]) {
            throw std::runtime_error(
                "intervention target is absent from GGUF shard set: " +
                intervention->targets[index].tensor_name);
        }
    }

    std::set<std::string> tensor_names;
    std::size_t global_tensor_count = 0;
    const std::size_t shard_count = shards.size();
    std::int32_t declared_tensor_count = -1;
    for (std::size_t index = 0; index < shard_count; ++index) {
        const ShardPlan & shard = *shards[index];
        if (shard_count > 1) {
            if (shard.split.count != shard_count || shard.split.number != index) {
                throw std::runtime_error("GGUF split.no/count does not match ordered shard filenames");
            }
            if (declared_tensor_count < 0) {
                declared_tensor_count = shard.split.tensor_count;
            } else if (shard.split.tensor_count != declared_tensor_count) {
                throw std::runtime_error("inconsistent split.tensors.count across shards");
            }
        } else if (shard.split.count != 1) {
            throw std::runtime_error("split metadata requires a complete ordered shard set");
        }
        global_tensor_count = checked_sum(global_tensor_count, shard.tensors.size(),
                                          "global tensor count");
        for (const TensorPlan & tensor : shard.tensors) {
            if (!tensor_names.insert(tensor.name).second) {
                throw std::runtime_error("duplicate tensor across GGUF shards: " + tensor.name);
            }
        }
    }
    if (shard_count > 1 &&
        (declared_tensor_count < 0 ||
         static_cast<std::size_t>(declared_tensor_count) != global_tensor_count)) {
        throw std::runtime_error("split.tensors.count does not match global tensor inventory");
    }

    quant::SizeReport report;
    report.runtime_reserve_bytes = options.runtime_reserve_bytes;
    report.budget_bytes = options.device_budget_bytes;
    report.shard_bytes.reserve(shard_count);
    if (intervention) {
        report.intervention_manifest_sha256 = intervention->manifest_sha256;
        report.intervention_target_names_sha256 = intervention->target_names_sha256;
        report.intervention_targets.reserve(intervention->targets.size());
        for (const InterventionTarget & target : intervention->targets) {
            report.intervention_targets.push_back(target.tensor_name);
        }
        std::sort(report.intervention_targets.begin(), report.intervention_targets.end());
        report.intervention_validated = true;
    }
    for (const auto & shard : shards) {
        if (report.artifact_bytes > std::numeric_limits<std::uint64_t>::max() -
                                    shard->output_size) {
            throw std::runtime_error("aggregate output artifact size overflows UINT64");
        }
        report.artifact_bytes += shard->output_size;
        report.shard_bytes.push_back(shard->output_size);
    }
    const bool total_fits_u64 = report.artifact_bytes <=
        std::numeric_limits<std::uint64_t>::max() - report.runtime_reserve_bytes;
    report.fits = options.budget_configured && total_fits_u64 &&
                  report.artifact_bytes + report.runtime_reserve_bytes <=
                      report.budget_bytes;
    if (options.budget_configured && !report.fits && !options.dry_size_json) {
        throw std::runtime_error(
            "aggregate artifact plus configured native-262K runtime reserve exceeds device budget: " +
            quant::size_report_json(report));
    }
    if (options.dry_size_json) {
        return report;
    }

    // Validate every destination before creating any partial. Each partial is
    // fully synced before any final shard name is linked into place.
    for (const auto & shard : shards) {
        struct stat status {};
        if (::lstat(shard->output_path.c_str(), &status) == 0) {
            throw std::runtime_error("output already exists: " + shard->output_path.string());
        }
        if (errno != ENOENT) {
            throw std::runtime_error("cannot inspect output path: " +
                                     std::string(std::strerror(errno)));
        }
    }
    std::vector<std::unique_ptr<OwnedPartial>> partials;
    partials.reserve(shard_count);
    for (const auto & shard : shards) {
        partials.push_back(std::make_unique<OwnedPartial>(shard->output_path));
    }
    std::vector<quant::InterventionMetric> intervention_metrics;
    if (intervention) intervention_metrics.reserve(intervention->targets.size());
    for (std::size_t index = 0; index < shard_count; ++index) {
        write_shard(*shards[index], *partials[index], options.threads,
                    intervention_metrics);
        partials[index]->sync_and_close();
    }
    TransactionMarker transaction(options.output_path);
    try {
        for (std::size_t index = 0; index < shard_count; ++index) {
            partials[index]->promote_no_clobber(shards[index]->output_path);
        }
        transaction.commit();
    } catch (...) {
        std::fprintf(stderr,
                     "warning: incomplete shard transaction retained marker %s\n",
                     transaction.path().c_str());
        for (auto & partial : partials) {
            std::fprintf(stderr, "warning: retaining owned partial for recovery: %s\n",
                         partial->path().c_str());
            partial->retain_for_recovery();
        }
        throw;
    }
    ggml_quantize_free();
    report.intervention_applied = intervention != nullptr;
    std::sort(intervention_metrics.begin(), intervention_metrics.end(),
              [](const quant::InterventionMetric & left,
                 const quant::InterventionMetric & right) {
                  return left.tensor_name < right.tensor_name;
              });
    report.intervention_metrics = std::move(intervention_metrics);
    return report;
}

}  // namespace

int main(int argc, char ** argv) {
    quant::Options options;
    std::string error;
    if (!quant::parse_options(argc, argv, options, error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        usage(argv[0]);
        return 2;
    }
    if (options.build_info_json) {
        std::printf("%s\n", quant::build_info_json(EMBER_CONFIGURED_GIT_HEAD).c_str());
        return 0;
    }
    try {
        const quant::SizeReport report = run_quantization(options);
        if (options.dry_size_json || !options.intervention_manifest_path.empty()) {
            std::printf("%s\n", quant::size_report_json(report).c_str());
            if (options.dry_size_json) return report.fits ? 0 : 3;
        }
        return 0;
    } catch (const std::exception & exception) {
        std::fprintf(stderr, "error: %s\n", exception.what());
        ggml_quantize_free();
        return 1;
    }
}
