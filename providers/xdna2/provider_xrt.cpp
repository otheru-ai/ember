// XRT provider for Ember's experimental XDNA2 selected-expert ABI.
//
// This is intentionally a bring-up provider: weights are cached as persistent
// host-only BOs and synchronized only once, but each selected expert currently
// requires one gate, one up, and one down launch. RYZEN_AI_FINDINGS.md in
// otheru/vit-scout measured dispatch-heavy hybrid execution losing badly, so
// this provider remains opt-in until a fused selected-expert instruction stream
// and hardware measurements prove an end-to-end win.

#include "moe_expert_compute_xdna.h"
#include "rocmfp2_pack.h"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using ember::xdna2::pack_rocmfp2_gemv;
using ember::xdna2::rocmfp2_projection_bytes;
using ember::xdna2::rocmfp2_supported_shape;

void set_error(char * destination, size_t capacity, const std::string & message) {
    if (!destination || capacity == 0) return;
    std::snprintf(destination, capacity, "%s", message.c_str());
}

const char * nonempty_env(const char * name) {
    const char * value = std::getenv(name);
    return value && value[0] ? value : nullptr;
}

size_t cache_capacity_bytes() {
    constexpr size_t fallback_mb = 1024;
    const char * raw = nonempty_env("EMBER_XDNA_WEIGHT_CACHE_MB");
    if (!raw) return fallback_mb * 1024 * 1024;
    errno = 0;
    char * end = nullptr;
    const unsigned long long mb = std::strtoull(raw, &end, 10);
    if (errno == ERANGE || end == raw || *end != '\0' || mb < 16 ||
        mb > std::numeric_limits<size_t>::max() / (1024 * 1024)) {
        return fallback_mb * 1024 * 1024;
    }
    return static_cast<size_t>(mb) * 1024 * 1024;
}

unsigned device_index() {
    const char * raw = nonempty_env("EMBER_XDNA_DEVICE");
    if (!raw) return 0;
    errno = 0;
    char * end = nullptr;
    const unsigned long value = std::strtoul(raw, &end, 10);
    if (errno == ERANGE || end == raw || *end != '\0' ||
        value > std::numeric_limits<unsigned>::max()) {
        throw std::runtime_error("invalid EMBER_XDNA_DEVICE");
    }
    return static_cast<unsigned>(value);
}

std::string artifact_path(int k, int n, const char * suffix) {
    const char * directory = nonempty_env("EMBER_XDNA_ARTIFACT_DIR");
    std::string path = directory ? directory : "/usr/local/share/ember/xdna2";
    path += "/gemv_" + std::to_string(k) + "x" + std::to_string(n) + suffix;
    return path;
}

std::vector<uint32_t> read_instructions(const std::string & path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open instruction stream: " + path);
    const std::streamsize length = file.tellg();
    if (length <= 0 || length % static_cast<std::streamsize>(sizeof(uint32_t)) != 0) {
        throw std::runtime_error("invalid instruction stream: " + path);
    }
    file.seekg(0);
    std::vector<uint32_t> words(static_cast<size_t>(length) / sizeof(uint32_t));
    if (!file.read(reinterpret_cast<char *>(words.data()), length)) {
        throw std::runtime_error("short instruction stream read: " + path);
    }
    return words;
}

uint16_t float_to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
}

float bf16_to_float(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

class GemvProgram {
public:
    GemvProgram(xrt::device & device, int k, int n)
        : device_(device), k_(k), n_(n) {
        const std::string xclbin_path = artifact_path(k, n, ".xclbin");
        const std::string insts_path = artifact_path(k, n, ".insts");
        xclbin_ = std::make_unique<xrt::xclbin>(xclbin_path);
        const auto kernels = xclbin_->get_kernels();
        const auto found = std::find_if(kernels.begin(), kernels.end(),
            [](const xrt::xclbin::kernel & kernel) {
                return kernel.get_name().rfind("MLIR_AIE", 0) == 0;
            });
        if (found == kernels.end()) {
            throw std::runtime_error("MLIR_AIE kernel absent from " + xclbin_path);
        }
        device_.register_xclbin(*xclbin_);
        context_ = std::make_unique<xrt::hw_context>(device_, xclbin_->get_uuid());
        kernel_ = std::make_unique<xrt::kernel>(*context_, found->get_name());

        instructions_ = read_instructions(insts_path);
        instruction_bo_ = std::make_unique<xrt::bo>(
            device_, instructions_.size() * sizeof(uint32_t),
            XCL_BO_FLAGS_CACHEABLE, kernel_->group_id(1));
        std::memcpy(instruction_bo_->map<void *>(), instructions_.data(),
                    instructions_.size() * sizeof(uint32_t));
        instruction_bo_->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        input_bo_ = std::make_unique<xrt::bo>(
            device_, static_cast<size_t>(k_) * sizeof(uint16_t),
            XRT_BO_FLAGS_HOST_ONLY, kernel_->group_id(3));
        output_bo_ = std::make_unique<xrt::bo>(
            device_, static_cast<size_t>(n_) * sizeof(uint16_t),
            XRT_BO_FLAGS_HOST_ONLY, kernel_->group_id(5));
        scratch_bo_ = std::make_unique<xrt::bo>(
            device_, 1, XRT_BO_FLAGS_HOST_ONLY, kernel_->group_id(6));
        trace_bo_ = std::make_unique<xrt::bo>(
            device_, 1, XRT_BO_FLAGS_HOST_ONLY, kernel_->group_id(7));
    }

    std::unique_ptr<xrt::bo> make_weight_bo(const std::vector<uint8_t> & packed) {
        auto result = std::make_unique<xrt::bo>(
            device_, packed.size(), XRT_BO_FLAGS_HOST_ONLY, kernel_->group_id(4));
        std::memcpy(result->map<void *>(), packed.data(), packed.size());
        result->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        return result;
    }

    void run(const float * input, xrt::bo & weights, float * output) {
        uint16_t * input_bf16 = input_bo_->map<uint16_t *>();
        for (int i = 0; i < k_; ++i) input_bf16[i] = float_to_bf16(input[i]);
        input_bo_->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto invocation = (*kernel_)(
            3, *instruction_bo_, static_cast<uint32_t>(instructions_.size()),
            *input_bo_, weights, *output_bo_, *scratch_bo_, *trace_bo_);
        const ert_cmd_state state = invocation.wait();
        if (state != ERT_CMD_STATE_COMPLETED) {
            throw std::runtime_error("XDNA2 GEMV command did not complete");
        }
        output_bo_->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        const uint16_t * output_bf16 = output_bo_->map<uint16_t *>();
        for (int i = 0; i < n_; ++i) output[i] = bf16_to_float(output_bf16[i]);
    }

private:
    xrt::device & device_;
    int k_;
    int n_;
    std::unique_ptr<xrt::xclbin> xclbin_;
    std::unique_ptr<xrt::hw_context> context_;
    std::unique_ptr<xrt::kernel> kernel_;
    std::vector<uint32_t> instructions_;
    std::unique_ptr<xrt::bo> instruction_bo_;
    std::unique_ptr<xrt::bo> input_bo_;
    std::unique_ptr<xrt::bo> output_bo_;
    std::unique_ptr<xrt::bo> scratch_bo_;
    std::unique_ptr<xrt::bo> trace_bo_;
};

struct WeightKey {
    const void * source = nullptr;
    size_t bytes = 0;
    int k = 0;
    int n = 0;

    bool operator==(const WeightKey & other) const {
        return source == other.source && bytes == other.bytes &&
               k == other.k && n == other.n;
    }
};

struct WeightKeyHash {
    size_t operator()(const WeightKey & key) const {
        size_t hash = std::hash<const void *>{}(key.source);
        hash ^= std::hash<size_t>{}(key.bytes) + 0x9e3779b9u + (hash << 6) + (hash >> 2);
        hash ^= std::hash<int>{}(key.k) + 0x9e3779b9u + (hash << 6) + (hash >> 2);
        hash ^= std::hash<int>{}(key.n) + 0x9e3779b9u + (hash << 6) + (hash >> 2);
        return hash;
    }
};

struct CachedWeight {
    std::unique_ptr<xrt::bo> bo;
    size_t bytes = 0;
    uint64_t last_use = 0;
};

class Provider {
public:
    explicit Provider(const ember_xdna_moe_config_v1 & config)
        : config_(config), device_(device_index()),
          gate_up_(device_, config.n_embd, config.n_ff_exp),
          down_(device_, config.n_ff_exp, config.n_embd),
          capacity_(cache_capacity_bytes()) {
        if (config.n_embd <= 0 || config.n_ff_exp <= 0 ||
            !rocmfp2_supported_shape(config.n_embd, config.n_ff_exp) ||
            !rocmfp2_supported_shape(config.n_ff_exp, config.n_embd)) {
            throw std::runtime_error("model dimensions are unsupported by XDNA2 GEMV");
        }
    }

    int compute(const ember_xdna_moe_batch_v1 & batch, std::string * error) {
        std::lock_guard<std::mutex> guard(lock_);
        if (!healthy_.load()) {
            if (error) *error = "XDNA2 provider is unhealthy";
            return 0;
        }
        if (batch.n_tokens != 1 || batch.n_selected <= 0 || !batch.input ||
            !batch.router_weights || !batch.output || !batch.expert_weights ||
            batch.n_embd != config_.n_embd || batch.n_ff_exp != config_.n_ff_exp) {
            if (error) *error = "unsupported or invalid XDNA2 batch";
            return 0;
        }

        try {
            std::fill(batch.output, batch.output + batch.n_embd, 0.0f);
            std::vector<float> gate(static_cast<size_t>(batch.n_ff_exp));
            std::vector<float> up(static_cast<size_t>(batch.n_ff_exp));
            std::vector<float> hidden(static_cast<size_t>(batch.n_ff_exp));
            std::vector<float> projected(static_cast<size_t>(batch.n_embd));

            for (int slot = 0; slot < batch.n_selected; ++slot) {
                const auto & view = batch.expert_weights[slot];
                if (view.struct_size < sizeof(ember_xdna_moe_weight_view_v1) ||
                    view.fused_gate_up || !view.gate || !view.up || !view.down ||
                    view.gate_format != EMBER_XDNA_MOE_WEIGHT_ROCMFP2 ||
                    view.up_format != EMBER_XDNA_MOE_WEIGHT_ROCMFP2 ||
                    view.down_format != EMBER_XDNA_MOE_WEIGHT_ROCMFP2) {
                    if (error) *error = "selected expert is not separate ROCMFP2";
                    return 0;
                }

                auto gate_weight = weight(view.gate, view.gate_bytes,
                                          batch.n_embd, batch.n_ff_exp, gate_up_);
                auto up_weight = weight(view.up, view.up_bytes,
                                        batch.n_embd, batch.n_ff_exp, gate_up_);
                auto down_weight = weight(view.down, view.down_bytes,
                                          batch.n_ff_exp, batch.n_embd, down_);
                gate_up_.run(batch.input, *gate_weight->bo, gate.data());
                gate_up_.run(batch.input, *up_weight->bo, up.data());

                for (int i = 0; i < batch.n_ff_exp; ++i) {
                    float gate_value = gate[(size_t)i] * view.gate_scale;
                    float up_value = up[(size_t)i] * view.up_scale;
                    if (config_.swiglu_clamp > 1.0e-6f) {
                        gate_value = std::min(gate_value, config_.swiglu_clamp);
                        up_value = std::max(-config_.swiglu_clamp,
                                            std::min(up_value, config_.swiglu_clamp));
                    }
                    hidden[(size_t)i] =
                        (gate_value / (1.0f + std::exp(-gate_value))) * up_value;
                }

                down_.run(hidden.data(), *down_weight->bo, projected.data());
                const float scale = view.down_scale * batch.router_weights[slot];
                for (int i = 0; i < batch.n_embd; ++i) {
                    batch.output[i] += projected[(size_t)i] * scale;
                }
                ++experts_;
            }
            ++calls_;
            return 1;
        } catch (const std::exception & exception) {
            healthy_.store(false);
            if (error) *error = exception.what();
            return 0;
        }
    }

    bool healthy() const { return healthy_.load(); }

    ~Provider() {
        if (nonempty_env("DFLASH_MOE_XDNA_TRACE")) {
            std::fprintf(stderr,
                "[xdna2-xrt] calls=%llu experts=%llu cache_hits=%llu "
                "cache_misses=%llu cache_bytes=%zu\n",
                static_cast<unsigned long long>(calls_),
                static_cast<unsigned long long>(experts_),
                static_cast<unsigned long long>(cache_hits_),
                static_cast<unsigned long long>(cache_misses_), cache_bytes_);
        }
    }

private:
    std::shared_ptr<CachedWeight> weight(const void * raw, size_t raw_bytes,
                                         int k, int n, GemvProgram & program) {
        const size_t required = rocmfp2_projection_bytes(k, n);
        if (!raw || required == 0 || raw_bytes < required) {
            throw std::runtime_error("invalid ROCMFP2 projection view");
        }
        const WeightKey key{raw, raw_bytes, k, n};
        const auto found = cache_.find(key);
        if (found != cache_.end()) {
            found->second->last_use = ++clock_;
            ++cache_hits_;
            return found->second;
        }

        std::vector<uint8_t> packed;
        std::string pack_error;
        if (!pack_rocmfp2_gemv(raw, raw_bytes, k, n, packed, &pack_error)) {
            throw std::runtime_error(pack_error);
        }
        if (packed.size() > capacity_) {
            throw std::runtime_error("one expert projection exceeds XDNA2 cache");
        }
        while (cache_bytes_ + packed.size() > capacity_ && !cache_.empty()) {
            const auto victim = std::min_element(cache_.begin(), cache_.end(),
                [](const auto & left, const auto & right) {
                    return left.second->last_use < right.second->last_use;
                });
            cache_bytes_ -= victim->second->bytes;
            cache_.erase(victim);
        }

        auto entry = std::make_shared<CachedWeight>();
        entry->bo = program.make_weight_bo(packed);
        entry->bytes = packed.size();
        entry->last_use = ++clock_;
        cache_bytes_ += entry->bytes;
        cache_.emplace(key, entry);
        ++cache_misses_;
        return entry;
    }

    ember_xdna_moe_config_v1 config_{};
    xrt::device device_;
    GemvProgram gate_up_;
    GemvProgram down_;
    const size_t capacity_;
    size_t cache_bytes_ = 0;
    uint64_t clock_ = 0;
    uint64_t calls_ = 0;
    uint64_t experts_ = 0;
    uint64_t cache_hits_ = 0;
    uint64_t cache_misses_ = 0;
    std::atomic<bool> healthy_{true};
    std::mutex lock_;
    std::unordered_map<WeightKey, std::shared_ptr<CachedWeight>, WeightKeyHash> cache_;
};

void * create_provider(const ember_xdna_moe_config_v1 * config,
                       char * error, size_t error_capacity) {
    if (!config || config->abi_version != EMBER_XDNA_MOE_PROVIDER_ABI_VERSION ||
        config->struct_size < sizeof(ember_xdna_moe_config_v1)) {
        set_error(error, error_capacity, "invalid XDNA2 provider config ABI");
        return nullptr;
    }
    try {
        return new Provider(*config);
    } catch (const std::exception & exception) {
        set_error(error, error_capacity, exception.what());
        return nullptr;
    }
}

int compute_provider(void * context, const ember_xdna_moe_batch_v1 * batch,
                     char * error, size_t error_capacity) {
    if (!context || !batch ||
        batch->abi_version != EMBER_XDNA_MOE_PROVIDER_ABI_VERSION ||
        batch->struct_size < sizeof(ember_xdna_moe_batch_v1)) {
        set_error(error, error_capacity, "invalid XDNA2 batch ABI");
        return 0;
    }
    std::string message;
    const int result = static_cast<Provider *>(context)->compute(*batch, &message);
    if (!result) set_error(error, error_capacity, message);
    return result;
}

int healthy_provider(void * context) {
    return context && static_cast<Provider *>(context)->healthy() ? 1 : 0;
}

void destroy_provider(void * context) {
    delete static_cast<Provider *>(context);
}

const ember_xdna_moe_provider_v1 kProvider = {
    EMBER_XDNA_MOE_PROVIDER_ABI_VERSION,
    sizeof(ember_xdna_moe_provider_v1),
    "xrt-iron-rocmfp2-bringup",
    create_provider,
    compute_provider,
    healthy_provider,
    destroy_provider,
};

}  // namespace

extern "C" const ember_xdna_moe_provider_v1 *
ember_xdna_moe_get_provider_v1(void) {
    return &kProvider;
}
