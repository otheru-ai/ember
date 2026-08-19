// Complete asynchronous CPU/XDNA2 DSpark provider.
//
// The engine owns the managed-UMA draft tensors. At create time this provider
// converts only the dense Q8 matrices into the resident AIE layout; the 10+ GiB
// routed ROCMFP4 tensors remain zero-copy CPU views. One worker owns the XRT
// context and serializes NPU descriptors, while submit() only copies request
// activations. During each FFN, the resident shared expert overlaps the CPU's
// routed experts. The GPU remains free to verify another resident session.

#include "dspark_cpu_ops.h"
#include "dspark_draft_compute_xdna.h"
#include "q8_0_pack.h"
#include "rocmfp4_pack.h"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr int kBatch = 5;
constexpr int kLayers = 3;
constexpr int kEmbd = 4096;
constexpr int kHc = 4;
constexpr int kHcWidth = kHc * kEmbd;
constexpr int kMix = 2 * kHc + kHc * kHc;
constexpr int kHeads = 64;
constexpr int kHeadDim = 512;
constexpr int kRope = 64;
constexpr int kQa = 1024;
constexpr int kFf = 2048;
constexpr int kExperts = 256;
constexpr int kTopK = 6;
constexpr int kOutGroups = 8;
constexpr int kOutGroupDim = 4096;
constexpr int kOutLowGroup = 1024;
constexpr int kHeader = kBatch * ember::xdna2::kQ8TileK;
constexpr int kPacketBf16 = 256;
constexpr int kOutputsPerGroup = ember::xdna2::kQ8OutputsPerPass;
constexpr size_t kGroupPacketBf16 =
    static_cast<size_t>(ember::xdna2::kQ8AieRows) *
    ember::xdna2::kQ8AieColumns * kBatch * kPacketBf16;
constexpr int kTypeF32 = 0;
constexpr int kTypeF16 = 1;
constexpr int kTypeQ8 = 8;
constexpr int kTypeBF16 = 30;
constexpr int kTypeRocmfp4Fast = 101;

void trace_stats(const char * label, int layer,
                 const std::vector<float> & values) {
    if (!std::getenv("DFLASH_DS4_DSPARK_DEBUG")) return;
    double squares = 0.0;
    size_t nonfinite = 0;
    float minimum = std::numeric_limits<float>::infinity();
    float maximum = -std::numeric_limits<float>::infinity();
    for (const float value : values) {
        if (!std::isfinite(value)) {
            ++nonfinite;
            continue;
        }
        squares += static_cast<double>(value) * value;
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
    std::fprintf(stderr,
                 "[xdna-dspark-dbg] %-10s%d ne=%zu nnan=%zu "
                 "rms=%.4f min=%.3f max=%.3f\n",
                 label, layer, values.size(), nonfinite,
                 values.empty() ? 0.0 :
                     std::sqrt(squares / static_cast<double>(values.size())),
                 minimum, maximum);
}

void trace_difference(const char * label, int layer,
                      const std::vector<float> & actual,
                      const std::vector<float> & reference) {
    if (!std::getenv("DFLASH_DS4_DSPARK_DEBUG") ||
        actual.size() != reference.size() || actual.empty()) return;
    float maximum = 0.0f;
    double absolute = 0.0;
    double dot = 0.0;
    double actual_squares = 0.0;
    double reference_squares = 0.0;
    for (size_t index = 0; index < actual.size(); ++index) {
        maximum = std::max(maximum,
                           std::fabs(actual[index] - reference[index]));
        absolute += std::fabs(actual[index] - reference[index]);
        dot += static_cast<double>(actual[index]) * reference[index];
        actual_squares += static_cast<double>(actual[index]) * actual[index];
        reference_squares +=
            static_cast<double>(reference[index]) * reference[index];
    }
    const double cosine = actual_squares > 0.0 && reference_squares > 0.0
        ? dot / std::sqrt(actual_squares * reference_squares) : 0.0;
    std::fprintf(stderr,
                 "[xdna-dspark-q8] %-10s%d ne=%zu max_abs=%.9g "
                 "mean_abs=%.9g cosine=%.10f\n",
                 label, layer, actual.size(), maximum,
                 absolute / static_cast<double>(actual.size()), cosine);
}

void set_error(char * error, size_t capacity, const std::string & message) {
    if (!error || capacity == 0) return;
    const size_t count = std::min(capacity - 1, message.size());
    std::memcpy(error, message.data(), count);
    error[count] = '\0';
}

uint16_t float_to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
}

float fp16_to_float(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    const uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t fraction = value & 0x03ffu;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (fraction == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((fraction & 0x0400u) == 0) {
                fraction <<= 1;
                ++shift;
            }
            fraction &= 0x03ffu;
            bits = sign | static_cast<uint32_t>(113 - shift) << 23 |
                   fraction << 13;
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | fraction << 13;
    } else {
        bits = sign | (exponent + 112u) << 23 | fraction << 13;
    }
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float bf16_to_float(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

void store_raw_float(uint16_t * destination, float value) {
    std::memcpy(destination, &value, sizeof(value));
}

std::vector<uint32_t> read_instructions(const std::string & path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open " + path);
    const std::streamsize length = file.tellg();
    if (length <= 0 || length % static_cast<std::streamsize>(sizeof(uint32_t)))
        throw std::runtime_error("invalid instruction stream " + path);
    file.seekg(0);
    std::vector<uint32_t> words(static_cast<size_t>(length) / sizeof(uint32_t));
    if (!file.read(reinterpret_cast<char *>(words.data()), length))
        throw std::runtime_error("short instruction stream " + path);
    return words;
}

class ViewTable {
public:
    ViewTable(const ember_xdna_dspark_tensor_view_v1 * views, uint32_t count) {
        if (!views || count == 0) throw std::runtime_error("draft weight views absent");
        for (uint32_t index = 0; index < count; ++index) {
            const auto & view = views[index];
            if (view.abi_version != EMBER_XDNA_DSPARK_PROVIDER_ABI_VERSION ||
                view.struct_size < sizeof(view) || !view.name || !view.data)
                throw std::runtime_error("invalid draft weight view");
            views_.emplace(view.name, &view);
        }
    }

    const ember_xdna_dspark_tensor_view_v1 & require(
            const std::string & name, int type,
            std::initializer_list<int64_t> dims) const {
        const auto & view = require_shape(name, dims);
        if (view.type != type) {
            throw std::runtime_error(
                "wrong type for " + name + ": got " +
                std::to_string(view.type) + ", expected " +
                std::to_string(type));
        }
        return view;
    }

    const ember_xdna_dspark_tensor_view_v1 & require_shape(
            const std::string & name,
            std::initializer_list<int64_t> dims) const {
        const auto found = views_.find(name);
        if (found == views_.end()) throw std::runtime_error("missing " + name);
        const auto & view = *found->second;
        size_t dimension = 0;
        for (int64_t expected : dims) {
            if (dimension >= 4 || view.dims[dimension] != expected)
                throw std::runtime_error("wrong shape for " + name);
            ++dimension;
        }
        return view;
    }

    const ember_xdna_dspark_tensor_view_v1 * find(
            const std::string & name) const {
        const auto found = views_.find(name);
        return found == views_.end() ? nullptr : found->second;
    }

private:
    std::unordered_map<std::string,
                       const ember_xdna_dspark_tensor_view_v1 *> views_;
};

std::vector<float> copy_f32(
        const ember_xdna_dspark_tensor_view_v1 & view, size_t elements) {
    std::vector<float> result(elements);
    if (view.type == kTypeF32) {
        if (view.bytes < elements * sizeof(float))
            throw std::runtime_error(std::string("short F32 tensor ") + view.name);
        const auto * source = static_cast<const float *>(view.data);
        std::copy_n(source, elements, result.data());
        return result;
    }
    if (view.type == kTypeF16 || view.type == kTypeBF16) {
        if (view.bytes < elements * sizeof(uint16_t))
            throw std::runtime_error(std::string("short 16-bit tensor ") + view.name);
        const auto * source = static_cast<const uint16_t *>(view.data);
        for (size_t index = 0; index < elements; ++index) {
            result[index] = view.type == kTypeF16
                ? fp16_to_float(source[index])
                : bf16_to_float(source[index]);
        }
        return result;
    }
    throw std::runtime_error(
        std::string("unsupported floating tensor type ") +
        std::to_string(view.type) + " for " + view.name);
}

struct CommandBuffers {
    std::vector<uint32_t> words;
    xrt::bo instructions;
    xrt::bo input;
    xrt::bo weight;
    xrt::bo projection;
    xrt::bo staging;

    CommandBuffers(xrt::device & device, xrt::kernel & kernel,
                   const std::string & instruction_path, size_t input_bytes,
                   const std::vector<uint8_t> & packed, size_t projection_bytes,
                   size_t staging_bytes)
        : words(read_instructions(instruction_path)),
          instructions(device, words.size() * sizeof(uint32_t),
                       XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1)),
          input(device, input_bytes, XRT_BO_FLAGS_HOST_ONLY,
                kernel.group_id(3)),
          weight(device, packed.size(), XRT_BO_FLAGS_HOST_ONLY,
                 kernel.group_id(4)),
          projection(device, projection_bytes, XRT_BO_FLAGS_HOST_ONLY,
                     kernel.group_id(5)),
          staging(device, staging_bytes, XRT_BO_FLAGS_HOST_ONLY,
                  kernel.group_id(6)) {
        std::memcpy(instructions.map<void *>(), words.data(),
                    words.size() * sizeof(uint32_t));
        std::memcpy(weight.map<void *>(), packed.data(), packed.size());
        instructions.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        weight.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    std::unique_ptr<xrt::run> start(xrt::kernel & kernel) {
        auto command = std::make_unique<xrt::run>(kernel);
        command->set_arg(0, 3);
        command->set_arg(1, instructions);
        command->set_arg(2, static_cast<uint32_t>(words.size()));
        command->set_arg(3, input);
        command->set_arg(4, weight);
        command->set_arg(5, projection);
        command->set_arg(6, staging);
        command->start();
        return command;
    }
};

class ProjectionRun {
public:
    ProjectionRun(xrt::device & device, xrt::kernel & kernel,
                  const std::string & instructions,
                  std::vector<uint8_t> packed, int k, int real_n,
                  int physical_groups, int logical_group_n = 0)
        : k_(k), real_n_(real_n), groups_(physical_groups),
          logical_group_n_(logical_group_n),
          buffers_(device, kernel, instructions,
                   (static_cast<size_t>(kHeader) +
                    static_cast<size_t>(groups_) * kBatch * k_) *
                       sizeof(uint16_t),
                   packed, static_cast<size_t>(groups_) *
                       kGroupPacketBf16 * sizeof(uint16_t), 1) {}

    bool run(xrt::kernel & kernel, const float * activations,
             std::vector<float> & output, std::string * error) {
        if (!activations) {
            if (error) *error = "null projection activation";
            return false;
        }
        const size_t input_elements = static_cast<size_t>(kHeader) +
            static_cast<size_t>(groups_) * kBatch * k_;
        auto * mapped = buffers_.input.map<uint16_t *>();
        std::fill_n(mapped, input_elements, static_cast<uint16_t>(0));
        mapped[1] = float_to_bf16(static_cast<float>(
            k_ / ember::xdna2::kQ8TileK));
        mapped[2] = float_to_bf16(static_cast<float>(groups_));
        for (int group = 0; group < groups_; ++group) {
            const int activation_group = logical_group_n_ > 0 ? group : 0;
            for (int token = 0; token < kBatch; ++token) {
                const float * source = activations +
                    (static_cast<size_t>(activation_group) * kBatch + token) * k_;
                uint16_t * destination = mapped + kHeader +
                    (static_cast<size_t>(group) * kBatch + token) * k_;
                for (int lane = 0; lane < k_; ++lane)
                    destination[lane] = float_to_bf16(source[lane]);
            }
        }
        buffers_.input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto command = buffers_.start(kernel);
        if (command->wait() != ERT_CMD_STATE_COMPLETED) {
            if (error) *error = "resident projection command failed";
            return false;
        }
        buffers_.projection.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        const float * staging = buffers_.projection.map<const float *>();
        output.resize(static_cast<size_t>(kBatch) * real_n_);
        for (int token = 0; token < kBatch; ++token) {
            for (int lane = 0; lane < real_n_; ++lane) {
                const int logical_group = logical_group_n_ > 0
                    ? lane / logical_group_n_ : 0;
                const int logical_lane = logical_group_n_ > 0
                    ? lane % logical_group_n_ : lane;
                const int physical_lane = logical_group_n_ > 0
                    ? logical_group * kOutputsPerGroup + logical_lane : lane;
                const int group = physical_lane / kOutputsPerGroup;
                const int within_group = physical_lane % kOutputsPerGroup;
                const int row = within_group /
                    ember::xdna2::kQ8OutputsPerRow;
                const int within_row = within_group %
                    ember::xdna2::kQ8OutputsPerRow;
                const int column = within_row / ember::xdna2::kQ8TileN;
                const int output_lane = within_row % ember::xdna2::kQ8TileN;
                const size_t source = static_cast<size_t>(group) *
                        kGroupPacketBf16 / 2 +
                    static_cast<size_t>(row * ember::xdna2::kQ8AieColumns +
                                        column) * kBatch * kPacketBf16 / 2 +
                    static_cast<size_t>(token) * ember::xdna2::kQ8TileN +
                    output_lane;
                output[static_cast<size_t>(token) * real_n_ + lane] =
                    staging[source];
            }
        }
        return true;
    }

private:
    int k_;
    int real_n_;
    int groups_;
    int logical_group_n_;
    CommandBuffers buffers_;
};

class SharedRun {
public:
    SharedRun(xrt::device & device, xrt::kernel & kernel,
              const std::string & instructions, std::vector<uint8_t> packed)
        : buffers_(device, kernel, instructions,
                   (static_cast<size_t>(kHeader) +
                    static_cast<size_t>(kBatch) *
                        (kEmbd + ember::xdna2::kQ8TileK)) * sizeof(uint16_t),
                   packed, 1, kGroupPacketBf16 * sizeof(uint16_t)) {}

    std::unique_ptr<xrt::run> start(xrt::kernel & kernel,
                                    const float * activations) {
        constexpr int row_elements = kEmbd + ember::xdna2::kQ8TileK;
        auto * mapped = buffers_.input.map<uint16_t *>();
        std::fill_n(mapped, static_cast<size_t>(kHeader) +
                    static_cast<size_t>(kBatch) * row_elements,
                    static_cast<uint16_t>(0));
        mapped[3] = float_to_bf16(1.0f);
        for (int token = 0; token < kBatch; ++token) {
            uint16_t * row = mapped + kHeader +
                static_cast<size_t>(token) * row_elements;
            const float * source = activations +
                static_cast<size_t>(token) * kEmbd;
            for (int lane = 0; lane < kEmbd; ++lane)
                row[lane] = float_to_bf16(source[lane]);
            store_raw_float(row + kEmbd, 1.0f);
            store_raw_float(row + kEmbd + 2, 1.0f);
            store_raw_float(row + kEmbd + 4, 10.0f);
            store_raw_float(row + kEmbd + 6, 1.0f);
        }
        buffers_.input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        return buffers_.start(kernel);
    }

    bool finish(std::unique_ptr<xrt::run> command,
                std::vector<float> & output, std::string * error) {
        if (!command || command->wait() != ERT_CMD_STATE_COMPLETED) {
            if (error) *error = "resident shared-expert command failed";
            return false;
        }
        buffers_.staging.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        const uint16_t * staging = buffers_.staging.map<const uint16_t *>();
        output.resize(static_cast<size_t>(kBatch) * kEmbd);
        for (int row = 0; row < 4; ++row) {
            for (int column = 0; column < 8; ++column) {
                const size_t core = static_cast<size_t>(row * 8 + column);
                for (int token = 0; token < kBatch; ++token) {
                    const float * packet = reinterpret_cast<const float *>(
                        staging + core * kBatch * kPacketBf16 +
                        static_cast<size_t>(token) * kPacketBf16);
                    for (int group = 0; group < 2; ++group) {
                        for (int lane = 0; lane < 64; ++lane) {
                            const size_t destination =
                                static_cast<size_t>(token) * kEmbd +
                                static_cast<size_t>(group * 2048 + row * 512 +
                                                    column * 64 + lane);
                            output[destination] = packet[group * 64 + lane];
                        }
                    }
                }
            }
        }
        return true;
    }

private:
    CommandBuffers buffers_;
};

struct HcWeights {
    std::vector<float> fn;
    std::vector<float> base;
    float scale[3] = {};
};

struct RoutedViews {
    const uint8_t * gate = nullptr;
    const uint8_t * up = nullptr;
    const uint8_t * down = nullptr;
    size_t gate_stride = 0;
    size_t up_stride = 0;
    size_t down_stride = 0;
};

struct RawQ8View {
    const void * data = nullptr;
    size_t bytes = 0;
};

struct LayerState {
    HcWeights attn_hc;
    HcWeights ffn_hc;
    std::vector<float> attn_norm;
    std::vector<float> qa_norm;
    std::vector<float> kv_norm;
    std::vector<float> sinks;
    std::vector<float> ffn_norm;
    std::vector<float> router;
    std::vector<float> router_bias;
    RawQ8View qa;
    RawQ8View kv;
    RawQ8View qb_raw;
    RawQ8View oa_raw;
    RawQ8View ob_raw;
    RoutedViews routed;
    std::unique_ptr<ProjectionRun> qakv;
    std::unique_ptr<ProjectionRun> qb;
    std::unique_ptr<ProjectionRun> oa;
    std::unique_ptr<ProjectionRun> ob;
    std::unique_ptr<SharedRun> shared;
};

struct Job {
    int committed = 0;
    int ctx_len = 0;
    std::vector<float> noise;
    std::vector<float> context_kv;
    std::vector<float> hidden;
    std::vector<float> confidence;
    std::string error;
    std::mutex lock;
    std::condition_variable complete;
    bool done = false;
    bool success = false;
    bool cancelled = false;
};

class ProviderContext {
public:
    explicit ProviderContext(const ember_xdna_dspark_config_v1 & config)
        : n_swa_(config.n_swa), views_(config.weight_views,
                                      config.weight_view_count),
          device_(0), xclbin_(artifact("dspark_resident_q8_v1_b5.xclbin")) {
        if (!config.weights_cpu_accessible)
            throw std::runtime_error("draft weights are not managed-UMA accessible");
        if (config.n_embd != kEmbd || config.n_target_layers != kLayers ||
            config.block_size != kBatch || config.head_dim != kHeadDim ||
            config.n_swa <= 0)
            throw std::runtime_error("unsupported DSpark dimensions");
        const auto kernels = xclbin_.get_kernels();
        const auto found = std::find_if(
            kernels.begin(), kernels.end(),
            [](const xrt::xclbin::kernel & candidate) {
                return candidate.get_name().rfind("MLIR_AIE", 0) == 0;
            });
        if (found == kernels.end()) throw std::runtime_error("AIE kernel absent");
        device_.register_xclbin(xclbin_);
        hardware_ = std::make_unique<xrt::hw_context>(device_, xclbin_.get_uuid());
        kernel_ = std::make_unique<xrt::kernel>(*hardware_, found->get_name());
        load_weights();
        worker_ = std::thread([this] { worker_loop(); });
    }

    ~ProviderContext() {
        {
            std::lock_guard<std::mutex> guard(queue_lock_);
            stopping_ = true;
        }
        queue_ready_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

    Job * submit(const ember_xdna_dspark_request_v1 & request) {
        if (request.ctx_len > n_swa_)
            throw std::runtime_error("DSpark context exceeds sliding window");
        auto job = std::make_unique<Job>();
        job->committed = request.committed;
        job->ctx_len = request.ctx_len;
        job->noise.assign(request.noise_embed,
                          request.noise_embed + static_cast<size_t>(kBatch) * kEmbd);
        if (request.ctx_len > 0) {
            if (!request.context_kv)
                throw std::runtime_error("complete provider requires context_kv");
            const size_t count = static_cast<size_t>(kLayers) *
                request.ctx_len * kHeadDim;
            job->context_kv.assign(request.context_kv,
                                   request.context_kv + count);
        }
        Job * result = job.release();
        {
            std::lock_guard<std::mutex> guard(queue_lock_);
            if (stopping_ || queue_.size() >= 8) {
                delete result;
                throw std::runtime_error("DSpark provider queue is full or stopping");
            }
            queue_.push_back(result);
        }
        queue_ready_.notify_one();
        return result;
    }

    bool healthy() const { return healthy_.load(); }

private:
    static std::string artifact(const std::string & name) {
        const char * directory = std::getenv("EMBER_XDNA_ARTIFACT_DIR");
        if (!directory || !*directory)
            throw std::runtime_error("EMBER_XDNA_ARTIFACT_DIR is not set");
        return std::string(directory) + "/" + name;
    }

    static std::string layer_name(int layer, const char * suffix) {
        return "blk." + std::to_string(layer) + "." + suffix;
    }

    static HcWeights load_hc(const ViewTable & views, int layer,
                             const char * prefix) {
        HcWeights result;
        const std::string base = layer_name(layer, prefix);
        result.fn = copy_f32(views.require_shape(
            base + "_fn.weight", {kHcWidth, kMix}),
            static_cast<size_t>(kHcWidth) * kMix);
        result.base = copy_f32(views.require_shape(
            base + "_base.weight", {kMix}), kMix);
        const std::vector<float> scale = copy_f32(views.require_shape(
            base + "_scale.weight", {3}), 3);
        std::copy(scale.begin(), scale.end(), result.scale);
        return result;
    }

    std::vector<uint8_t> pack_projection(const std::string & name,
                                         int k, int n) const {
        const auto & view = views_.require(name, kTypeQ8, {k, n});
        std::vector<uint8_t> packed;
        std::string error;
        if (!ember::xdna2::pack_q8_projection_corrected_bf16(
                view.data, view.bytes, k, n, packed, &error))
            throw std::runtime_error(name + ": " + error);
        return packed;
    }

    static RawQ8View raw_q8(const ember_xdna_dspark_tensor_view_v1 & view) {
        return {view.data, view.bytes};
    }

    static bool reference_projection(const RawQ8View & weight,
                                     const float * input, int rows,
                                     int k, int n,
                                     std::vector<float> & output) {
        output.resize(static_cast<size_t>(rows) * n);
        std::atomic<bool> ok{true};
        ember::xdna2::dspark_parallel_for(rows, [&](int row) {
            if (!ember::xdna2::q8_gemm_raw_reference(
                    weight.data, weight.bytes,
                    input + static_cast<size_t>(row) * k, k, n,
                    output.data() + static_cast<size_t>(row) * n))
                ok.store(false, std::memory_order_relaxed);
        });
        return ok.load(std::memory_order_relaxed);
    }

    void load_weights() {
        layers_.resize(kLayers);
        for (int layer = 0; layer < kLayers; ++layer) {
            LayerState & state = layers_[static_cast<size_t>(layer)];
            state.attn_hc = load_hc(views_, layer, "hc_attn");
            state.ffn_hc = load_hc(views_, layer, "hc_ffn");
            state.attn_norm = copy_f32(views_.require_shape(
                layer_name(layer, "attn_norm.weight"), {kEmbd}), kEmbd);
            state.qa_norm = copy_f32(views_.require_shape(
                layer_name(layer, "attn_q_a_norm.weight"), {kQa}), kQa);
            state.kv_norm = copy_f32(views_.require_shape(
                layer_name(layer, "attn_kv_a_norm.weight"),
                {kHeadDim}), kHeadDim);
            if (const auto * sinks = views_.find(
                    layer_name(layer, "attn_sinks.weight")))
                state.sinks = copy_f32(*sinks, kHeads);
            state.ffn_norm = copy_f32(views_.require_shape(
                layer_name(layer, "ffn_norm.weight"), {kEmbd}), kEmbd);
            state.router = copy_f32(views_.require_shape(
                layer_name(layer, "ffn_gate_inp.weight"),
                {kEmbd, kExperts}), static_cast<size_t>(kEmbd) * kExperts);
            if (const auto * bias = views_.find(
                    layer_name(layer, "exp_probs_b.bias")))
                state.router_bias = copy_f32(*bias, kExperts);

            const auto & qa_view = views_.require(
                layer_name(layer, "attn_q_a.weight"),
                kTypeQ8, {kEmbd, kQa});
            const auto & kv_view = views_.require(
                layer_name(layer, "attn_kv.weight"),
                kTypeQ8, {kEmbd, kHeadDim});
            const auto & qb_view = views_.require(
                layer_name(layer, "attn_q_b.weight"),
                kTypeQ8, {kQa, kHeads * kHeadDim});
            const auto & oa_view = views_.require(
                layer_name(layer, "attn_output_a.weight"), kTypeQ8,
                {kOutGroupDim, kOutLowGroup * kOutGroups});
            const auto & ob_view = views_.require(
                layer_name(layer, "attn_output_b.weight"), kTypeQ8,
                {kOutLowGroup * kOutGroups, kEmbd});
            state.qa = raw_q8(qa_view);
            state.kv = raw_q8(kv_view);
            state.qb_raw = raw_q8(qb_view);
            state.oa_raw = raw_q8(oa_view);
            state.ob_raw = raw_q8(ob_view);

            const auto & gate = views_.require(
                layer_name(layer, "ffn_gate_exps.weight"),
                kTypeRocmfp4Fast, {kEmbd, kFf, kExperts});
            const auto & up = views_.require(
                layer_name(layer, "ffn_up_exps.weight"),
                kTypeRocmfp4Fast, {kEmbd, kFf, kExperts});
            const auto & down = views_.require(
                layer_name(layer, "ffn_down_exps.weight"),
                kTypeRocmfp4Fast, {kFf, kEmbd, kExperts});
            state.routed = {
                static_cast<const uint8_t *>(gate.data),
                static_cast<const uint8_t *>(up.data),
                static_cast<const uint8_t *>(down.data),
                gate.strides[2], up.strides[2], down.strides[2]};

            std::vector<uint8_t> qa = pack_projection(
                layer_name(layer, "attn_q_a.weight"), kEmbd, kQa);
            std::vector<uint8_t> kv = pack_projection(
                layer_name(layer, "attn_kv.weight"), kEmbd, kHeadDim);
            std::vector<uint8_t> qakv, qakv_padded;
            std::string error;
            if (!ember::xdna2::concat_q8_projection_rows(
                    qa, kEmbd, kQa, kv, kHeadDim, qakv, &error) ||
                !ember::xdna2::pad_q8_projection_rows(
                    qakv, kEmbd, kQa + kHeadDim, 2048,
                    qakv_padded, &error))
                throw std::runtime_error("qakv pack: " + error);
            std::vector<uint8_t> qb = pack_projection(
                layer_name(layer, "attn_q_b.weight"), kQa,
                kHeads * kHeadDim);
            std::vector<uint8_t> oa;
            if (!ember::xdna2::pack_q8_grouped_projection_corrected_bf16(
                    oa_view.data, oa_view.bytes, kOutGroupDim, kOutLowGroup,
                    kOutGroups, kOutputsPerGroup, oa, &error))
                throw std::runtime_error("output-A pack: " + error);
            std::vector<uint8_t> ob = pack_projection(
                layer_name(layer, "attn_output_b.weight"),
                kOutLowGroup * kOutGroups, kEmbd);
            const auto & shared_gate = views_.require(
                layer_name(layer, "ffn_gate_shexp.weight"), kTypeQ8,
                {kEmbd, kFf});
            const auto & shared_up = views_.require(
                layer_name(layer, "ffn_up_shexp.weight"), kTypeQ8,
                {kEmbd, kFf});
            const auto & shared_down = views_.require(
                layer_name(layer, "ffn_down_shexp.weight"), kTypeQ8,
                {kFf, kEmbd});
            std::vector<uint8_t> shared;
            if (!ember::xdna2::pack_q8_expert_v2(
                    shared_gate.data, shared_gate.bytes,
                    shared_up.data, shared_up.bytes,
                    shared_down.data, shared_down.bytes, shared, &error))
                throw std::runtime_error("shared expert pack: " + error);

            state.qakv = std::make_unique<ProjectionRun>(
                device_, *kernel_, artifact(
                    "dspark_resident_q8_v1_projection_4096x2048_b5.insts"),
                std::move(qakv_padded), kEmbd, kQa + kHeadDim, 1);
            state.qb = std::make_unique<ProjectionRun>(
                device_, *kernel_, artifact(
                    "dspark_resident_q8_v1_projection_1024x32768_b5.insts"),
                std::move(qb), kQa, kHeads * kHeadDim, 16);
            state.oa = std::make_unique<ProjectionRun>(
                device_, *kernel_, artifact(
                    "dspark_resident_q8_v1_projection_4096x16384_b5.insts"),
                std::move(oa), kOutGroupDim, kOutLowGroup * kOutGroups,
                kOutGroups, kOutLowGroup);
            state.ob = std::make_unique<ProjectionRun>(
                device_, *kernel_, artifact(
                    "dspark_resident_q8_v1_projection_8192x4096_b5.insts"),
                std::move(ob), kOutLowGroup * kOutGroups, kEmbd, 2);
            state.shared = std::make_unique<SharedRun>(
                device_, *kernel_, artifact(
                    "dspark_resident_q8_v1_shared_4096x2048x4096_b5.insts"),
                std::move(shared));
        }
        output_hc_fn_ = copy_f32(views_.require_shape(
            "output_hc_fn.weight", {kHcWidth, kHc}),
            static_cast<size_t>(kHcWidth) * kHc);
        output_hc_base_ = copy_f32(views_.require_shape(
            "output_hc_base.weight", {kHc}), kHc);
        output_hc_scale_ = copy_f32(views_.require_shape(
            "output_hc_scale.weight", {1}), 1).front();
        output_norm_ = copy_f32(views_.require_shape(
            "output_norm.weight", {kEmbd}), kEmbd);
    }

    bool routed_experts(int layer_index, const LayerState & layer,
                        const std::vector<float> & normalized,
                        std::vector<float> & output, std::string * error) {
        std::vector<int32_t> selected;
        std::vector<float> weights;
        const bool route_debug =
            std::getenv("DFLASH_DS4_DSPARK_ROUTE_DEBUG") != nullptr;
        std::vector<ember::xdna2::DsparkRouteBoundary> boundaries;
        if (!ember::xdna2::dspark_route_topk(
                normalized.data(), layer.router.data(),
                layer.router_bias.empty() ? nullptr : layer.router_bias.data(),
                kBatch, kEmbd, kExperts, kTopK, 1.5f,
                selected, weights, error,
                route_debug ? &boundaries : nullptr))
            return false;
        if (std::getenv("DFLASH_DS4_DSPARK_DEBUG") || route_debug) {
            for (int token = 0; token < kBatch; ++token) {
                std::fprintf(stderr,
                             "[xdna-dspark-route] layer=%d token=%d ids=",
                             layer_index, token);
                for (int slot = 0; slot < kTopK; ++slot) {
                    std::fprintf(stderr, "%s%d", slot ? "," : "",
                        selected[static_cast<size_t>(token * kTopK + slot)]);
                }
                if (route_debug) {
                    const auto & boundary =
                        boundaries[static_cast<size_t>(token)];
                    std::fprintf(stderr,
                                 " boundary=%d/%d margin=%.9g",
                                 boundary.selected_expert,
                                 boundary.rejected_expert,
                                 boundary.margin);
                }
                std::fputc('\n', stderr);
            }
        }
        const int routes = kBatch * kTopK;
        const size_t gate_bytes =
            ember::xdna2::rocmfp4_projection_bytes(kEmbd, kFf);
        const size_t down_bytes =
            ember::xdna2::rocmfp4_projection_bytes(kFf, kEmbd);
        std::vector<float> gate(static_cast<size_t>(routes) * kFf);
        std::vector<float> up(static_cast<size_t>(routes) * kFf);
        std::vector<float> hidden(static_cast<size_t>(routes) * kFf);
        std::vector<float> down(static_cast<size_t>(routes) * kEmbd);
        std::atomic<bool> ok{true};
        ember::xdna2::dspark_parallel_for(routes * 2, [&](int task) {
            const int route = task / 2;
            const int token = route / kTopK;
            const int expert = selected[static_cast<size_t>(route)];
            const bool is_gate = (task & 1) == 0;
            const uint8_t * source = (is_gate ? layer.routed.gate :
                layer.routed.up) + static_cast<size_t>(expert) *
                (is_gate ? layer.routed.gate_stride : layer.routed.up_stride);
            float * destination = (is_gate ? gate.data() : up.data()) +
                static_cast<size_t>(route) * kFf;
            if (!ember::xdna2::rocmfp4_gemm_cpu(
                    source, gate_bytes,
                    normalized.data() + static_cast<size_t>(token) * kEmbd,
                    kEmbd, kFf, 1.0f, destination))
                ok.store(false, std::memory_order_relaxed);
        });
        if (!ok.load(std::memory_order_relaxed)) {
            if (error) *error = "routed gate/up projection failed";
            return false;
        }
        ember::xdna2::dspark_parallel_for(routes * kFf, [&](int index) {
            const float gate_value = std::min(gate[static_cast<size_t>(index)],
                                              10.0f);
            const float up_value = std::max(-10.0f,
                std::min(up[static_cast<size_t>(index)], 10.0f));
            hidden[static_cast<size_t>(index)] =
                (gate_value / (1.0f + std::exp(-gate_value))) * up_value;
        });
        ember::xdna2::dspark_parallel_for(routes, [&](int route) {
            const int expert = selected[static_cast<size_t>(route)];
            if (!ember::xdna2::rocmfp4_gemm_cpu(
                    layer.routed.down + static_cast<size_t>(expert) *
                        layer.routed.down_stride,
                    down_bytes,
                    hidden.data() + static_cast<size_t>(route) * kFf,
                    kFf, kEmbd, 1.0f,
                    down.data() + static_cast<size_t>(route) * kEmbd))
                ok.store(false, std::memory_order_relaxed);
        });
        if (!ok.load(std::memory_order_relaxed)) {
            if (error) *error = "routed down projection failed";
            return false;
        }
        output.assign(static_cast<size_t>(kBatch) * kEmbd, 0.0f);
        ember::xdna2::dspark_parallel_for(kBatch * kEmbd, [&](int index) {
            const int token = index / kEmbd;
            const int lane = index % kEmbd;
            float sum = 0.0f;
            for (int route = 0; route < kTopK; ++route) {
                const size_t routed_index =
                    static_cast<size_t>(token * kTopK + route) * kEmbd + lane;
                sum += down[routed_index] *
                    weights[static_cast<size_t>(token * kTopK + route)];
            }
            output[static_cast<size_t>(index)] = sum;
        });
        return true;
    }

    bool compute(Job & job) {
        std::string error;
        const bool debug = std::getenv("DFLASH_DS4_DSPARK_DEBUG") != nullptr;
        unsigned cpu_q8_mask = 0;
        if (const char * raw = std::getenv("EMBER_DSPARK_CPU_Q8_MASK")) {
            char * end = nullptr;
            const unsigned long parsed = std::strtoul(raw, &end, 0);
            if (end != raw && *end == '\0' && parsed <= 15)
                cpu_q8_mask = static_cast<unsigned>(parsed);
        }
        // Output-A only feeds output-B. A CPU output-A diagnostic therefore
        // also needs the CPU output-B projection to observe the substitution.
        if (cpu_q8_mask & 4u) cpu_q8_mask |= 8u;
        std::vector<float> state(static_cast<size_t>(kBatch) * kHcWidth);
        for (int token = 0; token < kBatch; ++token) {
            for (int hc = 0; hc < kHc; ++hc) {
                std::copy_n(job.noise.data() + static_cast<size_t>(token) * kEmbd,
                            kEmbd, state.data() +
                                (static_cast<size_t>(token) * kHc + hc) * kEmbd);
            }
        }
        std::vector<int32_t> query_positions(kBatch);
        std::vector<int32_t> kv_positions(job.ctx_len + kBatch);
        for (int token = 0; token < kBatch; ++token)
            query_positions[static_cast<size_t>(token)] = job.committed + token;
        for (int row = 0; row < job.ctx_len; ++row)
            kv_positions[static_cast<size_t>(row)] =
                job.committed - job.ctx_len + row;
        for (int token = 0; token < kBatch; ++token)
            kv_positions[static_cast<size_t>(job.ctx_len + token)] =
                job.committed + token;

        for (int layer_index = 0; layer_index < kLayers; ++layer_index) {
            LayerState & layer = layers_[static_cast<size_t>(layer_index)];
            std::vector<float> working, normalized;
            ember::xdna2::DsparkHcSplit split;
            if (!ember::xdna2::dspark_hc_pre(
                    state.data(), layer.attn_hc.fn.data(),
                    layer.attn_hc.base.data(), layer.attn_hc.scale,
                    kBatch, kEmbd, kHc, 20, 1.0e-6f,
                    working, split, &error) ||
                !ember::xdna2::dspark_weighted_rms_norm(
                    working.data(), layer.attn_norm.data(), kBatch, kEmbd,
                    1.0e-6f, normalized, &error)) {
                job.error = error;
                return false;
            }
            std::vector<float> qakv;
            if (!layer.qakv->run(*kernel_, normalized.data(), qakv, &error)) {
                job.error = error;
                return false;
            }
            std::vector<float> qa(static_cast<size_t>(kBatch) * kQa);
            std::vector<float> block_kv(static_cast<size_t>(kBatch) * kHeadDim);
            for (int token = 0; token < kBatch; ++token) {
                const float * source = qakv.data() +
                    static_cast<size_t>(token) * (kQa + kHeadDim);
                std::copy_n(source, kQa,
                            qa.data() + static_cast<size_t>(token) * kQa);
                std::copy_n(source + kQa, kHeadDim,
                            block_kv.data() +
                                static_cast<size_t>(token) * kHeadDim);
            }
            if (debug || (cpu_q8_mask & 1u)) {
                std::vector<float> qa_reference, kv_reference;
                if (!reference_projection(layer.qa, normalized.data(),
                                          kBatch, kEmbd, kQa,
                                          qa_reference) ||
                    !reference_projection(layer.kv, normalized.data(),
                                          kBatch, kEmbd, kHeadDim,
                                          kv_reference)) {
                    job.error = "Q8 q-a/KV diagnostic reference failed";
                    return false;
                }
                if (debug) {
                    trace_difference("qa_L", layer_index, qa, qa_reference);
                    trace_difference("kv_L", layer_index,
                                     block_kv, kv_reference);
                }
                if (cpu_q8_mask & 1u) {
                    qa = std::move(qa_reference);
                    block_kv = std::move(kv_reference);
                }
            }
            std::vector<float> qa_normalized, block_kv_normalized;
            if (!ember::xdna2::dspark_weighted_rms_norm(
                    qa.data(), layer.qa_norm.data(), kBatch, kQa, 1.0e-6f,
                    qa_normalized, &error) ||
                !ember::xdna2::dspark_weighted_rms_norm(
                    block_kv.data(), layer.kv_norm.data(), kBatch, kHeadDim,
                    1.0e-6f, block_kv_normalized, &error)) {
                job.error = error;
                return false;
            }
            std::vector<float> q;
            if (!layer.qb->run(*kernel_, qa_normalized.data(), q, &error)) {
                job.error = error;
                return false;
            }
            if (debug || (cpu_q8_mask & 2u)) {
                std::vector<float> reference;
                if (!reference_projection(layer.qb_raw,
                                          qa_normalized.data(), kBatch,
                                          kQa, kHeads * kHeadDim,
                                          reference)) {
                    job.error = "Q8 q-b diagnostic reference failed";
                    return false;
                }
                if (debug)
                    trace_difference("qb_L", layer_index, q, reference);
                if (cpu_q8_mask & 2u) q = std::move(reference);
            }
            std::vector<float> ones(kHeadDim, 1.0f), q_normalized;
            if (!ember::xdna2::dspark_weighted_rms_norm(
                    q.data(), ones.data(), kBatch * kHeads, kHeadDim,
                    1.0e-6f, q_normalized, &error)) {
                job.error = error;
                return false;
            }
            std::vector<float> kv(static_cast<size_t>(job.ctx_len + kBatch) *
                                  kHeadDim);
            if (job.ctx_len > 0) {
                const size_t layer_elements =
                    static_cast<size_t>(job.ctx_len) * kHeadDim;
                std::copy_n(job.context_kv.data() +
                                static_cast<size_t>(layer_index) * layer_elements,
                            layer_elements, kv.data());
            }
            std::copy(block_kv_normalized.begin(), block_kv_normalized.end(),
                      kv.begin() + static_cast<size_t>(job.ctx_len) * kHeadDim);
            std::vector<float> attention;
            if (!ember::xdna2::dspark_attention_reduce(
                    q_normalized.data(), kv.data(),
                    layer.sinks.empty() ? nullptr : layer.sinks.data(),
                    query_positions.data(), kv_positions.data(), kBatch,
                    job.ctx_len, kHeads, kHeadDim, kRope, 10000.0f,
                    attention, &error)) {
                job.error = error;
                return false;
            }
            std::vector<float> oa_input(
                static_cast<size_t>(kOutGroups) * kBatch * kOutGroupDim);
            for (int group = 0; group < kOutGroups; ++group) {
                for (int token = 0; token < kBatch; ++token) {
                    std::copy_n(
                        attention.data() +
                            static_cast<size_t>(token) * kHeads * kHeadDim +
                            static_cast<size_t>(group) * kOutGroupDim,
                        kOutGroupDim,
                        oa_input.data() +
                            (static_cast<size_t>(group) * kBatch + token) *
                                kOutGroupDim);
                }
            }
            std::vector<float> oa, attn_output;
            if (!layer.oa->run(*kernel_, oa_input.data(), oa, &error) ||
                !layer.ob->run(*kernel_, oa.data(), attn_output, &error)) {
                job.error = error;
                return false;
            }
            if (debug || (cpu_q8_mask & 4u)) {
                std::vector<float> oa_reference(
                    static_cast<size_t>(kBatch) *
                    kOutLowGroup * kOutGroups);
                const size_t group_bytes =
                    ember::xdna2::q8_projection_bytes(
                        kOutGroupDim, kOutLowGroup);
                std::atomic<bool> ok{true};
                ember::xdna2::dspark_parallel_for(
                    kBatch * kOutGroups, [&](int task) {
                        const int group = task / kBatch;
                        const int token = task % kBatch;
                        const auto * weight =
                            static_cast<const uint8_t *>(layer.oa_raw.data) +
                            static_cast<size_t>(group) * group_bytes;
                        const float * input = oa_input.data() +
                            (static_cast<size_t>(group) * kBatch + token) *
                                kOutGroupDim;
                        float * output = oa_reference.data() +
                            static_cast<size_t>(token) *
                                kOutLowGroup * kOutGroups +
                            static_cast<size_t>(group) * kOutLowGroup;
                        if (!ember::xdna2::q8_gemm_raw_reference(
                                weight, group_bytes, input, kOutGroupDim,
                                kOutLowGroup, output))
                            ok.store(false, std::memory_order_relaxed);
                    });
                if (!ok.load(std::memory_order_relaxed)) {
                    job.error = "Q8 output-A diagnostic reference failed";
                    return false;
                }
                if (debug)
                    trace_difference("oa_L", layer_index,
                                     oa, oa_reference);
                if (cpu_q8_mask & 4u) oa = std::move(oa_reference);
            }
            if (debug || (cpu_q8_mask & 8u)) {
                std::vector<float> ob_reference;
                if (!reference_projection(
                        layer.ob_raw, oa.data(), kBatch,
                        kOutLowGroup * kOutGroups, kEmbd, ob_reference)) {
                    job.error = "Q8 output-B diagnostic reference failed";
                    return false;
                }
                if (debug)
                    trace_difference("ob_L", layer_index,
                                     attn_output, ob_reference);
                if (cpu_q8_mask & 8u)
                    attn_output = std::move(ob_reference);
            }
            trace_stats("attn_L", layer_index, attn_output);
            std::vector<float> next;
            if (!ember::xdna2::dspark_hc_post(
                    state.data(), attn_output.data(), split, kEmbd,
                    next, &error)) {
                job.error = error;
                return false;
            }
            state = std::move(next);

            if (!ember::xdna2::dspark_hc_pre(
                    state.data(), layer.ffn_hc.fn.data(),
                    layer.ffn_hc.base.data(), layer.ffn_hc.scale,
                    kBatch, kEmbd, kHc, 20, 1.0e-6f,
                    working, split, &error) ||
                !ember::xdna2::dspark_weighted_rms_norm(
                    working.data(), layer.ffn_norm.data(), kBatch, kEmbd,
                    1.0e-6f, normalized, &error)) {
                job.error = error;
                return false;
            }
            auto shared_command = layer.shared->start(*kernel_, normalized.data());
            std::vector<float> routed, shared;
            const bool routed_ok = routed_experts(
                layer_index, layer, normalized, routed, &error);
            const bool shared_ok = layer.shared->finish(
                std::move(shared_command), shared, &error);
            if (!routed_ok || !shared_ok) {
                job.error = error;
                return false;
            }
            ember::xdna2::dspark_parallel_for(kBatch * kEmbd, [&](int index) {
                shared[static_cast<size_t>(index)] +=
                    routed[static_cast<size_t>(index)];
            });
            trace_stats("ffn_L", layer_index, shared);
            if (!ember::xdna2::dspark_hc_post(
                    state.data(), shared.data(), split, kEmbd,
                    next, &error)) {
                job.error = error;
                return false;
            }
            state = std::move(next);
            trace_stats("hcL", layer_index, state);
        }
        if (!ember::xdna2::dspark_hc_out(
                state.data(), output_hc_fn_.data(), output_hc_base_.data(),
                output_hc_scale_, kBatch, kEmbd, kHc, 1.0e-6f,
                job.confidence, &job.error) ||
            !ember::xdna2::dspark_weighted_rms_norm(
                job.confidence.data(), output_norm_.data(), kBatch, kEmbd,
                1.0e-6f, job.hidden, &job.error))
            return false;
        trace_stats("confidence", -1, job.confidence);
        trace_stats("hidden", -1, job.hidden);
        return true;
    }

    void worker_loop() {
        for (;;) {
            Job * job = nullptr;
            {
                std::unique_lock<std::mutex> lock(queue_lock_);
                queue_ready_.wait(lock, [this] {
                    return stopping_ || !queue_.empty();
                });
                if (stopping_ && queue_.empty()) return;
                job = queue_.front();
                queue_.pop_front();
            }
            bool cancelled = false;
            {
                std::lock_guard<std::mutex> guard(job->lock);
                cancelled = job->cancelled;
            }
            bool success = false;
            if (!cancelled) success = compute(*job);
            if (!success && !cancelled) healthy_.store(false);
            {
                std::lock_guard<std::mutex> guard(job->lock);
                job->success = success;
                job->done = true;
                if (cancelled && job->error.empty()) job->error = "job cancelled";
            }
            job->complete.notify_all();
        }
    }

    int n_swa_;
    ViewTable views_;
    xrt::device device_;
    xrt::xclbin xclbin_;
    std::unique_ptr<xrt::hw_context> hardware_;
    std::unique_ptr<xrt::kernel> kernel_;
    std::vector<LayerState> layers_;
    std::vector<float> output_hc_fn_;
    std::vector<float> output_hc_base_;
    float output_hc_scale_ = 0.0f;
    std::vector<float> output_norm_;
    std::mutex queue_lock_;
    std::condition_variable queue_ready_;
    std::deque<Job *> queue_;
    std::thread worker_;
    std::atomic<bool> healthy_{true};
    bool stopping_ = false;
};

void * provider_create(const ember_xdna_dspark_config_v1 * config,
                       char * error, size_t capacity) {
    try {
        if (!config ||
            config->abi_version != EMBER_XDNA_DSPARK_PROVIDER_ABI_VERSION ||
            config->struct_size < sizeof(*config))
            throw std::runtime_error("incompatible DSpark provider config");
        return new ProviderContext(*config);
    } catch (const std::exception & exception) {
        set_error(error, capacity, exception.what());
        return nullptr;
    }
}

void * provider_submit(void * raw,
                       const ember_xdna_dspark_request_v1 * request,
                       char * error, size_t capacity) {
    try {
        auto * context = static_cast<ProviderContext *>(raw);
        if (!context || !request ||
            request->abi_version != EMBER_XDNA_DSPARK_PROVIDER_ABI_VERSION ||
            request->struct_size < sizeof(*request) ||
            request->n_embd != kEmbd ||
            request->n_target_layers != kLayers ||
            request->block_size != kBatch || request->committed < 0 ||
            request->ctx_len < 0 || !request->noise_embed)
            throw std::runtime_error("invalid DSpark request");
        return context->submit(*request);
    } catch (const std::exception & exception) {
        set_error(error, capacity, exception.what());
        return nullptr;
    }
}

int provider_wait(void *, void * raw_job,
                  ember_xdna_dspark_result_v1 * result,
                  char * error, size_t capacity) {
    auto * job = static_cast<Job *>(raw_job);
    const size_t count = static_cast<size_t>(kBatch) * kEmbd;
    if (!job || !result ||
        result->abi_version != EMBER_XDNA_DSPARK_PROVIDER_ABI_VERSION ||
        result->struct_size < sizeof(*result) || !result->hidden ||
        !result->confidence_hidden || result->hidden_capacity < count ||
        result->confidence_capacity < count) {
        set_error(error, capacity, "invalid DSpark result buffer");
        return 0;
    }
    std::unique_lock<std::mutex> lock(job->lock);
    job->complete.wait(lock, [job] { return job->done; });
    if (!job->success) {
        set_error(error, capacity,
                  job->error.empty() ? "DSpark job failed" : job->error);
        return 0;
    }
    std::copy(job->hidden.begin(), job->hidden.end(), result->hidden);
    std::copy(job->confidence.begin(), job->confidence.end(),
              result->confidence_hidden);
    return 1;
}

void provider_cancel(void *, void * raw_job) {
    auto * job = static_cast<Job *>(raw_job);
    if (!job) return;
    std::unique_lock<std::mutex> lock(job->lock);
    job->cancelled = true;
    job->complete.wait(lock, [job] { return job->done; });
}

void provider_destroy_job(void *, void * raw_job) {
    delete static_cast<Job *>(raw_job);
}

int provider_healthy(void * raw) {
    const auto * context = static_cast<const ProviderContext *>(raw);
    return context && context->healthy() ? 1 : 0;
}

void provider_destroy(void * raw) {
    delete static_cast<ProviderContext *>(raw);
}

const ember_xdna_dspark_provider_v1 provider = {
    EMBER_XDNA_DSPARK_PROVIDER_ABI_VERSION,
    sizeof(ember_xdna_dspark_provider_v1),
    "ember-xdna2-dspark-cpu-routed",
    provider_create,
    provider_submit,
    provider_wait,
    provider_cancel,
    provider_destroy_job,
    provider_healthy,
    provider_destroy,
};

}  // namespace

extern "C" const ember_xdna_dspark_provider_v1 *
ember_xdna_dspark_get_provider_v1() {
    return &provider;
}
