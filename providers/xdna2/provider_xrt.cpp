// XRT provider for Ember's experimental XDNA2 selected-expert ABI.
//
// Gen2 keeps projection partials in FP32 but rounds each projection through a
// BF16 output FIFO. Gen3 extends FP32 through that FIFO and host BO. Gen4 keeps
// the FP32 boundary and replaces scalar ROCMFP2 decode with vector uint4 unpack,
// BF16 dequantization, and 64-lane accumulation. Generations 2-4 use one atomic
// XRT runlist for all selected gate/up projections plus one for all downs. Gen5
// keeps all three projections and exact clamped SwiGLU in one resident spatial
// graph per selected expert. It accepts up to five activation rows so a DSpark
// draft block can enter one XRT runlist without cross-token accumulation. Gen6
// is a fixed four-row shared-expert graph: every 128x64 ROCMFP2 tile is decoded
// once and reused across all four activation rows. The provider stays opt-in
// until trained-weight equivalence and end-to-end speedup are measured; UMA is
// not treated as implicit ROCm/XRT coherence.

#include "moe_expert_compute_xdna.h"
#include "rocmfp2_pack.h"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_kernel.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
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

using ProviderClock = std::chrono::steady_clock;
constexpr int kGen5MaxBatchTokens = 5;
constexpr int kGen6BatchTokens = 4;

double elapsed_ms(ProviderClock::time_point begin,
                  ProviderClock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

using ember::xdna2::pack_rocmfp2_gemv;
using ember::xdna2::pack_rocmfp2_gemv_v4;
using ember::xdna2::pack_rocmfp2_expert_v5;
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

float finite_max_abs(const float * values, size_t count, bool * finite) {
    float maximum = 0.0f;
    *finite = true;
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i])) {
            *finite = false;
            return maximum;
        }
        maximum = std::max(maximum, std::fabs(values[i]));
    }
    return maximum;
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

int kernel_generation() {
    const char * raw = nonempty_env("EMBER_XDNA_KERNEL_GEN");
    if (!raw) return 4;
    if (std::strcmp(raw, "1") == 0) return 1;
    if (std::strcmp(raw, "2") == 0) return 2;
    if (std::strcmp(raw, "3") == 0) return 3;
    if (std::strcmp(raw, "4") == 0) return 4;
    if (std::strcmp(raw, "5") == 0) return 5;
    if (std::strcmp(raw, "6") == 0) return 6;
    throw std::runtime_error(
        "EMBER_XDNA_KERNEL_GEN must be 1, 2, 3, 4, 5, or 6");
}

std::string artifact_directory() {
    const char * directory = nonempty_env("EMBER_XDNA_ARTIFACT_DIR");
    return directory ? directory : "/usr/local/share/ember/xdna2";
}

std::string artifact_path(int k, int n, const char * suffix, int generation) {
    std::string path = artifact_directory();
    if (generation == 4) path += "/gemv_v4_";
    else if (generation == 3) path += "/gemv_v3_";
    else if (generation == 2) path += "/gemv_v2_";
    else path += "/gemv_";
    path += std::to_string(k) + "x" + std::to_string(n) + suffix;
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
    struct Invocation {
        const float * input = nullptr;
        xrt::bo * weights = nullptr;
        float * output = nullptr;
    };

    GemvProgram(xrt::device & device, int k, int n, int max_runs,
                int generation)
        : device_(device), k_(k), n_(n), generation_(generation) {
        if (max_runs < 1) throw std::runtime_error("invalid XDNA2 run capacity");
        const std::string xclbin_path =
            artifact_path(k, n, ".xclbin", generation_);
        const std::string insts_path =
            artifact_path(k, n, ".insts", generation_);
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

        const size_t output_element_bytes =
            generation_ >= 3 ? sizeof(float) : sizeof(uint16_t);
        for (int i = 0; i < max_runs; ++i) {
            input_bos_.push_back(std::make_unique<xrt::bo>(
                device_, static_cast<size_t>(k_) * sizeof(uint16_t),
                XRT_BO_FLAGS_HOST_ONLY, kernel_->group_id(3)));
            output_bos_.push_back(std::make_unique<xrt::bo>(
                device_, static_cast<size_t>(n_) * output_element_bytes,
                XRT_BO_FLAGS_HOST_ONLY, kernel_->group_id(5)));
        }
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

    void run_many(const std::vector<Invocation> & invocations) {
        if (invocations.empty()) return;
        if (invocations.size() > input_bos_.size()) {
            throw std::runtime_error("XDNA2 runlist exceeds configured capacity");
        }

        const bool trace = nonempty_env("DFLASH_MOE_XDNA_TRACE") != nullptr;
        const auto start = ProviderClock::now();

        // Gate and up share one activation.  Down projections each have their
        // own SwiGLU result.  Reuse one BO for identical input pointers so the
        // common gate/up phase performs only one activation synchronization.
        std::vector<size_t> input_slot(invocations.size());
        std::vector<const float *> unique_inputs;
        for (size_t run_index = 0; run_index < invocations.size(); ++run_index) {
            const Invocation & invocation = invocations[run_index];
            if (!invocation.input || !invocation.weights || !invocation.output) {
                throw std::runtime_error("invalid XDNA2 GEMV invocation");
            }
            const auto found = std::find(unique_inputs.begin(), unique_inputs.end(),
                                         invocation.input);
            size_t slot = static_cast<size_t>(
                std::distance(unique_inputs.begin(), found));
            if (found == unique_inputs.end()) {
                slot = unique_inputs.size();
                unique_inputs.push_back(invocation.input);
                uint16_t * input_bf16 = input_bos_[slot]->map<uint16_t *>();
                for (int i = 0; i < k_; ++i)
                    input_bf16[i] = float_to_bf16(invocation.input[i]);
                input_bos_[slot]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            }
            input_slot[run_index] = slot;
        }
        const auto inputs_ready = ProviderClock::now();

        auto configure_run = [&](size_t index) {
            xrt::run run(*kernel_);
            run.set_arg(0, 3);
            run.set_arg(1, *instruction_bo_);
            run.set_arg(2, static_cast<uint32_t>(instructions_.size()));
            run.set_arg(3, *input_bos_[input_slot[index]]);
            run.set_arg(4, *invocations[index].weights);
            run.set_arg(5, *output_bos_[index]);
            run.set_arg(6, *scratch_bo_);
            run.set_arg(7, *trace_bo_);
            return run;
        };

        if (generation_ >= 2) {
            // XRT 2.26 runlists submit the phase atomically and wait once.  The
            // runs still execute in order on one AIE context, but host dispatch
            // and synchronization no longer scale with selected-expert count.
            xrt::runlist list(*context_);
            for (size_t i = 0; i < invocations.size(); ++i)
                list.add(configure_run(i));
            list.execute();
            list.wait();
        } else {
            for (size_t i = 0; i < invocations.size(); ++i) {
                xrt::run run = configure_run(i);
                run.start();
                if (run.wait() != ERT_CMD_STATE_COMPLETED) {
                    throw std::runtime_error("XDNA2 GEMV command did not complete");
                }
            }
        }
        const auto execution_ready = ProviderClock::now();

        for (size_t run_index = 0; run_index < invocations.size(); ++run_index) {
            output_bos_[run_index]->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            if (generation_ >= 3) {
                const float * source = output_bos_[run_index]->map<float *>();
                std::copy(source, source + n_, invocations[run_index].output);
            } else {
                const uint16_t * source =
                    output_bos_[run_index]->map<uint16_t *>();
                for (int i = 0; i < n_; ++i)
                    invocations[run_index].output[i] = bf16_to_float(source[i]);
            }
        }
        const auto outputs_ready = ProviderClock::now();
        if (trace) {
            std::fprintf(stderr,
                "[xdna2-xrt-gemv] k=%d n=%d runs=%zu inputs=%zu "
                "upload_ms=%.3f execute_ms=%.3f download_ms=%.3f "
                "total_ms=%.3f\n",
                k_, n_, invocations.size(), unique_inputs.size(),
                elapsed_ms(start, inputs_ready),
                elapsed_ms(inputs_ready, execution_ready),
                elapsed_ms(execution_ready, outputs_ready),
                elapsed_ms(start, outputs_ready));
        }
    }

    void run(const float * input, xrt::bo & weights, float * output) {
        run_many({Invocation{input, &weights, output}});
    }

private:
    xrt::device & device_;
    int k_;
    int n_;
    int generation_;
    std::unique_ptr<xrt::xclbin> xclbin_;
    std::unique_ptr<xrt::hw_context> context_;
    std::unique_ptr<xrt::kernel> kernel_;
    std::vector<uint32_t> instructions_;
    std::unique_ptr<xrt::bo> instruction_bo_;
    std::vector<std::unique_ptr<xrt::bo>> input_bos_;
    std::vector<std::unique_ptr<xrt::bo>> output_bos_;
    std::unique_ptr<xrt::bo> scratch_bo_;
    std::unique_ptr<xrt::bo> trace_bo_;
};

class ExpertProgram {
public:
    struct Invocation {
        const float * input = nullptr;
        xrt::bo * weights = nullptr;
        float gate_scale = 1.0f;
        float up_scale = 1.0f;
        float clamp = 0.0f;
        float * output = nullptr;
    };

    ExpertProgram(xrt::device & device, int max_runs) : device_(device) {
        if (max_runs < 1) throw std::runtime_error("invalid XDNA2 expert capacity");
        const std::string base = artifact_directory() +
            "/expert_v5_4096x2048x4096";
        xclbin_ = std::make_unique<xrt::xclbin>(base + ".xclbin");
        const auto kernels = xclbin_->get_kernels();
        const auto found = std::find_if(kernels.begin(), kernels.end(),
            [](const xrt::xclbin::kernel & kernel) {
                return kernel.get_name().rfind("MLIR_AIE", 0) == 0;
            });
        if (found == kernels.end()) throw std::runtime_error("Gen5 kernel absent");
        device_.register_xclbin(*xclbin_);
        context_ = std::make_unique<xrt::hw_context>(device_, xclbin_->get_uuid());
        kernel_ = std::make_unique<xrt::kernel>(*context_, found->get_name());
        instructions_ = read_instructions(base + ".insts");
        instruction_bo_ = std::make_unique<xrt::bo>(
            device_, instructions_.size() * sizeof(uint32_t),
            XCL_BO_FLAGS_CACHEABLE, kernel_->group_id(1));
        std::memcpy(instruction_bo_->map<void *>(), instructions_.data(),
                    instructions_.size() * sizeof(uint32_t));
        instruction_bo_->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        for (int i = 0; i < max_runs; ++i) {
            input_bos_.push_back(std::make_unique<xrt::bo>(
                device_, (4096 + 128) * sizeof(uint16_t),
                XRT_BO_FLAGS_HOST_ONLY, kernel_->group_id(3)));
            staging_bos_.push_back(std::make_unique<xrt::bo>(
                device_, 8192 * sizeof(uint16_t),
                XRT_BO_FLAGS_HOST_ONLY, kernel_->group_id(5)));
        }
        dummy6_ = std::make_unique<xrt::bo>(
            device_, 1, XRT_BO_FLAGS_HOST_ONLY, kernel_->group_id(6));
        dummy7_ = std::make_unique<xrt::bo>(
            device_, 1, XRT_BO_FLAGS_HOST_ONLY, kernel_->group_id(7));
    }

    std::unique_ptr<xrt::bo> make_weight_bo(const std::vector<uint8_t> & packed) {
        auto result = std::make_unique<xrt::bo>(
            device_, packed.size(), XRT_BO_FLAGS_HOST_ONLY, kernel_->group_id(4));
        std::memcpy(result->map<void *>(), packed.data(), packed.size());
        result->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        return result;
    }

    void run_many(const std::vector<Invocation> & invocations) {
        if (invocations.empty()) return;
        if (invocations.size() > input_bos_.size())
            throw std::runtime_error("XDNA2 Gen5 runlist exceeds capacity");
        auto raw_float = [](uint16_t * destination, float value) {
            std::memcpy(destination, &value, sizeof(value));
        };
        for (size_t index = 0; index < invocations.size(); ++index) {
            const Invocation & invocation = invocations[index];
            if (!invocation.input || !invocation.weights || !invocation.output)
                throw std::runtime_error("invalid XDNA2 Gen5 invocation");
            uint16_t * input = input_bos_[index]->map<uint16_t *>();
            for (int i = 0; i < 4096; ++i)
                input[i] = float_to_bf16(invocation.input[i]);
            std::fill(input + 4096, input + 4096 + 128, 0);
            raw_float(input + 4096, invocation.gate_scale);
            raw_float(input + 4098, invocation.up_scale);
            raw_float(input + 4100, invocation.clamp);
            input_bos_[index]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        auto configure = [&](size_t index) {
            xrt::run run(*kernel_);
            run.set_arg(0, 3);
            run.set_arg(1, *instruction_bo_);
            run.set_arg(2, static_cast<uint32_t>(instructions_.size()));
            run.set_arg(3, *input_bos_[index]);
            run.set_arg(4, *invocations[index].weights);
            run.set_arg(5, *staging_bos_[index]);
            run.set_arg(6, *dummy6_);
            run.set_arg(7, *dummy7_);
            return run;
        };
        xrt::runlist list(*context_);
        for (size_t i = 0; i < invocations.size(); ++i) list.add(configure(i));
        list.execute();
        list.wait();
        for (size_t index = 0; index < invocations.size(); ++index) {
            staging_bos_[index]->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            const uint16_t * staging = staging_bos_[index]->map<uint16_t *>();
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 8; ++col) {
                    const size_t core = static_cast<size_t>(row * 8 + col);
                    const float * packet = reinterpret_cast<const float *>(
                        staging + core * 256);
                    for (int group = 0; group < 2; ++group) {
                        for (int lane = 0; lane < 64; ++lane) {
                            const size_t output = static_cast<size_t>(group * 2048 +
                                row * 8 * 64 + col * 64 + lane);
                            invocations[index].output[output] =
                                packet[group * 64 + lane];
                        }
                    }
                }
            }
        }
    }

private:
    xrt::device & device_;
    std::unique_ptr<xrt::xclbin> xclbin_;
    std::unique_ptr<xrt::hw_context> context_;
    std::unique_ptr<xrt::kernel> kernel_;
    std::vector<uint32_t> instructions_;
    std::unique_ptr<xrt::bo> instruction_bo_;
    std::vector<std::unique_ptr<xrt::bo>> input_bos_;
    std::vector<std::unique_ptr<xrt::bo>> staging_bos_;
    std::unique_ptr<xrt::bo> dummy6_;
    std::unique_ptr<xrt::bo> dummy7_;
};

class BatchExpertProgram {
public:
    explicit BatchExpertProgram(xrt::device & device) : device_(device) {
        const std::string base = artifact_directory() +
            "/expert_v6_4096x2048x4096_b4";
        xclbin_ = std::make_unique<xrt::xclbin>(base + ".xclbin");
        const auto kernels = xclbin_->get_kernels();
        const auto found = std::find_if(kernels.begin(), kernels.end(),
            [](const xrt::xclbin::kernel & kernel) {
                return kernel.get_name().rfind("MLIR_AIE", 0) == 0;
            });
        if (found == kernels.end())
            throw std::runtime_error("Gen6 kernel absent");
        device_.register_xclbin(*xclbin_);
        context_ = std::make_unique<xrt::hw_context>(device_, xclbin_->get_uuid());
        kernel_ = std::make_unique<xrt::kernel>(*context_, found->get_name());
        instructions_ = read_instructions(base + ".insts");
        instruction_bo_ = std::make_unique<xrt::bo>(
            device_, instructions_.size() * sizeof(uint32_t),
            XCL_BO_FLAGS_CACHEABLE, kernel_->group_id(1));
        std::memcpy(instruction_bo_->map<void *>(), instructions_.data(),
                    instructions_.size() * sizeof(uint32_t));
        instruction_bo_->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        input_bo_ = std::make_unique<xrt::bo>(
            device_, static_cast<size_t>(kGen6BatchTokens) * (4096 + 128) *
                         sizeof(uint16_t),
            XRT_BO_FLAGS_HOST_ONLY, kernel_->group_id(3));
        staging_bo_ = std::make_unique<xrt::bo>(
            device_, static_cast<size_t>(kGen6BatchTokens) * 8192 *
                         sizeof(uint16_t),
            XRT_BO_FLAGS_HOST_ONLY, kernel_->group_id(5));
        dummy6_ = std::make_unique<xrt::bo>(
            device_, 1, XRT_BO_FLAGS_HOST_ONLY, kernel_->group_id(6));
        dummy7_ = std::make_unique<xrt::bo>(
            device_, 1, XRT_BO_FLAGS_HOST_ONLY, kernel_->group_id(7));
    }

    std::unique_ptr<xrt::bo> make_weight_bo(
            const std::vector<uint8_t> & packed) {
        auto result = std::make_unique<xrt::bo>(
            device_, packed.size(), XRT_BO_FLAGS_HOST_ONLY,
            kernel_->group_id(4));
        std::memcpy(result->map<void *>(), packed.data(), packed.size());
        result->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        return result;
    }

    void run(const float * input_rows, xrt::bo & weights,
             float gate_scale, float up_scale, float clamp,
             float * output_rows) {
        if (!input_rows || !output_rows)
            throw std::runtime_error("invalid XDNA2 Gen6 invocation");
        auto raw_float = [](uint16_t * destination, float value) {
            std::memcpy(destination, &value, sizeof(value));
        };
        uint16_t * input = input_bo_->map<uint16_t *>();
        for (int token = 0; token < kGen6BatchTokens; ++token) {
            uint16_t * row = input + static_cast<size_t>(token) * (4096 + 128);
            for (int i = 0; i < 4096; ++i) {
                row[i] = float_to_bf16(
                    input_rows[static_cast<size_t>(token) * 4096 + i]);
            }
            std::fill(row + 4096, row + 4096 + 128, 0);
            raw_float(row + 4096, gate_scale);
            raw_float(row + 4098, up_scale);
            raw_float(row + 4100, clamp);
        }
        input_bo_->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        xrt::run run(*kernel_);
        run.set_arg(0, 3);
        run.set_arg(1, *instruction_bo_);
        run.set_arg(2, static_cast<uint32_t>(instructions_.size()));
        run.set_arg(3, *input_bo_);
        run.set_arg(4, weights);
        run.set_arg(5, *staging_bo_);
        run.set_arg(6, *dummy6_);
        run.set_arg(7, *dummy7_);
        run.start();
        if (run.wait() != ERT_CMD_STATE_COMPLETED)
            throw std::runtime_error("XDNA2 Gen6 command did not complete");
        staging_bo_->sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        const uint16_t * staging = staging_bo_->map<uint16_t *>();
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 8; ++col) {
                const size_t core = static_cast<size_t>(row * 8 + col);
                for (int token = 0; token < kGen6BatchTokens; ++token) {
                    const float * packet = reinterpret_cast<const float *>(
                        staging + core * kGen6BatchTokens * 256 +
                        static_cast<size_t>(token) * 256);
                    for (int group = 0; group < 2; ++group) {
                        for (int lane = 0; lane < 64; ++lane) {
                            const size_t output =
                                static_cast<size_t>(token) * 4096 +
                                static_cast<size_t>(group * 2048 +
                                    row * 8 * 64 + col * 64 + lane);
                            output_rows[output] = packet[group * 64 + lane];
                        }
                    }
                }
            }
        }
    }

private:
    xrt::device & device_;
    std::unique_ptr<xrt::xclbin> xclbin_;
    std::unique_ptr<xrt::hw_context> context_;
    std::unique_ptr<xrt::kernel> kernel_;
    std::vector<uint32_t> instructions_;
    std::unique_ptr<xrt::bo> instruction_bo_;
    std::unique_ptr<xrt::bo> input_bo_;
    std::unique_ptr<xrt::bo> staging_bo_;
    std::unique_ptr<xrt::bo> dummy6_;
    std::unique_ptr<xrt::bo> dummy7_;
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

struct ExpertKey {
    const void * gate = nullptr;
    const void * up = nullptr;
    const void * down = nullptr;
    size_t gate_bytes = 0;
    size_t up_bytes = 0;
    size_t down_bytes = 0;
    bool operator==(const ExpertKey & other) const {
        return gate == other.gate && up == other.up && down == other.down &&
               gate_bytes == other.gate_bytes && up_bytes == other.up_bytes &&
               down_bytes == other.down_bytes;
    }
};

struct ExpertKeyHash {
    size_t operator()(const ExpertKey & key) const {
        size_t hash = std::hash<const void *>{}(key.gate);
        auto mix = [&](size_t value) {
            hash ^= value + 0x9e3779b9u + (hash << 6) + (hash >> 2);
        };
        mix(std::hash<const void *>{}(key.up));
        mix(std::hash<const void *>{}(key.down));
        mix(key.gate_bytes); mix(key.up_bytes); mix(key.down_bytes);
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
        : config_(config), generation_(kernel_generation()),
          device_(device_index()), capacity_(cache_capacity_bytes()) {
        if (config.n_embd <= 0 || config.n_ff_exp <= 0 ||
            !rocmfp2_supported_shape(config.n_embd, config.n_ff_exp) ||
            !rocmfp2_supported_shape(config.n_ff_exp, config.n_embd)) {
            throw std::runtime_error("model dimensions are unsupported by XDNA2 GEMV");
        }
        if (generation_ == 6) {
            if (config.n_embd != 4096 || config.n_ff_exp != 2048)
                throw std::runtime_error("Gen6 requires 4096x2048 experts");
            batch_expert_ = std::make_unique<BatchExpertProgram>(device_);
        } else if (generation_ == 5) {
            if (config.n_embd != 4096 || config.n_ff_exp != 2048)
                throw std::runtime_error("Gen5 requires 4096x2048 experts");
            expert_ = std::make_unique<ExpertProgram>(
                device_, std::max(1, config.n_expert_used) *
                             kGen5MaxBatchTokens);
        } else {
            gate_up_ = std::make_unique<GemvProgram>(
                device_, config.n_embd, config.n_ff_exp,
                std::max(2, config.n_expert_used * 2), generation_);
            down_ = std::make_unique<GemvProgram>(
                device_, config.n_ff_exp, config.n_embd,
                std::max(1, config.n_expert_used), generation_);
        }
    }

    int compute(const ember_xdna_moe_batch_v1 & batch, std::string * error) {
        std::lock_guard<std::mutex> guard(lock_);
        if (!healthy_.load()) {
            if (error) *error = "XDNA2 provider is unhealthy";
            return 0;
        }
        if (batch.n_tokens <= 0 || batch.n_selected <= 0 || !batch.input ||
            !batch.router_weights || !batch.output || !batch.expert_weights ||
            batch.n_selected > config_.n_expert_used ||
            batch.n_embd != config_.n_embd || batch.n_ff_exp != config_.n_ff_exp) {
            if (error) *error = "unsupported or invalid XDNA2 batch";
            return 0;
        }
        const bool invalid_shape =
            (generation_ == 6 &&
             (batch.n_tokens != kGen6BatchTokens || batch.n_selected != 1)) ||
            (generation_ == 5 && batch.n_tokens > kGen5MaxBatchTokens) ||
            (generation_ < 5 && batch.n_tokens != 1);
        if (invalid_shape) {
            if (error) {
                *error = generation_ == 6
                    ? "XDNA2 Gen6 requires exactly four tokens and one expert"
                    : generation_ == 5
                        ? "XDNA2 Gen5 supports at most five tokens"
                        : "XDNA2 Gen1-4 require one token";
            }
            return 0;
        }

        try {
            const bool trace = nonempty_env("DFLASH_MOE_XDNA_TRACE") != nullptr;
            const auto start = ProviderClock::now();
            bool input_finite = false;
            const float input_max = finite_max_abs(
                batch.input,
                static_cast<size_t>(batch.n_tokens) *
                    static_cast<size_t>(batch.n_embd),
                &input_finite);
            if (!input_finite) {
                if (error) *error = "non-finite XDNA input at layer " +
                    std::to_string(batch.layer_idx);
                return 0;
            }
            std::fill(batch.output,
                      batch.output +
                          static_cast<size_t>(batch.n_tokens) *
                              static_cast<size_t>(batch.n_embd),
                      0.0f);
            const size_t selected = static_cast<size_t>(batch.n_selected);
            const size_t tokens = static_cast<size_t>(batch.n_tokens);
            const size_t slots = tokens * selected;
            const size_t ff = static_cast<size_t>(batch.n_ff_exp);
            const size_t embd = static_cast<size_t>(batch.n_embd);
            std::vector<float> gate(selected * ff);
            std::vector<float> up(selected * ff);
            std::vector<float> hidden(selected * ff);
            std::vector<float> projected(selected * embd);
            std::vector<std::shared_ptr<CachedWeight>> gate_weights(selected);
            std::vector<std::shared_ptr<CachedWeight>> up_weights(selected);
            std::vector<std::shared_ptr<CachedWeight>> down_weights(selected);
            const auto buffers_ready = ProviderClock::now();

            if (generation_ == 6) {
                const auto & first = batch.expert_weights[0];
                if (!batch.expert_ids ||
                    first.struct_size < sizeof(ember_xdna_moe_weight_view_v1) ||
                    first.fused_gate_up || !first.gate || !first.up || !first.down ||
                    first.gate_format != EMBER_XDNA_MOE_WEIGHT_ROCMFP2 ||
                    first.up_format != EMBER_XDNA_MOE_WEIGHT_ROCMFP2 ||
                    first.down_format != EMBER_XDNA_MOE_WEIGHT_ROCMFP2) {
                    if (error) *error = "Gen6 shared expert is not separate ROCMFP2";
                    return 0;
                }
                for (size_t slot = 1; slot < slots; ++slot) {
                    const auto & view = batch.expert_weights[slot];
                    if (view.struct_size <
                            sizeof(ember_xdna_moe_weight_view_v1) ||
                        batch.expert_ids[slot] != batch.expert_ids[0] ||
                        view.gate != first.gate || view.up != first.up ||
                        view.down != first.down ||
                        view.gate_bytes != first.gate_bytes ||
                        view.up_bytes != first.up_bytes ||
                        view.down_bytes != first.down_bytes ||
                        view.gate_scale != first.gate_scale ||
                        view.up_scale != first.up_scale ||
                        view.down_scale != first.down_scale) {
                        if (error) *error =
                            "Gen6 requires one shared expert across all rows";
                        return 0;
                    }
                }
                const std::shared_ptr<CachedWeight> packed = expert_weight(first);
                const auto weights_ready = ProviderClock::now();
                projected.resize(tokens * embd);
                batch_expert_->run(
                    batch.input, *packed->bo, first.gate_scale, first.up_scale,
                    config_.swiglu_clamp, projected.data());
                const auto execution_ready = ProviderClock::now();
                for (size_t token = 0; token < tokens; ++token) {
                    bool finite = false;
                    (void)finite_max_abs(projected.data() + token * embd,
                                         embd, &finite);
                    if (!finite) {
                        if (error) *error = "non-finite XDNA Gen6 output";
                        return 0;
                    }
                    const float scale = first.down_scale *
                        batch.router_weights[token];
                    for (int i = 0; i < batch.n_embd; ++i) {
                        batch.output[token * embd + static_cast<size_t>(i)] =
                            projected[token * embd + static_cast<size_t>(i)] *
                            scale;
                    }
                    ++experts_;
                }
                const auto accumulated = ProviderClock::now();
                kernel_runs_ += tokens;
                ++submissions_;
                if (trace) {
                    std::fprintf(stderr,
                        "[xdna2-xrt] generation=6 layer=%d tokens=%d "
                        "setup_ms=%.3f pack_ms=%.3f execute_ms=%.3f "
                        "accum_ms=%.3f total_ms=%.3f\n",
                        batch.layer_idx, batch.n_tokens,
                        elapsed_ms(start, buffers_ready),
                        elapsed_ms(buffers_ready, weights_ready),
                        elapsed_ms(weights_ready, execution_ready),
                        elapsed_ms(execution_ready, accumulated),
                        elapsed_ms(start, accumulated));
                }
                ++calls_;
                return 1;
            }

            if (generation_ == 5) {
                std::vector<std::shared_ptr<CachedWeight>> expert_weights(slots);
                for (size_t slot = 0; slot < slots; ++slot) {
                    const auto & view = batch.expert_weights[slot];
                    if (view.struct_size < sizeof(ember_xdna_moe_weight_view_v1) ||
                        view.fused_gate_up || !view.gate || !view.up || !view.down ||
                        view.gate_format != EMBER_XDNA_MOE_WEIGHT_ROCMFP2 ||
                        view.up_format != EMBER_XDNA_MOE_WEIGHT_ROCMFP2 ||
                        view.down_format != EMBER_XDNA_MOE_WEIGHT_ROCMFP2) {
                        if (error) *error = "selected expert is not separate ROCMFP2";
                        return 0;
                    }
                    expert_weights[slot] = expert_weight(view);
                }
                const auto weights_ready = ProviderClock::now();
                projected.resize(slots * embd);
                std::vector<ExpertProgram::Invocation> runs;
                runs.reserve(slots);
                for (size_t slot = 0; slot < slots; ++slot) {
                    const auto & view = batch.expert_weights[slot];
                    const size_t token = slot / selected;
                    runs.push_back({batch.input + token * embd,
                                    expert_weights[slot]->bo.get(),
                                    view.gate_scale, view.up_scale,
                                    config_.swiglu_clamp,
                                    projected.data() + slot * embd});
                }
                expert_->run_many(runs);
                const auto execution_ready = ProviderClock::now();
                for (size_t index = 0; index < slots; ++index) {
                    const auto & view = batch.expert_weights[index];
                    bool finite = false;
                    (void)finite_max_abs(projected.data() + index * embd,
                                         embd, &finite);
                    if (!finite) {
                        if (error) *error = "non-finite XDNA Gen5 output";
                        return 0;
                    }
                    const size_t token = index / selected;
                    const float scale = view.down_scale *
                        batch.router_weights[index];
                    for (int i = 0; i < batch.n_embd; ++i)
                        batch.output[token * embd + static_cast<size_t>(i)] +=
                            projected[index * embd + static_cast<size_t>(i)] *
                            scale;
                    ++experts_;
                }
                const auto accumulated = ProviderClock::now();
                kernel_runs_ += slots;
                ++submissions_;
                if (trace) {
                    std::fprintf(stderr,
                        "[xdna2-xrt] generation=5 layer=%d tokens=%d experts=%d "
                        "setup_ms=%.3f pack_ms=%.3f execute_ms=%.3f "
                        "accum_ms=%.3f total_ms=%.3f\n",
                        batch.layer_idx, batch.n_tokens, batch.n_selected,
                        elapsed_ms(start, buffers_ready),
                        elapsed_ms(buffers_ready, weights_ready),
                        elapsed_ms(weights_ready, execution_ready),
                        elapsed_ms(execution_ready, accumulated),
                        elapsed_ms(start, accumulated));
                }
                ++calls_;
                return 1;
            }

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

                const size_t index = static_cast<size_t>(slot);
                gate_weights[index] = weight(
                    view.gate, view.gate_bytes,
                    batch.n_embd, batch.n_ff_exp, *gate_up_);
                up_weights[index] = weight(
                    view.up, view.up_bytes,
                    batch.n_embd, batch.n_ff_exp, *gate_up_);
                down_weights[index] = weight(
                    view.down, view.down_bytes,
                    batch.n_ff_exp, batch.n_embd, *down_);
            }
            const auto weights_ready = ProviderClock::now();

            std::vector<GemvProgram::Invocation> gate_up_runs;
            gate_up_runs.reserve(selected * 2);
            for (size_t slot = 0; slot < selected; ++slot) {
                gate_up_runs.push_back({batch.input, gate_weights[slot]->bo.get(),
                                        gate.data() + slot * ff});
                gate_up_runs.push_back({batch.input, up_weights[slot]->bo.get(),
                                        up.data() + slot * ff});
            }
            gate_up_->run_many(gate_up_runs);
            const auto gate_up_ready = ProviderClock::now();

            for (int slot = 0; slot < batch.n_selected; ++slot) {
                const size_t index = static_cast<size_t>(slot);
                const auto & view = batch.expert_weights[slot];
                bool gate_finite = false;
                bool up_finite = false;
                (void)finite_max_abs(gate.data() + index * ff, ff, &gate_finite);
                (void)finite_max_abs(up.data() + index * ff, ff, &up_finite);
                if (!gate_finite || !up_finite) {
                    if (error) *error = "non-finite XDNA gate/up output at layer " +
                        std::to_string(batch.layer_idx) + " expert " +
                        std::to_string(batch.expert_ids[slot]);
                    return 0;
                }

                for (int i = 0; i < batch.n_ff_exp; ++i) {
                    const size_t offset = index * ff + static_cast<size_t>(i);
                    float gate_value = gate[offset] * view.gate_scale;
                    float up_value = up[offset] * view.up_scale;
                    if (config_.swiglu_clamp > 1.0e-6f) {
                        gate_value = std::min(gate_value, config_.swiglu_clamp);
                        up_value = std::max(-config_.swiglu_clamp,
                                            std::min(up_value, config_.swiglu_clamp));
                    }
                    hidden[offset] =
                        (gate_value / (1.0f + std::exp(-gate_value))) * up_value;
                }
            }
            const auto swiglu_ready = ProviderClock::now();

            std::vector<GemvProgram::Invocation> down_runs;
            down_runs.reserve(selected);
            for (size_t slot = 0; slot < selected; ++slot) {
                down_runs.push_back({hidden.data() + slot * ff,
                                     down_weights[slot]->bo.get(),
                                     projected.data() + slot * embd});
            }
            down_->run_many(down_runs);
            const auto down_ready = ProviderClock::now();

            for (int slot = 0; slot < batch.n_selected; ++slot) {
                const size_t index = static_cast<size_t>(slot);
                const auto & view = batch.expert_weights[slot];
                bool projected_finite = false;
                (void)finite_max_abs(projected.data() + index * embd, embd,
                                     &projected_finite);
                if (!projected_finite) {
                    if (error) *error = "non-finite XDNA down output at layer " +
                        std::to_string(batch.layer_idx) + " expert " +
                        std::to_string(batch.expert_ids[slot]);
                    return 0;
                }
                const float scale = view.down_scale * batch.router_weights[slot];
                for (int i = 0; i < batch.n_embd; ++i) {
                    batch.output[i] +=
                        projected[index * embd + static_cast<size_t>(i)] * scale;
                }
                ++experts_;
            }
            const auto accumulated = ProviderClock::now();
            kernel_runs_ += selected * 3;
            submissions_ += generation_ >= 2 ? 2 : selected * 3;
            bool output_finite = false;
            const float output_max = finite_max_abs(
                batch.output, static_cast<size_t>(batch.n_embd), &output_finite);
            if (!output_finite) {
                if (error) *error = "non-finite XDNA accumulated output at layer " +
                    std::to_string(batch.layer_idx);
                return 0;
            }
            if (trace) {
                std::fprintf(stderr,
                    "[xdna2-xrt] layer=%d experts=%d input_max=%.7g "
                    "output_max=%.7g setup_ms=%.3f pack_ms=%.3f "
                    "gate_up_ms=%.3f swiglu_ms=%.3f down_ms=%.3f "
                    "accum_ms=%.3f total_ms=%.3f\n",
                    batch.layer_idx, batch.n_selected,
                    static_cast<double>(input_max),
                    static_cast<double>(output_max),
                    elapsed_ms(start, buffers_ready),
                    elapsed_ms(buffers_ready, weights_ready),
                    elapsed_ms(weights_ready, gate_up_ready),
                    elapsed_ms(gate_up_ready, swiglu_ready),
                    elapsed_ms(swiglu_ready, down_ready),
                    elapsed_ms(down_ready, accumulated),
                    elapsed_ms(start, accumulated));
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
                "[xdna2-xrt] generation=%d calls=%llu experts=%llu "
                "kernel_runs=%llu submissions=%llu cache_hits=%llu "
                "cache_misses=%llu cache_bytes=%zu\n",
                generation_,
                static_cast<unsigned long long>(calls_),
                static_cast<unsigned long long>(experts_),
                static_cast<unsigned long long>(kernel_runs_),
                static_cast<unsigned long long>(submissions_),
                static_cast<unsigned long long>(cache_hits_),
                static_cast<unsigned long long>(cache_misses_), cache_bytes_);
        }
    }

private:
    std::shared_ptr<CachedWeight> expert_weight(
            const ember_xdna_moe_weight_view_v1 & view) {
        const ExpertKey key{view.gate, view.up, view.down,
                            view.gate_bytes, view.up_bytes, view.down_bytes};
        const auto found = expert_cache_.find(key);
        if (found != expert_cache_.end()) {
            found->second->last_use = ++clock_;
            ++cache_hits_;
            return found->second;
        }
        std::vector<uint8_t> packed;
        std::string pack_error;
        if (!pack_rocmfp2_expert_v5(
                view.gate, view.gate_bytes, view.up, view.up_bytes,
                view.down, view.down_bytes, packed, &pack_error)) {
            throw std::runtime_error(pack_error);
        }
        if (packed.size() > capacity_)
            throw std::runtime_error("one fused expert exceeds XDNA2 cache");
        while (cache_bytes_ + packed.size() > capacity_ &&
               !expert_cache_.empty()) {
            const auto victim = std::min_element(
                expert_cache_.begin(), expert_cache_.end(),
                [](const auto & left, const auto & right) {
                    return left.second->last_use < right.second->last_use;
                });
            cache_bytes_ -= victim->second->bytes;
            expert_cache_.erase(victim);
        }
        auto entry = std::make_shared<CachedWeight>();
        entry->bo = generation_ == 6
            ? batch_expert_->make_weight_bo(packed)
            : expert_->make_weight_bo(packed);
        entry->bytes = packed.size();
        entry->last_use = ++clock_;
        cache_bytes_ += entry->bytes;
        expert_cache_.emplace(key, entry);
        ++cache_misses_;
        return entry;
    }

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
        const bool packed_ok = generation_ == 4
            ? pack_rocmfp2_gemv_v4(raw, raw_bytes, k, n, packed, &pack_error)
            : pack_rocmfp2_gemv(raw, raw_bytes, k, n, packed, &pack_error);
        if (!packed_ok) {
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
    int generation_ = 4;
    xrt::device device_;
    std::unique_ptr<GemvProgram> gate_up_;
    std::unique_ptr<GemvProgram> down_;
    std::unique_ptr<ExpertProgram> expert_;
    std::unique_ptr<BatchExpertProgram> batch_expert_;
    const size_t capacity_;
    size_t cache_bytes_ = 0;
    uint64_t clock_ = 0;
    uint64_t calls_ = 0;
    uint64_t experts_ = 0;
    uint64_t kernel_runs_ = 0;
    uint64_t submissions_ = 0;
    uint64_t cache_hits_ = 0;
    uint64_t cache_misses_ = 0;
    std::atomic<bool> healthy_{true};
    std::mutex lock_;
    std::unordered_map<WeightKey, std::shared_ptr<CachedWeight>, WeightKeyHash> cache_;
    std::unordered_map<ExpertKey, std::shared_ptr<CachedWeight>, ExpertKeyHash>
        expert_cache_;
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
    "xrt-iron-rocmfp2-multigen",
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
