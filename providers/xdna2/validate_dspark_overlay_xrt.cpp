// Validate the complete resident DSpark dense overlay on XDNA2.
//
// One xclbin/hw_context alternates every trained five-row attention projection
// shape and the shared expert.  Shape changes are instruction-BO descriptor
// changes only; the image build separately requires every generated PDI to be
// byte-identical.  This is the hardware gate the complete draft provider uses.

#include "q8_0_pack.h"
#include "q8_model_weights.h"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kBatch = 5;
constexpr int kEmbd = 4096;
constexpr int kFf = 2048;
constexpr int kHeader = kBatch * ember::xdna2::kQ8TileK;
constexpr int kPacketBf16 = 256;
constexpr int kOutputsPerGroup = ember::xdna2::kQ8OutputsPerPass;
constexpr size_t kGroupPacketBf16 =
    static_cast<size_t>(ember::xdna2::kQ8AieRows) *
    ember::xdna2::kQ8AieColumns * kBatch * kPacketBf16;
using Clock = std::chrono::steady_clock;

struct ProjectionSpec {
    const char * label;
    const char * tensor;
    int k;
    int n;
    int instruction_n;
    int logical_group_n;
};

constexpr ProjectionSpec kProjectionSpecs[] = {
    {"qakv", nullptr, 4096, 1536, 2048, 0},
    {"qb", "blk.0.attn_q_b.weight", 1024, 32768, 32768, 0},
    // output-A is eight independent head groups. Each consumes a distinct
    // 4096-wide attention row and produces 1024 lanes, padded to one 2048-lane
    // resident group. Treating it as an ordinary wide GEMM is incorrect.
    {"oa_grouped", "blk.0.attn_output_a.weight", 4096, 8192, 16384, 1024},
    {"ob", "blk.0.attn_output_b.weight", 8192, 4096, 4096, 0},
};

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

float bf16_round(float value) {
    return bf16_to_float(float_to_bf16(value));
}

void store_raw_float(uint16_t * destination, float value) {
    std::memcpy(destination, &value, sizeof(value));
}

float exp_approx(float x) {
    if (x < -87.0f) return 0.0f;
    if (x > 87.0f) x = 87.0f;
    constexpr float inv_ln2 = 1.4426950408889634f;
    constexpr float ln2 = 0.6931471805599453f;
    const float scaled = x * inv_ln2;
    const int exponent = static_cast<int>(
        scaled + (scaled < 0.0f ? -0.5f : 0.5f));
    const float r = x - static_cast<float>(exponent) * ln2;
    float p = 1.0f / 39916800.0f;
    p = 1.0f / 3628800.0f + r * p;
    p = 1.0f / 362880.0f + r * p;
    p = 1.0f / 40320.0f + r * p;
    p = 1.0f / 5040.0f + r * p;
    p = 1.0f / 720.0f + r * p;
    p = 1.0f / 120.0f + r * p;
    p = 1.0f / 24.0f + r * p;
    p = 1.0f / 6.0f + r * p;
    p = 0.5f + r * p;
    p = 1.0f + r * p;
    p = 1.0f + r * p;
    const uint32_t bits = static_cast<uint32_t>(exponent + 127) << 23;
    float power = 0.0f;
    std::memcpy(&power, &bits, sizeof(power));
    return power * p;
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

struct Metrics {
    float max_abs = 0.0f;
    double mean_abs = 0.0;
    double cosine = 0.0;
};

Metrics compare(const std::vector<float> & actual,
                const std::vector<float> & expected) {
    if (actual.size() != expected.size())
        throw std::runtime_error("comparison shape mismatch");
    Metrics metrics;
    double error_sum = 0.0, dot = 0.0, aa = 0.0, ee = 0.0;
    for (size_t index = 0; index < actual.size(); ++index) {
        const float error = std::fabs(actual[index] - expected[index]);
        metrics.max_abs = std::max(metrics.max_abs, error);
        error_sum += error;
        dot += static_cast<double>(actual[index]) * expected[index];
        aa += static_cast<double>(actual[index]) * actual[index];
        ee += static_cast<double>(expected[index]) * expected[index];
    }
    metrics.mean_abs = error_sum / static_cast<double>(actual.size());
    if (aa > 0.0 && ee > 0.0) metrics.cosine = dot / std::sqrt(aa * ee);
    return metrics;
}

struct CommandBuffers {
    std::vector<uint32_t> instructions;
    std::unique_ptr<xrt::bo> instruction;
    std::unique_ptr<xrt::bo> input;
    std::unique_ptr<xrt::bo> weight;
    std::unique_ptr<xrt::bo> projection;
    std::unique_ptr<xrt::bo> staging;

    CommandBuffers(xrt::device & device, xrt::kernel & kernel,
                   const std::string & instruction_path, size_t input_bytes,
                   size_t weight_bytes, size_t projection_bytes,
                   size_t staging_bytes)
        : instructions(read_instructions(instruction_path)),
          instruction(std::make_unique<xrt::bo>(
              device, instructions.size() * sizeof(uint32_t),
              XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1))),
          input(std::make_unique<xrt::bo>(
              device, input_bytes, XRT_BO_FLAGS_HOST_ONLY,
              kernel.group_id(3))),
          weight(std::make_unique<xrt::bo>(
              device, weight_bytes, XRT_BO_FLAGS_HOST_ONLY,
              kernel.group_id(4))),
          projection(std::make_unique<xrt::bo>(
              device, projection_bytes, XRT_BO_FLAGS_HOST_ONLY,
              kernel.group_id(5))),
          staging(std::make_unique<xrt::bo>(
              device, staging_bytes, XRT_BO_FLAGS_HOST_ONLY,
              kernel.group_id(6))) {
        std::memcpy(instruction->map<void *>(), instructions.data(),
                    instructions.size() * sizeof(uint32_t));
        instruction->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    void run(xrt::kernel & kernel) {
        xrt::run command(kernel);
        command.set_arg(0, 3);
        command.set_arg(1, *instruction);
        command.set_arg(2, static_cast<uint32_t>(instructions.size()));
        command.set_arg(3, *input);
        command.set_arg(4, *weight);
        command.set_arg(5, *projection);
        command.set_arg(6, *staging);
        command.start();
        if (command.wait() != ERT_CMD_STATE_COMPLETED)
            throw std::runtime_error("resident command did not complete");
    }
};

class ProjectionJob {
public:
    ProjectionJob(xrt::device & device, xrt::kernel & kernel,
                  const std::string & directory, const std::string & model,
                  const ProjectionSpec & spec)
        : spec_(spec), padded_n_(std::max(spec.n, kOutputsPerGroup)),
          groups_(padded_n_ / kOutputsPerGroup) {
        std::string error;
        std::vector<uint8_t> packed;
        if (spec.tensor) {
            ember::xdna2::Q8ModelProjection projection;
            if (!ember::xdna2::load_q8_model_projection(
                    model.c_str(), spec.tensor, spec.k, spec.n,
                    projection, &error))
                throw std::runtime_error(error);
            raw_ = std::move(projection.raw);
            if (spec.logical_group_n > 0) {
                groups_ = spec.n / spec.logical_group_n;
                padded_n_ = groups_ * kOutputsPerGroup;
                if (!ember::xdna2::pack_q8_grouped_projection_corrected_bf16(
                        raw_.data(), raw_.size(), spec.k,
                        spec.logical_group_n, groups_, kOutputsPerGroup,
                        packed, &error))
                    throw std::runtime_error(error);
            } else {
                packed = std::move(projection.packed);
            }
        } else {
            ember::xdna2::Q8ModelProjection qa, kv;
            if (!ember::xdna2::load_q8_model_projection(
                    model.c_str(), "blk.0.attn_q_a.weight", spec.k, 1024,
                    qa, &error) ||
                !ember::xdna2::load_q8_model_projection(
                    model.c_str(), "blk.0.attn_kv.weight", spec.k, 512,
                    kv, &error))
                throw std::runtime_error(error);
            raw_ = std::move(qa.raw);
            raw_.insert(raw_.end(), kv.raw.begin(), kv.raw.end());
            std::vector<uint8_t> combined;
            if (!ember::xdna2::concat_q8_projection_rows(
                    qa.packed, spec.k, 1024, kv.packed, 512,
                    combined, &error) ||
                !ember::xdna2::pad_q8_projection_rows(
                    combined, spec.k, spec.n, padded_n_, packed, &error))
                throw std::runtime_error(error);
        }
        const std::string instructions = directory +
            "/dspark_resident_q8_v1_projection_" +
            std::to_string(spec.k) + "x" +
            std::to_string(spec.instruction_n) + "_b5.insts";
        const size_t input_elements = static_cast<size_t>(kHeader) +
            static_cast<size_t>(groups_) * kBatch * spec.k;
        buffers_ = std::make_unique<CommandBuffers>(
            device, kernel, instructions, input_elements * sizeof(uint16_t),
            packed.size(), static_cast<size_t>(groups_) *
                kGroupPacketBf16 * sizeof(uint16_t), 1);
        auto * input = buffers_->input->map<uint16_t *>();
        std::fill_n(input, input_elements, static_cast<uint16_t>(0));
        input[1] = float_to_bf16(static_cast<float>(
            spec.k / ember::xdna2::kQ8TileK));
        input[2] = float_to_bf16(static_cast<float>(groups_));
        const int activation_groups = spec.logical_group_n > 0 ? groups_ : 1;
        input_values_.resize(static_cast<size_t>(activation_groups) *
                             kBatch * spec.k);
        for (int token = 0; token < kBatch; ++token) {
            for (int lane = 0; lane < spec.k; ++lane) {
                for (int group = 0; group < groups_; ++group) {
                    const int activation_group =
                        spec.logical_group_n > 0 ? group : 0;
                    const uint16_t bits = float_to_bf16(
                        static_cast<float>((lane + token +
                            activation_group * 7) % 31 - 15) / 32.0f);
                    input_values_[
                        (static_cast<size_t>(activation_group) * kBatch +
                         token) * spec.k + lane] = bf16_to_float(bits);
                    const size_t destination = static_cast<size_t>(kHeader) +
                        (static_cast<size_t>(group) * kBatch + token) * spec.k +
                        lane;
                    input[destination] = bits;
                }
            }
        }
        std::memcpy(buffers_->weight->map<void *>(), packed.data(), packed.size());
        buffers_->input->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        buffers_->weight->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    void run(xrt::kernel & kernel) { buffers_->run(kernel); }
    const char * label() const { return spec_.label; }

    Metrics validate() {
        buffers_->projection->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        const float * output = buffers_->projection->map<const float *>();
        std::vector<float> actual(static_cast<size_t>(kBatch) * spec_.n);
        std::vector<float> expected(actual.size());
        const int logical_groups = spec_.logical_group_n > 0 ? groups_ : 1;
        const int outputs_per_logical_group = spec_.logical_group_n > 0
            ? spec_.logical_group_n : spec_.n;
        const size_t raw_group_bytes = ember::xdna2::q8_projection_bytes(
            spec_.k, outputs_per_logical_group);
        for (int logical_group = 0; logical_group < logical_groups;
             ++logical_group) {
            for (int token = 0; token < kBatch; ++token) {
                if (!ember::xdna2::q8_gemm_raw_reference(
                        raw_.data() +
                            static_cast<size_t>(logical_group) * raw_group_bytes,
                        raw_group_bytes,
                        input_values_.data() +
                            (static_cast<size_t>(logical_group) * kBatch +
                             token) * spec_.k,
                        spec_.k, outputs_per_logical_group,
                        expected.data() + static_cast<size_t>(token) * spec_.n +
                            static_cast<size_t>(logical_group) *
                                outputs_per_logical_group))
                    throw std::runtime_error("projection reference failed");
            }
        }
        for (int token = 0; token < kBatch; ++token) {
            for (int lane = 0; lane < spec_.n; ++lane) {
                const int logical_group = spec_.logical_group_n > 0
                    ? lane / spec_.logical_group_n : 0;
                const int logical_lane = spec_.logical_group_n > 0
                    ? lane % spec_.logical_group_n : lane;
                const int physical_lane = spec_.logical_group_n > 0
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
                actual[static_cast<size_t>(token) * spec_.n + lane] =
                    output[source];
            }
        }
        return compare(actual, expected);
    }

private:
    ProjectionSpec spec_;
    int padded_n_;
    int groups_;
    std::vector<uint8_t> raw_;
    std::vector<float> input_values_;
    std::unique_ptr<CommandBuffers> buffers_;
};

class SharedJob {
public:
    SharedJob(xrt::device & device, xrt::kernel & kernel,
              const std::string & directory, const std::string & model) {
        std::string error;
        if (!ember::xdna2::load_q8_model_shared_expert(
                model.c_str(), 0, weights_, &error))
            throw std::runtime_error(error);
        const std::string instructions = directory +
            "/dspark_resident_q8_v1_shared_4096x2048x4096_b5.insts";
        constexpr size_t row_elements = kEmbd + ember::xdna2::kQ8TileK;
        buffers_ = std::make_unique<CommandBuffers>(
            device, kernel, instructions,
            (static_cast<size_t>(kHeader) + kBatch * row_elements) *
                sizeof(uint16_t), weights_.packed.size(), 1,
            kGroupPacketBf16 * sizeof(uint16_t));
        auto * input = buffers_->input->map<uint16_t *>();
        std::fill_n(input, static_cast<size_t>(kHeader) + kBatch * row_elements,
                    static_cast<uint16_t>(0));
        input[3] = float_to_bf16(1.0f);
        input_values_.resize(static_cast<size_t>(kBatch) * kEmbd);
        for (int token = 0; token < kBatch; ++token) {
            uint16_t * row = input + kHeader +
                static_cast<size_t>(token) * row_elements;
            for (int lane = 0; lane < kEmbd; ++lane) {
                const float value = bf16_round(
                    static_cast<float>((lane * 37 + token * 53) % 257 - 128) /
                    128.0f);
                input_values_[static_cast<size_t>(token) * kEmbd + lane] = value;
                row[lane] = float_to_bf16(value);
            }
            store_raw_float(row + kEmbd, 1.0f);
            store_raw_float(row + kEmbd + 2, 1.0f);
            store_raw_float(row + kEmbd + 4, 10.0f);
            store_raw_float(row + kEmbd + 6, 1.0f);
        }
        std::memcpy(buffers_->weight->map<void *>(), weights_.packed.data(),
                    weights_.packed.size());
        buffers_->input->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        buffers_->weight->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    void run(xrt::kernel & kernel) { buffers_->run(kernel); }

    Metrics validate() {
        std::vector<float> expected(static_cast<size_t>(kBatch) * kEmbd);
        for (int token = 0; token < kBatch; ++token) {
            const float * input = input_values_.data() +
                static_cast<size_t>(token) * kEmbd;
            std::vector<float> gate(kFf), up(kFf), hidden(kFf), result(kEmbd);
            if (!ember::xdna2::q8_gemm_raw_reference(
                    weights_.gate.data(), weights_.gate.size(), input,
                    kEmbd, kFf, gate.data()) ||
                !ember::xdna2::q8_gemm_raw_reference(
                    weights_.up.data(), weights_.up.size(), input,
                    kEmbd, kFf, up.data()))
                throw std::runtime_error("shared gate/up reference failed");
            for (int lane = 0; lane < kFf; ++lane) {
                const float gate_value = std::min(gate[lane], 10.0f);
                const float up_value = std::max(-10.0f, std::min(up[lane], 10.0f));
                hidden[lane] = bf16_round(
                    (gate_value / (1.0f + exp_approx(-gate_value))) * up_value);
            }
            if (!ember::xdna2::q8_gemm_raw_reference(
                    weights_.down.data(), weights_.down.size(), hidden.data(),
                    kFf, kEmbd, result.data()))
                throw std::runtime_error("shared down reference failed");
            std::copy(result.begin(), result.end(),
                      expected.begin() + static_cast<size_t>(token) * kEmbd);
        }
        buffers_->staging->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        const uint16_t * staging = buffers_->staging->map<const uint16_t *>();
        std::vector<float> actual(static_cast<size_t>(kBatch) * kEmbd);
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
                            actual[destination] = packet[group * 64 + lane];
                        }
                    }
                }
            }
        }
        return compare(actual, expected);
    }

private:
    ember::xdna2::Q8ModelSharedExpert weights_;
    std::vector<float> input_values_;
    std::unique_ptr<CommandBuffers> buffers_;
};

template <typename Job>
double time_runs(Job & job, xrt::kernel & kernel, int repeats) {
    const auto begin = Clock::now();
    for (int run = 0; run < repeats; ++run) job.run(kernel);
    return std::chrono::duration<double, std::milli>(Clock::now() - begin)
               .count() / repeats;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 3 && argc != 4) {
        std::fprintf(stderr, "usage: %s ARTIFACT_DIR DRAFT.gguf [REPEATS]\n",
                     argv[0]);
        return 2;
    }
    try {
        const int repeats = argc == 4 ? std::atoi(argv[3]) : 20;
        if (repeats < 1 || repeats > 10000)
            throw std::runtime_error("REPEATS must be in [1,10000]");
        const std::string directory = argv[1];
        const std::string model = argv[2];
        xrt::device device(0);
        xrt::xclbin xclbin(directory + "/dspark_resident_q8_v1_b5.xclbin");
        const auto kernels = xclbin.get_kernels();
        const auto found = std::find_if(
            kernels.begin(), kernels.end(),
            [](const xrt::xclbin::kernel & candidate) {
                return candidate.get_name().rfind("MLIR_AIE", 0) == 0;
            });
        if (found == kernels.end()) throw std::runtime_error("AIE kernel absent");
        device.register_xclbin(xclbin);
        xrt::hw_context context(device, xclbin.get_uuid());
        xrt::kernel kernel(context, found->get_name());

        std::vector<std::unique_ptr<ProjectionJob>> projections;
        for (const ProjectionSpec & spec : kProjectionSpecs) {
            projections.push_back(std::make_unique<ProjectionJob>(
                device, kernel, directory, model, spec));
        }
        SharedJob shared(device, kernel, directory, model);

        bool pass = true;
        double standalone_ms = 0.0;
        for (auto & projection : projections) {
            projection->run(kernel);
            const Metrics metrics = projection->validate();
            const double milliseconds =
                time_runs(*projection, kernel, repeats);
            standalone_ms += milliseconds;
            std::printf(
                "dspark_overlay mode=%s sequence_ms=%.6f max_abs=%.8g "
                "mean_abs=%.8g cosine=%.10f\n",
                projection->label(), milliseconds, metrics.max_abs,
                metrics.mean_abs, metrics.cosine);
            pass = pass && metrics.cosine >= 0.99999 &&
                metrics.max_abs <= 0.001f;
        }
        shared.run(kernel);
        const Metrics shared_metrics = shared.validate();
        const double shared_ms = time_runs(shared, kernel, repeats);
        standalone_ms += shared_ms;
        std::printf(
            "dspark_overlay mode=shared sequence_ms=%.6f max_abs=%.8g "
            "mean_abs=%.8g cosine=%.10f\n",
            shared_ms, shared_metrics.max_abs, shared_metrics.mean_abs,
            shared_metrics.cosine);
        pass = pass && shared_metrics.cosine >= 0.99999 &&
            shared_metrics.max_abs <= 0.01f;

        const auto cycle_begin = Clock::now();
        for (int repeat = 0; repeat < repeats; ++repeat) {
            for (auto & projection : projections) projection->run(kernel);
            shared.run(kernel);
        }
        const double cycle_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - cycle_begin).count() / repeats;
        std::printf(
            "dspark_overlay cycle_ms=%.6f standalone_ms=%.6f "
            "descriptor_overhead_ms=%.6f repeats=%d\n",
            cycle_ms, standalone_ms, cycle_ms - standalone_ms, repeats);
        std::printf("DSPARK_OVERLAY_VALIDATION_%s\n", pass ? "PASS" : "FAIL");
        return pass ? 0 : 1;
    } catch (const std::exception & exception) {
        std::fprintf(stderr, "DSpark overlay validation failed: %s\n",
                     exception.what());
        return 1;
    }
}
