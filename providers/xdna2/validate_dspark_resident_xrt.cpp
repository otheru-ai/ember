// Validate the resident DSpark Q8 projection/shared-expert overlay on XDNA2.
// One xclbin and one hw_context execute two instruction streams alternately;
// this is the regression gate for descriptor-only mode changes.

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
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kBatch = 5;
constexpr int kEmbd = 4096;
constexpr int kFf = 2048;
constexpr int kHeader = kBatch * ember::xdna2::kQ8TileK;
constexpr int kPacketBf16 = 256;
constexpr int kProjectionN = 1536;
using Clock = std::chrono::steady_clock;

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

std::vector<uint32_t> read_instructions(const char * path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error(std::string("cannot open ") + path);
    const std::streamsize length = file.tellg();
    if (length <= 0 || length % static_cast<std::streamsize>(sizeof(uint32_t)))
        throw std::runtime_error("invalid instruction stream");
    file.seekg(0);
    std::vector<uint32_t> words(static_cast<size_t>(length) / sizeof(uint32_t));
    if (!file.read(reinterpret_cast<char *>(words.data()), length))
        throw std::runtime_error("short instruction stream");
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
    for (size_t i = 0; i < actual.size(); ++i) {
        const float error = std::fabs(actual[i] - expected[i]);
        metrics.max_abs = std::max(metrics.max_abs, error);
        error_sum += error;
        dot += static_cast<double>(actual[i]) * expected[i];
        aa += static_cast<double>(actual[i]) * actual[i];
        ee += static_cast<double>(expected[i]) * expected[i];
    }
    metrics.mean_abs = error_sum / static_cast<double>(actual.size());
    if (aa > 0.0 && ee > 0.0) metrics.cosine = dot / std::sqrt(aa * ee);
    return metrics;
}

struct Workload {
    std::vector<uint32_t> instructions;
    xrt::bo instruction_bo;
    xrt::bo input_bo;
    xrt::bo weight_bo;
    xrt::bo projection_bo;
    xrt::bo staging_bo;

    Workload(xrt::device & device, xrt::kernel & kernel,
             const char * instruction_path, size_t input_bytes,
             size_t weight_bytes, size_t projection_bytes,
             size_t staging_bytes)
        : instructions(read_instructions(instruction_path)),
          instruction_bo(device, instructions.size() * sizeof(uint32_t),
                         XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1)),
          input_bo(device, input_bytes, XRT_BO_FLAGS_HOST_ONLY,
                   kernel.group_id(3)),
          weight_bo(device, weight_bytes, XRT_BO_FLAGS_HOST_ONLY,
                    kernel.group_id(4)),
          projection_bo(device, projection_bytes, XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(5)),
          staging_bo(device, staging_bytes, XRT_BO_FLAGS_HOST_ONLY,
                     kernel.group_id(6)) {
        std::memcpy(instruction_bo.map<void *>(), instructions.data(),
                    instructions.size() * sizeof(uint32_t));
        instruction_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    void run(xrt::kernel & kernel) {
        xrt::run command(kernel);
        command.set_arg(0, 3);
        command.set_arg(1, instruction_bo);
        command.set_arg(2, static_cast<uint32_t>(instructions.size()));
        command.set_arg(3, input_bo);
        command.set_arg(4, weight_bo);
        command.set_arg(5, projection_bo);
        command.set_arg(6, staging_bo);
        command.start();
        if (command.wait() != ERT_CMD_STATE_COMPLETED)
            throw std::runtime_error("resident command did not complete");
    }
};

double time_runs(Workload & workload, xrt::kernel & kernel, int repeats) {
    const auto begin = Clock::now();
    for (int i = 0; i < repeats; ++i) workload.run(kernel);
    return std::chrono::duration<double, std::milli>(Clock::now() - begin)
               .count() / repeats;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 5 || argc > 6) {
        std::fprintf(stderr,
            "usage: %s IMAGE.xclbin PROJECTION.insts SHARED.insts "
            "DRAFT.gguf [REPEATS]\n", argv[0]);
        return 2;
    }
    try {
        const int repeats = argc == 6 ? std::atoi(argv[5]) : 20;
        if (repeats < 1 || repeats > 10000)
            throw std::runtime_error("REPEATS must be in [1,10000]");
        std::string error;

        ember::xdna2::Q8ModelProjection qa, kv;
        if (!ember::xdna2::load_q8_model_projection(
                argv[4], "blk.0.attn_q_a.weight", kEmbd, 1024, qa, &error) ||
            !ember::xdna2::load_q8_model_projection(
                argv[4], "blk.0.attn_kv.weight", kEmbd, 512, kv, &error))
            throw std::runtime_error(error);
        std::vector<uint8_t> projection_raw = qa.raw;
        projection_raw.insert(projection_raw.end(), kv.raw.begin(), kv.raw.end());
        std::vector<uint8_t> combined, projection_packed;
        if (!ember::xdna2::concat_q8_projection_rows(
                qa.packed, kEmbd, 1024, kv.packed, 512, combined, &error) ||
            !ember::xdna2::pad_q8_projection_rows(
                combined, kEmbd, kProjectionN, 2048,
                projection_packed, &error))
            throw std::runtime_error(error);

        ember::xdna2::Q8ModelSharedExpert shared;
        if (!ember::xdna2::load_q8_model_shared_expert(
                argv[4], 0, shared, &error))
            throw std::runtime_error(error);

        xrt::device device(0);
        xrt::xclbin xclbin{std::string(argv[1])};
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

        constexpr size_t packet_elements =
            static_cast<size_t>(32) * kBatch * kPacketBf16;
        Workload projection(
            device, kernel, argv[2],
            (static_cast<size_t>(kHeader) + kBatch * kEmbd) * sizeof(uint16_t),
            projection_packed.size(), packet_elements * sizeof(uint16_t), 1);
        auto * projection_input = projection.input_bo.map<uint16_t *>();
        std::fill_n(projection_input,
                    static_cast<size_t>(kHeader) + kBatch * kEmbd,
                    static_cast<uint16_t>(0));
        projection_input[1] = float_to_bf16(
            static_cast<float>(kEmbd / ember::xdna2::kQ8TileK));
        projection_input[2] = float_to_bf16(1.0f);
        std::vector<float> projection_values(static_cast<size_t>(kBatch) * kEmbd);
        for (int token = 0; token < kBatch; ++token) {
            for (int lane = 0; lane < kEmbd; ++lane) {
                const uint16_t bits = float_to_bf16(
                    static_cast<float>((lane + token) % 31 - 15) / 32.0f);
                projection_input[kHeader + static_cast<size_t>(token) * kEmbd +
                                 lane] = bits;
                projection_values[static_cast<size_t>(token) * kEmbd + lane] =
                    bf16_to_float(bits);
            }
        }
        std::memcpy(projection.weight_bo.map<void *>(), projection_packed.data(),
                    projection_packed.size());
        projection.input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        projection.weight_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        constexpr size_t shared_row = kEmbd + ember::xdna2::kQ8TileK;
        Workload shared_workload(
            device, kernel, argv[3],
            (static_cast<size_t>(kHeader) + kBatch * shared_row) *
                sizeof(uint16_t),
            shared.packed.size(), 1, packet_elements * sizeof(uint16_t));
        auto * shared_input = shared_workload.input_bo.map<uint16_t *>();
        std::fill_n(shared_input,
                    static_cast<size_t>(kHeader) + kBatch * shared_row,
                    static_cast<uint16_t>(0));
        shared_input[3] = float_to_bf16(1.0f);
        std::vector<float> shared_values(static_cast<size_t>(kBatch) * kEmbd);
        for (int token = 0; token < kBatch; ++token) {
            uint16_t * row = shared_input + kHeader +
                static_cast<size_t>(token) * shared_row;
            for (int lane = 0; lane < kEmbd; ++lane) {
                const float value = bf16_round(
                    static_cast<float>((lane * 37 + token * 53) % 257 - 128) /
                    128.0f);
                shared_values[static_cast<size_t>(token) * kEmbd + lane] = value;
                row[lane] = float_to_bf16(value);
            }
            store_raw_float(row + kEmbd, 1.0f);
            store_raw_float(row + kEmbd + 2, 1.0f);
            store_raw_float(row + kEmbd + 4, 10.0f);
            store_raw_float(row + kEmbd + 6, 1.0f);
        }
        std::memcpy(shared_workload.weight_bo.map<void *>(), shared.packed.data(),
                    shared.packed.size());
        shared_workload.input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        shared_workload.weight_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        projection.run(kernel);
        shared_workload.run(kernel);

        projection.projection_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        const auto * projection_output =
            projection.projection_bo.map<const float *>();
        std::vector<float> projection_actual(
            static_cast<size_t>(kBatch) * kProjectionN);
        std::vector<float> projection_expected(projection_actual.size());
        for (int token = 0; token < kBatch; ++token) {
            if (!ember::xdna2::q8_gemm_raw_reference(
                    projection_raw.data(), projection_raw.size(),
                    projection_values.data() + static_cast<size_t>(token) * kEmbd,
                    kEmbd, kProjectionN,
                    projection_expected.data() +
                        static_cast<size_t>(token) * kProjectionN))
                throw std::runtime_error("projection reference failed");
            for (int output = 0; output < kProjectionN; ++output) {
                const int row = output / ember::xdna2::kQ8OutputsPerRow;
                const int within_row = output % ember::xdna2::kQ8OutputsPerRow;
                const int column = within_row / ember::xdna2::kQ8TileN;
                const int lane = within_row % ember::xdna2::kQ8TileN;
                const size_t source =
                    static_cast<size_t>(row * ember::xdna2::kQ8AieColumns +
                                        column) * kBatch * kPacketBf16 / 2 +
                    static_cast<size_t>(token) * ember::xdna2::kQ8TileN + lane;
                projection_actual[static_cast<size_t>(token) * kProjectionN +
                                  output] = projection_output[source];
            }
        }
        const Metrics projection_metrics =
            compare(projection_actual, projection_expected);

        std::vector<float> shared_expected(static_cast<size_t>(kBatch) * kEmbd);
        for (int token = 0; token < kBatch; ++token) {
            const float * row = shared_values.data() +
                static_cast<size_t>(token) * kEmbd;
            std::vector<float> gate(kFf), up(kFf), hidden(kFf), result(kEmbd);
            if (!ember::xdna2::q8_gemm_raw_reference(
                    shared.gate.data(), shared.gate.size(), row, kEmbd, kFf,
                    gate.data()) ||
                !ember::xdna2::q8_gemm_raw_reference(
                    shared.up.data(), shared.up.size(), row, kEmbd, kFf,
                    up.data()))
                throw std::runtime_error("shared gate/up reference failed");
            for (int lane = 0; lane < kFf; ++lane) {
                const float g = std::min(gate[static_cast<size_t>(lane)], 10.0f);
                const float u = std::max(-10.0f,
                    std::min(up[static_cast<size_t>(lane)], 10.0f));
                hidden[static_cast<size_t>(lane)] = bf16_round(
                    (g / (1.0f + exp_approx(-g))) * u);
            }
            if (!ember::xdna2::q8_gemm_raw_reference(
                    shared.down.data(), shared.down.size(), hidden.data(),
                    kFf, kEmbd, result.data()))
                throw std::runtime_error("shared down reference failed");
            std::copy(result.begin(), result.end(),
                      shared_expected.begin() + static_cast<size_t>(token) * kEmbd);
        }
        shared_workload.staging_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        const auto * staging =
            shared_workload.staging_bo.map<const uint16_t *>();
        std::vector<float> shared_actual(static_cast<size_t>(kBatch) * kEmbd);
        for (int row = 0; row < 4; ++row) {
            for (int column = 0; column < 8; ++column) {
                const size_t core = static_cast<size_t>(row * 8 + column);
                for (int token = 0; token < kBatch; ++token) {
                    const float * packet = reinterpret_cast<const float *>(
                        staging + core * kBatch * kPacketBf16 +
                        static_cast<size_t>(token) * kPacketBf16);
                    for (int group = 0; group < 2; ++group) {
                        for (int lane = 0; lane < 64; ++lane) {
                            const size_t output =
                                static_cast<size_t>(token) * kEmbd +
                                static_cast<size_t>(group * 2048 + row * 512 +
                                                    column * 64 + lane);
                            shared_actual[output] = packet[group * 64 + lane];
                        }
                    }
                }
            }
        }
        const Metrics shared_metrics = compare(shared_actual, shared_expected);

        const double projection_ms = time_runs(projection, kernel, repeats);
        const double shared_ms = time_runs(shared_workload, kernel, repeats);
        const auto cycle_begin = Clock::now();
        for (int i = 0; i < repeats; ++i) {
            projection.run(kernel);
            shared_workload.run(kernel);
        }
        const double cycle_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - cycle_begin).count() / repeats;
        const double overhead = cycle_ms - projection_ms - shared_ms;

        std::printf(
            "dspark_resident_q8 projection_ms=%.6f shared_ms=%.6f "
            "cycle_ms=%.6f overhead_ms=%.6f repeats=%d\n",
            projection_ms, shared_ms, cycle_ms, overhead, repeats);
        std::printf(
            "dspark_resident_q8 projection_max_abs=%.8g "
            "projection_cosine=%.10f shared_max_abs=%.8g "
            "shared_cosine=%.10f\n",
            projection_metrics.max_abs, projection_metrics.cosine,
            shared_metrics.max_abs, shared_metrics.cosine);
        const bool pass = projection_metrics.cosine >= 0.99999 &&
            projection_metrics.max_abs <= 0.001f &&
            shared_metrics.cosine >= 0.99999 &&
            shared_metrics.max_abs <= 0.01f;
        std::printf("DSPARK_RESIDENT_Q8_VALIDATION_%s\n",
                    pass ? "PASS" : "FAIL");
        return pass ? 0 : 1;
    } catch (const std::exception & exception) {
        std::fprintf(stderr, "resident Q8 validation failed: %s\n",
                     exception.what());
        return 1;
    }
}
