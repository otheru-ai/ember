// Hardware validator for the five-row ROCMFP4_FAST DSpark expert image.
//
// This is intentionally separate from the target-model ROCMFP2 provider. It
// proves compact signed-codebook decode, packing, five-row reuse, fused SwiGLU,
// and XRT execution before the draft provider grows a full three-layer graph.

#include "rocmfp4_pack.h"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kBatch = 5;
constexpr int kEmbd = 4096;
constexpr int kFf = 2048;
constexpr int kPacketBf16 = 256;

uint16_t float_to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
}

float bf16_round(float value) {
    const uint32_t bits = static_cast<uint32_t>(float_to_bf16(value)) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

void store_raw_float(uint16_t * destination, float value) {
    std::memcpy(destination, &value, sizeof(value));
}

float exp_approx(float x) {
    if (x < -87.0f) return 0.0f;
    if (x > 87.0f) x = 87.0f;
    constexpr float kInvLn2 = 1.4426950408889634f;
    constexpr float kLn2 = 0.6931471805599453f;
    const float scaled = x * kInvLn2;
    const int exponent = static_cast<int>(
        scaled + (scaled < 0.0f ? -0.5f : 0.5f));
    const float r = x - static_cast<float>(exponent) * kLn2;
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
        throw std::runtime_error("short instruction stream read");
    return words;
}

void set_code(uint8_t * block, int lane, uint8_t code) {
    uint8_t & byte = lane < 16 ? block[lane] : block[lane - 16];
    const unsigned shift = lane < 16 ? 0u : 4u;
    byte = static_cast<uint8_t>((byte & ~(0x0fu << shift)) |
                                ((code & 0x0fu) << shift));
}

void fill_sparse_projection(std::vector<uint8_t> & raw, int k, int n,
                            uint32_t salt) {
    std::fill(raw.begin(), raw.end(), 0);
    const int blocks_per_output = k / ember::xdna2::kRocmfp4BlockWeights;
    for (int output = 0; output < n; ++output) {
        const int input = static_cast<int>(
            (static_cast<uint32_t>(output) * 109u + salt) %
            static_cast<uint32_t>(k));
        const int block_index = input / ember::xdna2::kRocmfp4BlockWeights;
        const int lane = input % ember::xdna2::kRocmfp4BlockWeights;
        uint8_t * block = raw.data() +
            (static_cast<size_t>(output) * blocks_per_output + block_index) *
                ember::xdna2::kRocmfp4BlockBytes;
        set_code(block, lane,
                 static_cast<uint8_t>(1u +
                    (static_cast<uint32_t>(output) + salt) % 15u));
        block[16] = 0x40;  // UE4M3 1.0
    }
}

struct Metrics {
    float max_abs = 0.0f;
    double mean_abs = 0.0;
    double cosine = 0.0;
    int max_index = -1;
};

Metrics compare(const std::vector<float> & actual,
                const std::vector<float> & expected) {
    Metrics metrics;
    double absolute_sum = 0.0;
    double dot = 0.0;
    double actual_squared = 0.0;
    double expected_squared = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float error = std::fabs(actual[i] - expected[i]);
        absolute_sum += error;
        dot += static_cast<double>(actual[i]) * expected[i];
        actual_squared += static_cast<double>(actual[i]) * actual[i];
        expected_squared += static_cast<double>(expected[i]) * expected[i];
        if (error > metrics.max_abs) {
            metrics.max_abs = error;
            metrics.max_index = static_cast<int>(i);
        }
    }
    metrics.mean_abs = absolute_sum / static_cast<double>(actual.size());
    if (actual_squared > 0.0 && expected_squared > 0.0)
        metrics.cosine = dot / std::sqrt(actual_squared * expected_squared);
    return metrics;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s IMAGE.xclbin IMAGE.insts\n", argv[0]);
        return 2;
    }
    try {
        constexpr float gate_scale = 0.125f;
        constexpr float up_scale = 0.125f;
        constexpr float down_scale = 0.25f;
        constexpr float clamp = 10.0f;

        std::vector<uint8_t> gate(
            ember::xdna2::rocmfp4_projection_bytes(kEmbd, kFf));
        std::vector<uint8_t> up(gate.size());
        std::vector<uint8_t> down(
            ember::xdna2::rocmfp4_projection_bytes(kFf, kEmbd));
        fill_sparse_projection(gate, kEmbd, kFf, 0x1357u);
        fill_sparse_projection(up, kEmbd, kFf, 0x2468u);
        fill_sparse_projection(down, kFf, kEmbd, 0x369cu);

        std::vector<uint8_t> packed;
        std::string pack_error;
        if (!ember::xdna2::pack_rocmfp4_expert_v7(
                gate.data(), gate.size(), up.data(), up.size(),
                down.data(), down.size(), packed, &pack_error)) {
            throw std::runtime_error(pack_error);
        }

        std::vector<float> input(static_cast<size_t>(kBatch) * kEmbd);
        for (int token = 0; token < kBatch; ++token) {
            for (int lane = 0; lane < kEmbd; ++lane) {
                input[static_cast<size_t>(token) * kEmbd + lane] =
                    bf16_round(static_cast<float>(
                        (lane * 37 + token * 53) % 257 - 128) / 128.0f);
            }
        }

        std::vector<float> expected;
        expected.reserve(static_cast<size_t>(kBatch) * kEmbd);
        for (int token = 0; token < kBatch; ++token) {
            const float * row = input.data() + static_cast<size_t>(token) * kEmbd;
            std::vector<float> gate_out(kFf), up_out(kFf), hidden(kFf), result(kEmbd);
            if (!ember::xdna2::rocmfp4_gemm_raw_reference(
                    gate.data(), gate.size(), row, kEmbd, kFf,
                    gate_scale, gate_out.data()) ||
                !ember::xdna2::rocmfp4_gemm_raw_reference(
                    up.data(), up.size(), row, kEmbd, kFf,
                    up_scale, up_out.data())) {
                throw std::runtime_error("gate/up reference failed");
            }
            for (int lane = 0; lane < kFf; ++lane) {
                const float g = std::min(gate_out[lane], clamp);
                const float u = std::max(-clamp, std::min(up_out[lane], clamp));
                hidden[lane] = bf16_round(
                    (g / (1.0f + exp_approx(-g))) * u);
            }
            if (!ember::xdna2::rocmfp4_gemm_raw_reference(
                    down.data(), down.size(), hidden.data(), kFf, kEmbd,
                    down_scale, result.data())) {
                throw std::runtime_error("down reference failed");
            }
            expected.insert(expected.end(), result.begin(), result.end());
        }

        xrt::device device(0);
        xrt::xclbin xclbin{std::string(argv[1])};
        const auto kernels = xclbin.get_kernels();
        const auto found = std::find_if(kernels.begin(), kernels.end(),
            [](const xrt::xclbin::kernel & kernel) {
                return kernel.get_name().rfind("MLIR_AIE", 0) == 0;
            });
        if (found == kernels.end()) throw std::runtime_error("AIE kernel absent");
        device.register_xclbin(xclbin);
        xrt::hw_context context(device, xclbin.get_uuid());
        xrt::kernel kernel(context, found->get_name());
        const std::vector<uint32_t> instructions = read_instructions(argv[2]);
        xrt::bo instruction_bo(device,
            instructions.size() * sizeof(uint32_t), XCL_BO_FLAGS_CACHEABLE,
            kernel.group_id(1));
        std::memcpy(instruction_bo.map<void *>(), instructions.data(),
                    instructions.size() * sizeof(uint32_t));
        instruction_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        xrt::bo input_bo(device,
            static_cast<size_t>(kBatch) * (kEmbd + 128) * sizeof(uint16_t),
            XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
        xrt::bo weight_bo(device, packed.size(), XRT_BO_FLAGS_HOST_ONLY,
                          kernel.group_id(4));
        xrt::bo staging_bo(device,
            static_cast<size_t>(kBatch) * 8192 * sizeof(uint16_t),
            XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
        xrt::bo dummy6(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
        xrt::bo dummy7(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));

        uint16_t * device_input = input_bo.map<uint16_t *>();
        for (int token = 0; token < kBatch; ++token) {
            uint16_t * destination = device_input +
                static_cast<size_t>(token) * (kEmbd + 128);
            for (int lane = 0; lane < kEmbd; ++lane) {
                destination[lane] = float_to_bf16(
                    input[static_cast<size_t>(token) * kEmbd + lane]);
            }
            std::fill(destination + kEmbd, destination + kEmbd + 128, 0);
            store_raw_float(destination + kEmbd, gate_scale);
            store_raw_float(destination + kEmbd + 2, up_scale);
            store_raw_float(destination + kEmbd + 4, clamp);
        }
        std::memcpy(weight_bo.map<void *>(), packed.data(), packed.size());
        input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        weight_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto run_once = [&]() {
            xrt::run run(kernel);
            run.set_arg(0, 3);
            run.set_arg(1, instruction_bo);
            run.set_arg(2, static_cast<uint32_t>(instructions.size()));
            run.set_arg(3, input_bo);
            run.set_arg(4, weight_bo);
            run.set_arg(5, staging_bo);
            run.set_arg(6, dummy6);
            run.set_arg(7, dummy7);
            run.start();
            if (run.wait() != ERT_CMD_STATE_COMPLETED)
                throw std::runtime_error("Gen7 command did not complete");
        };
        run_once();
        constexpr int timed_runs = 20;
        const auto begin = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < timed_runs; ++iteration) run_once();
        const double milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count() / timed_runs;
        staging_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        std::vector<float> actual(static_cast<size_t>(kBatch) * kEmbd);
        const uint16_t * staging = staging_bo.map<uint16_t *>();
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 8; ++col) {
                const size_t core = static_cast<size_t>(row * 8 + col);
                for (int token = 0; token < kBatch; ++token) {
                    const float * packet = reinterpret_cast<const float *>(
                        staging + core * kBatch * kPacketBf16 +
                        static_cast<size_t>(token) * kPacketBf16);
                    for (int group = 0; group < 2; ++group) {
                        for (int lane = 0; lane < 64; ++lane) {
                            const size_t output = static_cast<size_t>(token) * kEmbd +
                                static_cast<size_t>(group * 2048 +
                                    row * 8 * 64 + col * 64 + lane);
                            actual[output] = packet[group * 64 + lane] * down_scale;
                        }
                    }
                }
            }
        }

        const Metrics metrics = compare(actual, expected);
        std::printf("gen7 tokens=5 packed_bytes=%zu warm_ms=%.6f "
                    "max_abs=%.8g mean_abs=%.8g cosine=%.10f max_index=%d\n",
                    packed.size(), milliseconds, metrics.max_abs,
                    metrics.mean_abs, metrics.cosine, metrics.max_index);
        return metrics.cosine >= 0.99999 && metrics.max_abs <= 0.01f ? 0 : 1;
    } catch (const std::exception & exception) {
        std::fprintf(stderr, "Gen7 validation failed: %s\n", exception.what());
        return 1;
    }
}
