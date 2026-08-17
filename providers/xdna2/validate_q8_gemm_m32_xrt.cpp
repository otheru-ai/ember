// Hardware validator for the 128-row compensated DSpark Q8_0 GEMM.
// Inputs, corrected weights, and outputs are pre-tiled in the same 4x8x8
// AIE mmul order used by AMD IRON's optimized AIE2P GEMM kernel.

#include "q8_0_pack.h"
#include "q8_model_weights.h"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

#include <algorithm>
#include <array>
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

constexpr int kBatch = 128;
constexpr int kMPerRow = 32;
constexpr int kMmulM = 4;
constexpr int kMmulK = 8;
constexpr int kMmulN = 8;
constexpr int kOutputTile = kMPerRow * ember::xdna2::kQ8TileN;
constexpr std::array<int, 8> kReferenceTokens =
    {0, 31, 32, 63, 64, 95, 96, 127};

int parse_dimension(const char * text, const char * label) {
    char * end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value <= 0 || value > 131072)
        throw std::runtime_error(std::string("invalid ") + label);
    return static_cast<int>(value);
}

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

uint16_t float_to_fp16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int exponent = static_cast<int>((bits >> 23) & 0xffu) - 112;
    uint32_t mantissa = bits & 0x7fffffu;
    if (exponent <= 0) return static_cast<uint16_t>(sign);
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    mantissa += 0x0fffu + ((mantissa >> 13) & 1u);
    if (mantissa & 0x800000u) {
        mantissa = 0;
        if (++exponent >= 31)
            return static_cast<uint16_t>(sign | 0x7c00u);
    }
    return static_cast<uint16_t>(
        sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
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

void fill_projection(std::vector<uint8_t> & raw, int k, int n) {
    const int blocks = k / ember::xdna2::kQ8BlockWeights;
    for (int output = 0; output < n; ++output) {
        for (int block = 0; block < blocks; ++block) {
            uint8_t * q = raw.data() +
                (static_cast<size_t>(output) * static_cast<size_t>(blocks) +
                 static_cast<size_t>(block)) * ember::xdna2::kQ8BlockBytes;
            const uint16_t scale = float_to_fp16(
                static_cast<float>((output + block) % 7 + 1) / 1024.0f);
            std::memcpy(q, &scale, sizeof(scale));
            for (int lane = 0; lane < ember::xdna2::kQ8BlockWeights; ++lane) {
                const int value =
                    (output * 17 + block * 13 + lane * 7) % 31 - 15;
                q[2 + lane] = static_cast<uint8_t>(
                    static_cast<int8_t>(value));
            }
        }
    }
}

struct Metrics {
    float max_abs = 0.0f;
    double mean_abs = 0.0;
    double cosine = 0.0;
};

Metrics compare(const std::vector<float> & actual,
                const std::vector<float> & expected) {
    Metrics result;
    double error_sum = 0.0, dot = 0.0, aa = 0.0, ee = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float error = std::fabs(actual[i] - expected[i]);
        result.max_abs = std::max(result.max_abs, error);
        error_sum += error;
        dot += static_cast<double>(actual[i]) * expected[i];
        aa += static_cast<double>(actual[i]) * actual[i];
        ee += static_cast<double>(expected[i]) * expected[i];
    }
    result.mean_abs = error_sum / static_cast<double>(actual.size());
    if (aa > 0.0 && ee > 0.0) result.cosine = dot / std::sqrt(aa * ee);
    return result;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 5 && argc != 7) {
        std::fprintf(stderr,
            "usage: %s IMAGE.xclbin IMAGE.insts K N [DRAFT.gguf TENSOR]\n",
            argv[0]);
        return 2;
    }
    try {
        const int k = parse_dimension(argv[3], "K");
        const int n = parse_dimension(argv[4], "N");
        if (k % 4096 || n % ember::xdna2::kQ8OutputsPerRow)
            throw std::runtime_error("M=128 GEMM requires K%4096=N%512=0");
        const int k_tiles = k / ember::xdna2::kQ8TileK;
        const int groups = n / ember::xdna2::kQ8OutputsPerRow;
        const bool trained = argc == 7;

        std::vector<uint8_t> raw;
        std::vector<uint8_t> packed;
        std::string error;
        if (trained) {
            ember::xdna2::Q8ModelProjection projection;
            if (!ember::xdna2::load_q8_model_projection(
                    argv[5], argv[6], k, n, projection, &error))
                throw std::runtime_error(error);
            raw = std::move(projection.raw);
        } else {
            raw.resize(ember::xdna2::q8_projection_bytes(k, n));
            fill_projection(raw, k, n);
        }
        if (!ember::xdna2::pack_q8_gemm_m32_corrected_bf16(
                raw.data(), raw.size(), k, n, packed, &error))
            throw std::runtime_error(error);

        std::vector<float> input(static_cast<size_t>(kBatch) * k);
        for (int token = 0; token < kBatch; ++token) {
            for (int lane = 0; lane < k; ++lane) {
                input[static_cast<size_t>(token) * static_cast<size_t>(k) +
                      static_cast<size_t>(lane)] = bf16_round(
                    static_cast<float>((lane * 37 + token * 53) % 257 - 128) /
                    128.0f);
            }
        }
        std::vector<float> expected(kReferenceTokens.size() *
                                    static_cast<size_t>(n));
        for (size_t sample = 0; sample < kReferenceTokens.size(); ++sample) {
            const int token = kReferenceTokens[sample];
            if (!ember::xdna2::q8_gemm_raw_reference(
                    raw.data(), raw.size(),
                    input.data() + static_cast<size_t>(token) * k, k, n,
                    expected.data() + sample * static_cast<size_t>(n)))
                throw std::runtime_error("Q8 reference failed");
        }

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
        const std::vector<uint32_t> instructions = read_instructions(argv[2]);
        xrt::bo instruction_bo(device, instructions.size() * sizeof(uint32_t),
            XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
        std::memcpy(instruction_bo.map<void *>(), instructions.data(),
                    instructions.size() * sizeof(uint32_t));
        instruction_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        xrt::bo input_bo(device,
            static_cast<size_t>(groups) * kBatch * k * sizeof(uint16_t),
            XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
        xrt::bo weight_bo(device, packed.size(), XRT_BO_FLAGS_HOST_ONLY,
                          kernel.group_id(4));
        xrt::bo output_bo(device,
            static_cast<size_t>(kBatch) * n * sizeof(float),
            XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
        xrt::bo dummy6(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
        xrt::bo dummy7(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));

        auto * blocked_input = input_bo.map<uint16_t *>();
        constexpr int m_blocks = kMPerRow / kMmulM;
        constexpr int k_blocks = ember::xdna2::kQ8TileK / kMmulK;
        for (int row = 0; row < ember::xdna2::kQ8AieRows; ++row) {
            for (int group = 0; group < groups; ++group) {
              for (int kt = 0; kt < k_tiles; ++kt) {
                for (int mb = 0; mb < m_blocks; ++mb) {
                    for (int kb = 0; kb < k_blocks; ++kb) {
                        for (int ir = 0; ir < kMmulM; ++ir) {
                            for (int ik = 0; ik < kMmulK; ++ik) {
                                const int token = row * kMPerRow +
                                                  mb * kMmulM + ir;
                                const int lane = kt * ember::xdna2::kQ8TileK +
                                                 kb * kMmulK + ik;
                                size_t destination =
                                    static_cast<size_t>(
                                        (row * groups + group) * k_tiles +
                                        kt) *
                                        kMPerRow * ember::xdna2::kQ8TileK;
                                destination +=
                                    static_cast<size_t>(mb * k_blocks + kb) *
                                        kMmulM * kMmulK +
                                    static_cast<size_t>(ir * kMmulK + ik);
                                blocked_input[destination] = float_to_bf16(
                                    input[static_cast<size_t>(token) * k +
                                          static_cast<size_t>(lane)]);
                            }
                        }
                    }
                }
            }
            }
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
            run.set_arg(5, output_bo);
            run.set_arg(6, dummy6);
            run.set_arg(7, dummy7);
            run.start();
            if (run.wait() != ERT_CMD_STATE_COMPLETED)
                throw std::runtime_error("Q8 M=128 GEMM command failed");
        };
        run_once();
        constexpr int kTimedRuns = 20;
        const auto begin = std::chrono::steady_clock::now();
        for (int run = 0; run < kTimedRuns; ++run) run_once();
        const double milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count() / kTimedRuns;
        output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        const float * staging = output_bo.map<const float *>();
        std::vector<float> actual(expected.size());
        for (size_t sample = 0; sample < kReferenceTokens.size(); ++sample) {
            const int token = kReferenceTokens[sample];
            const int row = token / kMPerRow;
            const int local_token = token % kMPerRow;
            const int mb = local_token / kMmulM;
            const int ir = local_token % kMmulM;
            for (int group = 0; group < groups; ++group) {
                for (int col = 0; col < ember::xdna2::kQ8AieColumns; ++col) {
                    for (int lane = 0; lane < ember::xdna2::kQ8TileN;
                         ++lane) {
                        const int nb = lane / kMmulN;
                        const int jn = lane % kMmulN;
                        size_t source =
                            static_cast<size_t>(
                                (group * ember::xdna2::kQ8AieRows + row) *
                                    ember::xdna2::kQ8AieColumns + col) *
                                kOutputTile;
                        source +=
                            static_cast<size_t>(mb * 8 + nb) *
                                kMmulM * kMmulN +
                            static_cast<size_t>(ir * kMmulN + jn);
                        const size_t output =
                            sample * static_cast<size_t>(n) +
                            static_cast<size_t>(group *
                                ember::xdna2::kQ8OutputsPerRow +
                                col * ember::xdna2::kQ8TileN + lane);
                        actual[output] = staging[source];
                    }
                }
            }
        }
        const Metrics metrics = compare(actual, expected);
        std::printf(
            "q8_gemm_m32 mode=%s tensor=%s M=%d K=%d N=%d groups=%d "
            "packed_bytes=%zu sequence_ms=%.6f max_abs=%.8g "
            "mean_abs=%.8g cosine=%.10f\n",
            trained ? "trained" : "synthetic", trained ? argv[6] : "-",
            kBatch, k, n, groups, packed.size(), milliseconds,
            metrics.max_abs, metrics.mean_abs, metrics.cosine);
        return metrics.cosine >= 0.99999 && metrics.max_abs <= 0.05f ? 0 : 1;
    } catch (const std::exception & exception) {
        std::fprintf(stderr, "Q8 M=128 GEMM validation failed: %s\n",
                     exception.what());
        return 1;
    }
}
