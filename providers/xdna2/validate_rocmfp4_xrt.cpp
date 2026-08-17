// Hardware validator for the five-row ROCMFP4_FAST DSpark expert image.
//
// This is intentionally separate from the target-model ROCMFP2 provider. It
// proves compact signed-codebook decode, packing, five-row reuse, fused SwiGLU,
// route masking, resident expert sub-buffer selection, and XRT execution before
// the draft provider grows a full three-layer graph.

#include "rocmfp4_pack.h"
#include "rocmfp4_route_plan.h"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_kernel.h>

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
constexpr size_t kPageBytes = 4096;

size_t page_aligned(size_t bytes) {
    return (bytes + kPageBytes - 1) / kPageBytes * kPageBytes;
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
        const bool route_aware = std::string(argv[1]).find("expert_v8_") !=
                                 std::string::npos;
        int active_rows = kBatch;
        if (const char * raw = std::getenv("EMBER_XDNA_ROUTE_ROWS")) {
            char * end = nullptr;
            const long parsed = std::strtol(raw, &end, 10);
            if (end == raw || *end != '\0' || parsed < 1 || parsed > kBatch)
                throw std::runtime_error("EMBER_XDNA_ROUTE_ROWS must be in [1,5]");
            active_rows = static_cast<int>(parsed);
        }
        if (!route_aware && active_rows != kBatch)
            throw std::runtime_error("route masking requires a Gen8 image");
        int route_experts = 1;
        if (const char * raw = std::getenv("EMBER_XDNA_ROUTE_EXPERTS")) {
            char * end = nullptr;
            const long parsed = std::strtol(raw, &end, 10);
            if (end == raw || *end != '\0' || parsed < 1 || parsed > 64)
                throw std::runtime_error(
                    "EMBER_XDNA_ROUTE_EXPERTS must be in [1,64]");
            route_experts = static_cast<int>(parsed);
        }
        bool distinct_weights = false;
        if (const char * raw = std::getenv("EMBER_XDNA_DISTINCT_WEIGHTS")) {
            if (!std::strcmp(raw, "1")) distinct_weights = true;
            else if (std::strcmp(raw, "0"))
                throw std::runtime_error(
                    "EMBER_XDNA_DISTINCT_WEIGHTS must be 0 or 1");
        }
        if (distinct_weights && !route_aware)
            throw std::runtime_error("distinct weights require a Gen8 image");
        bool route_plan_mode = false;
        if (const char * raw = std::getenv("EMBER_XDNA_ROUTE_PLAN")) {
            if (!std::strcmp(raw, "1")) route_plan_mode = true;
            else if (std::strcmp(raw, "0"))
                throw std::runtime_error("EMBER_XDNA_ROUTE_PLAN must be 0 or 1");
        }
        if (route_plan_mode && !route_aware)
            throw std::runtime_error("route-plan validation requires Gen8");

        ember::xdna2::Rocmfp4RoutePlan route_plan;
        if (route_plan_mode) {
            int32_t selected[kBatch * ember::xdna2::kRocmfp4RouteMaxTopK];
            float weights[kBatch * ember::xdna2::kRocmfp4RouteMaxTopK];
            for (int token = 0; token < kBatch; ++token) {
                for (int slot = 0;
                     slot < ember::xdna2::kRocmfp4RouteMaxTopK; ++slot) {
                    const int index = token *
                        ember::xdna2::kRocmfp4RouteMaxTopK + slot;
                    selected[index] = index;  // worst case: 30 unique experts
                    weights[index] = static_cast<float>(slot + 1) / 14.0f;
                }
            }
            std::string route_error;
            if (!ember::xdna2::build_rocmfp4_route_plan(
                    selected, weights, kBatch,
                    ember::xdna2::kRocmfp4RouteMaxTopK,
                    ember::xdna2::kRocmfp4RouteMaxExperts,
                    route_plan, &route_error)) {
                throw std::runtime_error(route_error);
            }
            route_experts = static_cast<int>(route_plan.runs.size());
            active_rows = 1;
        }
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

        std::vector<float> expected(static_cast<size_t>(kBatch) * kEmbd, 0.0f);
        for (int token = 0; token < kBatch; ++token) {
            const float * row = input.data() + static_cast<size_t>(token) * kEmbd;
            std::vector<float> gate_out(kFf), up_out(kFf);
            if (!ember::xdna2::rocmfp4_gemm_raw_reference(
                    gate.data(), gate.size(), row, kEmbd, kFf,
                    gate_scale, gate_out.data()) ||
                !ember::xdna2::rocmfp4_gemm_raw_reference(
                    up.data(), up.size(), row, kEmbd, kFf,
                    up_scale, up_out.data())) {
                throw std::runtime_error("gate/up reference failed");
            }
            const int reference_runs = route_plan_mode ? route_experts : 1;
            for (int run = 0; run < reference_runs; ++run) {
                const float route = route_plan_mode
                    ? route_plan.runs[static_cast<size_t>(run)]
                          .row_weights[static_cast<size_t>(token)]
                    : route_aware && token < active_rows
                        ? 0.25f + static_cast<float>(token) * 0.125f
                        : route_aware ? 0.0f : 1.0f;
                if (route == 0.0f) continue;
                std::vector<float> hidden(kFf), result(kEmbd);
                for (int lane = 0; lane < kFf; ++lane) {
                    const float g = std::min(gate_out[lane], clamp);
                    const float u = std::max(
                        -clamp, std::min(up_out[lane], clamp));
                    hidden[static_cast<size_t>(lane)] = bf16_round(
                        (g / (1.0f + exp_approx(-g))) * u * route);
                }
                if (!ember::xdna2::rocmfp4_gemm_raw_reference(
                        down.data(), down.size(), hidden.data(), kFf, kEmbd,
                        down_scale, result.data())) {
                    throw std::runtime_error("down reference failed");
                }
                float * expected_row = expected.data() +
                    static_cast<size_t>(token) * kEmbd;
                for (int lane = 0; lane < kEmbd; ++lane)
                    expected_row[lane] += result[static_cast<size_t>(lane)];
            }
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
        const size_t input_bytes = static_cast<size_t>(kBatch) *
            (kEmbd + 128) * sizeof(uint16_t);
        const size_t staging_bytes = static_cast<size_t>(kBatch) *
            8192 * sizeof(uint16_t);
        const size_t buffer_runs = route_plan_mode
            ? static_cast<size_t>(route_experts) : 1u;
        const size_t input_stride = page_aligned(input_bytes);
        const size_t staging_stride = page_aligned(staging_bytes);
        if (input_stride > SIZE_MAX / buffer_runs ||
            staging_stride > SIZE_MAX / buffer_runs)
            throw std::runtime_error("route buffer allocation overflow");
        xrt::bo input_parent(device, input_stride * buffer_runs,
                             XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
        std::vector<xrt::bo> input_views;
        input_views.reserve(buffer_runs);
        for (size_t run = 0; run < buffer_runs; ++run)
            input_views.emplace_back(input_parent, input_bytes,
                                     run * input_stride);
        const size_t resident_experts = distinct_weights
            ? static_cast<size_t>(route_experts) : 1u;
        if (packed.size() > SIZE_MAX / resident_experts)
            throw std::runtime_error("resident expert allocation overflow");
        xrt::bo weight_parent(device, packed.size() * resident_experts,
                              XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
        std::vector<xrt::bo> weight_views;
        weight_views.reserve(resident_experts);
        for (size_t expert = 0; expert < resident_experts; ++expert)
            weight_views.emplace_back(weight_parent, packed.size(),
                                      expert * packed.size());
        xrt::bo staging_parent(device, staging_stride * buffer_runs,
                               XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
        std::vector<xrt::bo> staging_views;
        staging_views.reserve(buffer_runs);
        for (size_t run = 0; run < buffer_runs; ++run)
            staging_views.emplace_back(staging_parent, staging_bytes,
                                       run * staging_stride);
        xrt::bo dummy6(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
        xrt::bo dummy7(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));

        auto * input_base = input_parent.map<uint8_t *>();
        for (size_t run = 0; run < buffer_runs; ++run) {
            auto * device_input = reinterpret_cast<uint16_t *>(
                input_base + run * input_stride);
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
                const float route = route_plan_mode
                    ? route_plan.runs[run]
                          .row_weights[static_cast<size_t>(token)]
                    : route_aware && token < active_rows
                        ? 0.25f + static_cast<float>(token) * 0.125f
                        : route_aware ? 0.0f : 1.0f;
                store_raw_float(destination + kEmbd + 6, route);
            }
        }
        auto * resident_weights = weight_parent.map<uint8_t *>();
        for (size_t expert = 0; expert < resident_experts; ++expert)
            std::memcpy(resident_weights + expert * packed.size(),
                        packed.data(), packed.size());
        input_parent.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        weight_parent.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto configure_run = [&](int expert) {
            xrt::run run(kernel);
            run.set_arg(0, 3);
            run.set_arg(1, instruction_bo);
            run.set_arg(2, static_cast<uint32_t>(instructions.size()));
            const size_t buffer_view = route_plan_mode
                ? static_cast<size_t>(expert) : 0u;
            run.set_arg(3, input_views[buffer_view]);
            const size_t view = distinct_weights
                ? static_cast<size_t>(expert) : 0u;
            run.set_arg(4, weight_views[view]);
            run.set_arg(5, staging_views[buffer_view]);
            run.set_arg(6, dummy6);
            run.set_arg(7, dummy7);
            return run;
        };
        auto run_once = [&]() {
            if (route_experts == 1) {
                xrt::run run = configure_run(0);
                run.start();
                if (run.wait() != ERT_CMD_STATE_COMPLETED)
                    throw std::runtime_error("ROCMFP4 command did not complete");
                return;
            }
            xrt::runlist list(context);
            for (int expert = 0; expert < route_experts; ++expert)
                list.add(configure_run(expert));
            list.execute();
            list.wait();
        };
        run_once();
        constexpr int timed_runs = 20;
        const auto begin = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < timed_runs; ++iteration) run_once();
        const double milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count() / timed_runs;
        const auto read_begin = std::chrono::steady_clock::now();
        staging_parent.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        std::vector<float> actual(static_cast<size_t>(kBatch) * kEmbd, 0.0f);
        const auto * staging_base = staging_parent.map<const uint8_t *>();
        const size_t output_runs = route_plan_mode ? buffer_runs : 1u;
        for (size_t run = 0; run < output_runs; ++run) {
            const auto * staging = reinterpret_cast<const uint16_t *>(
                staging_base + run * staging_stride);
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 8; ++col) {
                    const size_t core = static_cast<size_t>(row * 8 + col);
                    for (int token = 0; token < kBatch; ++token) {
                        const float * packet = reinterpret_cast<const float *>(
                            staging + core * kBatch * kPacketBf16 +
                            static_cast<size_t>(token) * kPacketBf16);
                        for (int group = 0; group < 2; ++group) {
                            for (int lane = 0; lane < 64; ++lane) {
                                const size_t output =
                                    static_cast<size_t>(token) * kEmbd +
                                    static_cast<size_t>(group * 2048 +
                                        row * 8 * 64 + col * 64 + lane);
                                actual[output] +=
                                    packet[group * 64 + lane] * down_scale;
                            }
                        }
                    }
                }
            }
        }
        const double read_accumulate_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - read_begin).count();

        const Metrics metrics = compare(actual, expected);
        std::printf("gen%d tokens=5 active_rows=%d route_experts=%d "
                    "route_plan=%s weight_mode=%s resident_weight_bytes=%zu "
                    "packed_bytes=%zu sequence_ms=%.6f per_expert_ms=%.6f "
                    "read_accumulate_ms=%.6f "
                    "max_abs=%.8g mean_abs=%.8g cosine=%.10f max_index=%d\n",
                    route_aware ? 8 : 7, active_rows, route_experts,
                    route_plan_mode ? "yes" : "no",
                    distinct_weights ? "distinct" : "cache-hot",
                    packed.size() * resident_experts,
                    packed.size(), milliseconds,
                    milliseconds / static_cast<double>(route_experts),
                    read_accumulate_ms, metrics.max_abs,
                    metrics.mean_abs, metrics.cosine, metrics.max_index);
        return metrics.cosine >= 0.99999 && metrics.max_abs <= 0.01f ? 0 : 1;
    } catch (const std::exception & exception) {
        std::fprintf(stderr, "ROCMFP4 validation failed: %s\n", exception.what());
        return 1;
    }
}
