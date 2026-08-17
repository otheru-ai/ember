#include "q8_0_pack.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;
#define CHECK(condition, message) do { \
    if (condition) ++g_pass; \
    else { ++g_fail; std::printf("  FAIL: %s\n", message); } \
} while (0)

static uint16_t float_to_fp16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffu;
    if (exponent <= 0) return static_cast<uint16_t>(sign);
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    mantissa += 0x0fffu + ((mantissa >> 13) & 1u);
    if (mantissa & 0x800000u) {
        mantissa = 0;
        if (++exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    }
    return static_cast<uint16_t>(sign |
        (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
}

static float bf16_to_float(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

int main() {
    using namespace ember::xdna2;
    std::printf("ember XDNA Q8 pack tests\n");
    CHECK(q8_projection_bytes(128, 2048) == 278528,
          "raw Q8 projection size");
    CHECK(q8_packed_projection_bytes(128, 2048) == 524288,
          "BF16 packed projection size");
    CHECK(q8_corrected_packed_projection_bytes(128, 2048) == 1048576,
          "corrected packed projection size");
    CHECK(q8_expert_v1_bytes() == 50331648,
          "three-projection expert size");
    CHECK(q8_expert_v2_bytes() == 100663296,
          "corrected expert size");
    CHECK(!q8_supported_shape(127, 2048) &&
          !q8_supported_shape(128, 1024),
          "unsupported tile shapes rejected");

    constexpr int k = 128;
    constexpr int n = 2048;
    std::vector<uint8_t> raw(q8_projection_bytes(k, n), 0);
    const int blocks_per_output = k / kQ8BlockWeights;
    for (int output = 0; output < n; ++output) {
        for (int block = 0; block < blocks_per_output; ++block) {
            uint8_t * q = raw.data() +
                (static_cast<size_t>(output) *
                     static_cast<size_t>(blocks_per_output) +
                 static_cast<size_t>(block)) *
                kQ8BlockBytes;
            const uint16_t scale = float_to_fp16(
                static_cast<float>((output + block) % 9 + 1) / 256.0f);
            std::memcpy(q, &scale, sizeof(scale));
            for (int lane = 0; lane < kQ8BlockWeights; ++lane) {
                q[2 + lane] = static_cast<uint8_t>(static_cast<int8_t>(
                    (output * 17 + block * 13 + lane * 7) % 255 - 127));
            }
        }
    }
    std::vector<uint8_t> packed;
    std::string error;
    CHECK(pack_q8_gemm_bf16(raw.data(), raw.size(), k, n, packed, &error),
          "valid projection packs");
    CHECK(packed.size() == q8_packed_projection_bytes(k, n),
          "packed byte count");
    std::vector<float> input(k);
    for (int i = 0; i < k; ++i)
        input[static_cast<size_t>(i)] =
            static_cast<float>((i * 29) % 63 - 31) / 32.0f;
    std::vector<float> raw_output(n), packed_output(n);
    CHECK(q8_gemm_raw_reference(raw.data(), raw.size(), input.data(), k, n,
                                raw_output.data()),
          "raw reference evaluates");
    CHECK(q8_gemm_packed_bf16_reference(
              packed.data(), packed.size(), input.data(), k, n,
              packed_output.data()),
          "packed reference evaluates");
    float max_abs = 0.0f;
    double dot = 0.0;
    double raw_sq = 0.0;
    double packed_sq = 0.0;
    for (int i = 0; i < n; ++i) {
        const size_t index = static_cast<size_t>(i);
        max_abs = std::max(max_abs,
                           std::fabs(raw_output[index] - packed_output[index]));
        dot += static_cast<double>(raw_output[index]) * packed_output[index];
        raw_sq += static_cast<double>(raw_output[index]) * raw_output[index];
        packed_sq += static_cast<double>(packed_output[index]) *
                     packed_output[index];
    }
    const double cosine = dot / std::sqrt(raw_sq * packed_sq);
    CHECK(cosine > 0.99999 && max_abs < 0.05f,
          "one-time BF16 dequant retains Q8 projection fidelity");

    std::vector<uint8_t> corrected;
    CHECK(pack_q8_gemm_corrected_bf16(
              raw.data(), raw.size(), k, n, corrected, &error),
          "corrected projection packs");
    CHECK(corrected.size() == q8_corrected_packed_projection_bytes(k, n),
          "corrected byte count");
    bool corrected_layout = true;
    for (int output = 0; output < n && corrected_layout; ++output) {
        const int row = output / (kQ8AieColumns * kQ8TileN);
        const int column = (output / kQ8TileN) % kQ8AieColumns;
        const int output_lane = output % kQ8TileN;
        const size_t tile_index =
            (static_cast<size_t>(column) * kQ8AieRows) +
            static_cast<size_t>(row);
        const auto * high = reinterpret_cast<const uint16_t *>(
            corrected.data() + tile_index * kQ8CorrectedTileBytes);
        const uint16_t * low = high + static_cast<size_t>(k) * kQ8TileN;
        for (int input_lane = 0; input_lane < k; ++input_lane) {
            const int block = input_lane / kQ8BlockWeights;
            const int lane = input_lane % kQ8BlockWeights;
            const uint8_t * q = raw.data() +
                (static_cast<size_t>(output) * blocks_per_output +
                 static_cast<size_t>(block)) * kQ8BlockBytes;
            const size_t index = static_cast<size_t>(input_lane) * kQ8TileN +
                static_cast<size_t>(output_lane);
            const float reconstructed = bf16_to_float(high[index]) +
                bf16_to_float(low[index]);
            const float expected_weight = static_cast<float>(
                static_cast<int8_t>(q[2 + lane])) *
                static_cast<float>((output + block) % 9 + 1) / 256.0f;
            if (std::fabs(reconstructed - expected_weight) > 1.0e-5f) {
                corrected_layout = false;
                break;
            }
        }
    }
    CHECK(corrected_layout,
          "corrected planes reconstruct dequantized weights");

    CHECK(!pack_q8_gemm_bf16(raw.data(), raw.size() - 1, k, n,
                             packed, &error) &&
              error.find("shorter") != std::string::npos,
          "short projection rejected");
    CHECK(!pack_q8_gemm_bf16(nullptr, 0, k, n, packed, &error),
          "null projection rejected");
    CHECK(!pack_q8_gemm_corrected_bf16(
              raw.data(), raw.size() - 1, k, n, corrected, &error),
          "short corrected projection rejected");
    std::printf("  max_abs=%.8g cosine=%.10f\n", max_abs, cosine);
    std::printf("──────────────────────────────\n  %d passed, %d failed\n",
                g_pass, g_fail);
    return g_fail ? 1 : 0;
}
