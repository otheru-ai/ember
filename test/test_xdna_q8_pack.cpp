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
    CHECK(q8_supported_shape(128, 512) && q8_projection_rows(512) == 1 &&
          q8_supported_shape(128, 1024) && q8_projection_rows(1024) == 2 &&
          q8_supported_shape(128, 1536) && q8_projection_rows(1536) == 3 &&
          q8_projection_rows(2048) == 4,
          "narrow projections use only active AIE rows");
    CHECK(!q8_supported_shape(127, 2048) &&
          !q8_supported_shape(128, 256) &&
          !q8_supported_shape(128, 2560),
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

    constexpr int narrow_n = 512;
    std::vector<uint8_t> narrow_raw(
        q8_projection_bytes(k, narrow_n), 0);
    std::copy_n(raw.begin(), narrow_raw.size(),
              narrow_raw.begin());
    std::vector<uint8_t> narrow_packed;
    CHECK(pack_q8_gemm_bf16(narrow_raw.data(), narrow_raw.size(), k,
                            narrow_n, narrow_packed, &error) &&
              narrow_packed.size() ==
                  q8_packed_projection_bytes(k, narrow_n),
          "one-row projection packs without padding to four rows");
    std::vector<float> narrow_raw_output(narrow_n);
    std::vector<float> narrow_packed_output(narrow_n);
    CHECK(q8_gemm_raw_reference(narrow_raw.data(), narrow_raw.size(),
                                input.data(), k, narrow_n,
                                narrow_raw_output.data()) &&
          q8_gemm_packed_bf16_reference(
              narrow_packed.data(), narrow_packed.size(), input.data(), k,
              narrow_n, narrow_packed_output.data()),
          "one-row packed reference evaluates");
    double narrow_dot = 0.0, narrow_raw_sq = 0.0, narrow_packed_sq = 0.0;
    for (size_t i = 0; i < static_cast<size_t>(narrow_n); ++i) {
        narrow_dot += static_cast<double>(narrow_raw_output[i]) *
                      narrow_packed_output[i];
        narrow_raw_sq += static_cast<double>(narrow_raw_output[i]) *
                         narrow_raw_output[i];
        narrow_packed_sq += static_cast<double>(narrow_packed_output[i]) *
                            narrow_packed_output[i];
    }
    CHECK(narrow_dot / std::sqrt(narrow_raw_sq * narrow_packed_sq) > 0.99999,
          "one-row layout preserves projection fidelity");

    std::vector<uint8_t> task_packed;
    CHECK(!pack_q8_projection_corrected_bf16(
              raw.data(), raw.size(), k, n, task_packed, &error) &&
              error.find("multiple of 512") != std::string::npos,
          "task projection rejects a sub-descriptor K shape");
    constexpr int task_k = 512;
    std::vector<uint8_t> task_raw(q8_projection_bytes(task_k, n), 0);
    for (size_t block = 0;
         block < task_raw.size() / kQ8BlockBytes; ++block) {
        std::memcpy(task_raw.data() + block * kQ8BlockBytes,
                    raw.data() + (block % (raw.size() / kQ8BlockBytes)) *
                        kQ8BlockBytes,
                    kQ8BlockBytes);
    }
    std::vector<uint8_t> task_group_major;
    CHECK(pack_q8_gemm_corrected_bf16(
              task_raw.data(), task_raw.size(), task_k, n,
              task_group_major, &error) &&
          pack_q8_projection_corrected_bf16(
              task_raw.data(), task_raw.size(), task_k, n,
              task_packed, &error) &&
              task_packed.size() ==
                  q8_projection_task_packed_bytes(task_k, n),
          "task projection packs in column-major DMA order");
    bool task_layout = true;
    constexpr int k_tiles = task_k / kQ8TileK;
    constexpr int groups = n / kQ8OutputsPerPass;
    constexpr size_t task_stride =
        static_cast<size_t>(k_tiles * kQ8AieRows) *
            kQ8CorrectedTileBytes + 2 * sizeof(uint16_t);
    for (int column = 0; column < kQ8AieColumns && task_layout; ++column) {
        for (int group = 0; group < groups && task_layout; ++group) {
            for (int kt = 0; kt < k_tiles && task_layout; ++kt) {
                for (int row = 0; row < kQ8AieRows; ++row) {
                    const size_t source =
                        (((static_cast<size_t>(group) * kQ8AieColumns +
                           static_cast<size_t>(column)) * k_tiles +
                          static_cast<size_t>(kt)) * kQ8AieRows +
                         static_cast<size_t>(row)) * kQ8CorrectedTileBytes;
                    const size_t destination =
                        (static_cast<size_t>(column) * groups +
                         static_cast<size_t>(group)) * task_stride +
                        static_cast<size_t>(kt * kQ8AieRows + row) *
                            kQ8CorrectedTileBytes;
                    if (std::memcmp(task_group_major.data() + source,
                                    task_packed.data() + destination,
                                    kQ8CorrectedTileBytes) != 0) {
                        task_layout = false;
                        break;
                    }
                }
            }
        }
    }
    CHECK(task_layout, "column-major task layout retains every corrected tile");

    std::vector<uint8_t> task_narrow_raw(
        q8_projection_bytes(task_k, narrow_n));
    std::copy_n(task_raw.begin(), task_narrow_raw.size(),
                task_narrow_raw.begin());
    std::vector<uint8_t> task_narrow_packed;
    std::vector<uint8_t> task_padded;
    CHECK(pack_q8_projection_corrected_bf16(
              task_narrow_raw.data(), task_narrow_raw.size(), task_k,
              narrow_n, task_narrow_packed, &error) &&
          pad_q8_projection_rows(task_narrow_packed, task_k, narrow_n,
                                 1024, task_padded, &error) &&
          task_padded.size() ==
              q8_projection_task_packed_bytes(task_k, 1024),
          "one-row projection expands into a fixed two-row overlay");
    bool padded_layout = true;
    constexpr size_t narrow_task_stride =
        static_cast<size_t>(k_tiles) * kQ8CorrectedTileBytes +
        2 * sizeof(uint16_t);
    constexpr size_t padded_task_stride =
        static_cast<size_t>(k_tiles * 2) * kQ8CorrectedTileBytes +
        2 * sizeof(uint16_t);
    for (int column = 0;
         column < kQ8AieColumns && padded_layout; ++column) {
        for (int kt = 0; kt < k_tiles && padded_layout; ++kt) {
            const uint8_t * source_tile = task_narrow_packed.data() +
                static_cast<size_t>(column) * narrow_task_stride +
                static_cast<size_t>(kt) * kQ8CorrectedTileBytes;
            const uint8_t * real_tile = task_padded.data() +
                static_cast<size_t>(column) * padded_task_stride +
                static_cast<size_t>(kt * 2) * kQ8CorrectedTileBytes;
            const uint8_t * zero_tile = real_tile + kQ8CorrectedTileBytes;
            if (std::memcmp(source_tile, real_tile,
                            kQ8CorrectedTileBytes) != 0 ||
                std::any_of(zero_tile,
                            zero_tile + kQ8CorrectedTileBytes,
                            [](uint8_t value) { return value != 0; })) {
                padded_layout = false;
            }
        }
    }
    CHECK(padded_layout,
          "fixed-overlay padding preserves real rows and zeros added rows");
    CHECK(!pad_q8_projection_rows(task_padded, task_k, 1024, narrow_n,
                                  task_narrow_packed, &error),
          "fixed-overlay padding rejects a row shrink");
    std::vector<uint8_t> task_concatenated;
    CHECK(concat_q8_projection_rows(
              task_padded, task_k, 1024, task_narrow_packed, narrow_n,
              task_concatenated, &error) &&
          task_concatenated.size() ==
              q8_projection_task_packed_bytes(task_k, 1536),
          "same-input projections concatenate into adjacent overlay rows");
    bool concatenated_layout = true;
    constexpr size_t concatenated_task_stride =
        static_cast<size_t>(k_tiles * 3) * kQ8CorrectedTileBytes +
        2 * sizeof(uint16_t);
    for (int column = 0;
         column < kQ8AieColumns && concatenated_layout; ++column) {
        for (int kt = 0; kt < k_tiles && concatenated_layout; ++kt) {
            const uint8_t * first_rows = task_padded.data() +
                static_cast<size_t>(column) * padded_task_stride +
                static_cast<size_t>(kt * 2) * kQ8CorrectedTileBytes;
            const uint8_t * second_row = task_narrow_packed.data() +
                static_cast<size_t>(column) * narrow_task_stride +
                static_cast<size_t>(kt) * kQ8CorrectedTileBytes;
            const uint8_t * fused_rows = task_concatenated.data() +
                static_cast<size_t>(column) * concatenated_task_stride +
                static_cast<size_t>(kt * 3) * kQ8CorrectedTileBytes;
            if (std::memcmp(first_rows, fused_rows,
                            2 * kQ8CorrectedTileBytes) != 0 ||
                std::memcmp(second_row,
                            fused_rows + 2 * kQ8CorrectedTileBytes,
                            kQ8CorrectedTileBytes) != 0) {
                concatenated_layout = false;
            }
        }
    }
    CHECK(concatenated_layout,
          "concatenation preserves both projections' corrected tiles");
    CHECK(!concat_q8_projection_rows(
              task_packed, task_k, n, task_narrow_packed, narrow_n,
              task_concatenated, &error),
          "fixed-overlay concatenation rejects multiple output groups");

    std::vector<uint8_t> gemm_packed;
    CHECK(pack_q8_gemm_m32_corrected_bf16(
              task_raw.data(), task_raw.size(), task_k, n,
              gemm_packed, &error) &&
              gemm_packed.size() ==
                  q8_gemm_m32_packed_bytes(task_k, n),
          "m32 GEMM weights pack into blocked MMUL tasks");
    bool gemm_layout = true;
    constexpr int gemm_groups = n / kQ8OutputsPerRow;
    constexpr size_t gemm_tile_stride =
        kQ8CorrectedTileBytes + 2 * sizeof(uint16_t);
    for (int output_group = 0;
         output_group < gemm_groups && gemm_layout; ++output_group) {
        const int source_group = output_group / kQ8AieRows;
        const int source_row = output_group % kQ8AieRows;
        for (int column = 0;
             column < kQ8AieColumns && gemm_layout; ++column) {
            for (int kt = 0; kt < k_tiles && gemm_layout; ++kt) {
                const size_t source_tile =
                    (((static_cast<size_t>(source_group) * kQ8AieColumns +
                       static_cast<size_t>(column)) * k_tiles +
                      static_cast<size_t>(kt)) * kQ8AieRows +
                     static_cast<size_t>(source_row));
                const auto * source_high = reinterpret_cast<const uint16_t *>(
                    task_group_major.data() +
                        source_tile * kQ8CorrectedTileBytes);
                const auto * destination_high =
                    reinterpret_cast<const uint16_t *>(
                        gemm_packed.data() +
                        ((static_cast<size_t>(column) * gemm_groups +
                          static_cast<size_t>(output_group)) * k_tiles +
                         static_cast<size_t>(kt)) * gemm_tile_stride);
                for (int input_lane = 0;
                     input_lane < kQ8TileK && gemm_layout; ++input_lane) {
                    for (int output_lane = 0; output_lane < kQ8TileN;
                         ++output_lane) {
                        const int kb = input_lane / 8;
                        const int ir = input_lane % 8;
                        const int nb = output_lane / 8;
                        const int jn = output_lane % 8;
                        const size_t blocked =
                            (static_cast<size_t>(kb * 8 + nb) * 8 +
                             static_cast<size_t>(ir)) * 8 +
                            static_cast<size_t>(jn);
                        const size_t linear =
                            static_cast<size_t>(input_lane) * kQ8TileN +
                            static_cast<size_t>(output_lane);
                        if (destination_high[blocked] != source_high[linear]) {
                            gemm_layout = false;
                            break;
                        }
                    }
                }
                const uint16_t * sentinel = reinterpret_cast<const uint16_t *>(
                    reinterpret_cast<const uint8_t *>(destination_high) +
                    kQ8CorrectedTileBytes);
                if (sentinel[0] != 0 || sentinel[1] != 0)
                    gemm_layout = false;
            }
        }
    }
    CHECK(gemm_layout, "m32 MMUL blocking preserves every high-plane weight");

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
