#include "q8_0_pack.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace ember::xdna2 {
namespace {

bool checked_raw_size(int k, int n, size_t * size) {
    if (k <= 0 || n <= 0 || k % kQ8BlockWeights != 0) return false;
    const size_t blocks = static_cast<size_t>(k) / kQ8BlockWeights;
    if (static_cast<size_t>(n) > std::numeric_limits<size_t>::max() / blocks ||
        static_cast<size_t>(n) * blocks >
            std::numeric_limits<size_t>::max() / kQ8BlockBytes) return false;
    *size = static_cast<size_t>(n) * blocks * kQ8BlockBytes;
    return true;
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

uint16_t float_to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
}

float bf16_to_float(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

size_t tile_count(int k, int n) {
    return static_cast<size_t>(k / kQ8TileK) *
           static_cast<size_t>(n / kQ8TileN);
}

}  // namespace

bool q8_supported_shape(int k, int n) {
    return k > 0 && n > 0 && k % kQ8TileK == 0 &&
           n % kQ8OutputsPerPass == 0;
}

size_t q8_projection_bytes(int k, int n) {
    size_t bytes = 0;
    return checked_raw_size(k, n, &bytes) ? bytes : 0;
}

size_t q8_packed_projection_bytes(int k, int n) {
    if (!q8_supported_shape(k, n)) return 0;
    const size_t tiles = tile_count(k, n);
    if (tiles > std::numeric_limits<size_t>::max() / kQ8PackedTileBytes)
        return 0;
    return tiles * kQ8PackedTileBytes;
}

size_t q8_corrected_packed_projection_bytes(int k, int n) {
    if (!q8_supported_shape(k, n)) return 0;
    const size_t tiles = tile_count(k, n);
    if (tiles > std::numeric_limits<size_t>::max() / kQ8CorrectedTileBytes)
        return 0;
    return tiles * kQ8CorrectedTileBytes;
}

bool pack_q8_gemm_bf16(const void * raw, size_t raw_bytes, int k, int n,
                       std::vector<uint8_t> & packed, std::string * error) {
    size_t expected = 0;
    const size_t packed_bytes = q8_packed_projection_bytes(k, n);
    if (!raw || !checked_raw_size(k, n, &expected) || !packed_bytes) {
        if (error) *error = "Q8 AIE packing requires K%128=0 and N%2048=0";
        return false;
    }
    if (raw_bytes < expected) {
        if (error) *error = "Q8 projection is shorter than its shape";
        return false;
    }
    const auto * source = static_cast<const uint8_t *>(raw);
    const int blocks_per_output = k / kQ8BlockWeights;
    const int k_tiles = k / kQ8TileK;
    const int groups = n / kQ8OutputsPerPass;
    packed.resize(packed_bytes);
    size_t tile_index = 0;
    for (int group = 0; group < groups; ++group) {
        for (int column = 0; column < kQ8AieColumns; ++column) {
            for (int kt = 0; kt < k_tiles; ++kt) {
                for (int row = 0; row < kQ8AieRows; ++row, ++tile_index) {
                    auto * tile = reinterpret_cast<uint16_t *>(
                        packed.data() + tile_index * kQ8PackedTileBytes);
                    for (int input = 0; input < kQ8TileK; ++input) {
                        const int global_input = kt * kQ8TileK + input;
                        const int block = global_input / kQ8BlockWeights;
                        const int lane = global_input % kQ8BlockWeights;
                        for (int output_lane = 0; output_lane < kQ8TileN;
                             ++output_lane) {
                            const int output = group * kQ8OutputsPerPass +
                                row * kQ8AieColumns * kQ8TileN +
                                column * kQ8TileN + output_lane;
                            const uint8_t * q = source +
                                (static_cast<size_t>(output) *
                                     static_cast<size_t>(blocks_per_output) +
                                 static_cast<size_t>(block)) *
                                kQ8BlockBytes;
                            uint16_t scale_bits = 0;
                            std::memcpy(&scale_bits, q, sizeof(scale_bits));
                            const int8_t code = static_cast<int8_t>(q[2 + lane]);
                            tile[static_cast<size_t>(input) * kQ8TileN +
                                 static_cast<size_t>(output_lane)] = float_to_bf16(
                                static_cast<float>(code) *
                                fp16_to_float(scale_bits));
                        }
                    }
                }
            }
        }
    }
    return tile_index == tile_count(k, n);
}

bool pack_q8_gemm_corrected_bf16(const void * raw, size_t raw_bytes,
                                 int k, int n,
                                 std::vector<uint8_t> & packed,
                                 std::string * error) {
    size_t expected = 0;
    const size_t packed_bytes = q8_corrected_packed_projection_bytes(k, n);
    if (!raw || !checked_raw_size(k, n, &expected) || !packed_bytes) {
        if (error) *error =
            "corrected Q8 AIE packing requires K%128=0 and N%2048=0";
        return false;
    }
    if (raw_bytes < expected) {
        if (error) *error = "Q8 projection is shorter than its shape";
        return false;
    }
    const auto * source = static_cast<const uint8_t *>(raw);
    const int blocks_per_output = k / kQ8BlockWeights;
    const int k_tiles = k / kQ8TileK;
    const int groups = n / kQ8OutputsPerPass;
    packed.resize(packed_bytes);
    size_t tile_index = 0;
    for (int group = 0; group < groups; ++group) {
        for (int column = 0; column < kQ8AieColumns; ++column) {
            for (int kt = 0; kt < k_tiles; ++kt) {
                for (int row = 0; row < kQ8AieRows; ++row, ++tile_index) {
                    auto * high = reinterpret_cast<uint16_t *>(
                        packed.data() +
                        tile_index * kQ8CorrectedTileBytes);
                    uint16_t * low = high +
                        static_cast<size_t>(kQ8TileK) * kQ8TileN;
                    for (int input = 0; input < kQ8TileK; ++input) {
                        const int global_input = kt * kQ8TileK + input;
                        const int block = global_input / kQ8BlockWeights;
                        const int lane = global_input % kQ8BlockWeights;
                        for (int output_lane = 0; output_lane < kQ8TileN;
                             ++output_lane) {
                            const int output = group * kQ8OutputsPerPass +
                                row * kQ8AieColumns * kQ8TileN +
                                column * kQ8TileN + output_lane;
                            const uint8_t * q = source +
                                (static_cast<size_t>(output) *
                                     static_cast<size_t>(blocks_per_output) +
                                 static_cast<size_t>(block)) *
                                kQ8BlockBytes;
                            uint16_t scale_bits = 0;
                            std::memcpy(&scale_bits, q, sizeof(scale_bits));
                            const int8_t code =
                                static_cast<int8_t>(q[2 + lane]);
                            const float exact = static_cast<float>(code) *
                                fp16_to_float(scale_bits);
                            const uint16_t high_bits = float_to_bf16(exact);
                            const size_t index =
                                static_cast<size_t>(input) * kQ8TileN +
                                static_cast<size_t>(output_lane);
                            high[index] = high_bits;
                            low[index] = float_to_bf16(
                                exact - bf16_to_float(high_bits));
                        }
                    }
                }
            }
        }
    }
    return tile_index == tile_count(k, n);
}

size_t q8_expert_v1_bytes() {
    const size_t first = q8_packed_projection_bytes(4096, 2048);
    const size_t down = q8_packed_projection_bytes(2048, 4096);
    return first && down ? 2 * first + down : 0;
}

bool pack_q8_expert_v1(const void * gate, size_t gate_bytes,
                       const void * up, size_t up_bytes,
                       const void * down, size_t down_bytes,
                       std::vector<uint8_t> & packed, std::string * error) {
    std::vector<uint8_t> gate_tiles, up_tiles, down_tiles;
    if (!pack_q8_gemm_bf16(gate, gate_bytes, 4096, 2048,
                           gate_tiles, error) ||
        !pack_q8_gemm_bf16(up, up_bytes, 4096, 2048, up_tiles, error) ||
        !pack_q8_gemm_bf16(down, down_bytes, 2048, 4096,
                           down_tiles, error)) return false;
    constexpr int gate_k_tiles = 4096 / kQ8TileK;
    constexpr int down_k_tiles = 2048 / kQ8TileK;
    constexpr int down_groups = 4096 / kQ8OutputsPerPass;
    packed.resize(q8_expert_v1_bytes());
    size_t destination = 0;
    auto append = [&](const std::vector<uint8_t> & source, size_t tile) {
        std::memcpy(packed.data() + destination,
                    source.data() + tile * kQ8PackedTileBytes,
                    kQ8PackedTileBytes);
        destination += kQ8PackedTileBytes;
    };
    for (int column = 0; column < kQ8AieColumns; ++column) {
        for (int kt = 0; kt < gate_k_tiles; ++kt) {
            for (int row = 0; row < kQ8AieRows; ++row) {
                const size_t tile =
                    (static_cast<size_t>(column) * gate_k_tiles +
                     static_cast<size_t>(kt)) * kQ8AieRows +
                    static_cast<size_t>(row);
                append(gate_tiles, tile);
            }
            for (int row = 0; row < kQ8AieRows; ++row) {
                const size_t tile =
                    (static_cast<size_t>(column) * gate_k_tiles +
                     static_cast<size_t>(kt)) * kQ8AieRows +
                    static_cast<size_t>(row);
                append(up_tiles, tile);
            }
        }
        for (int group = 0; group < down_groups; ++group) {
            for (int kt = 0; kt < down_k_tiles; ++kt) {
                for (int row = 0; row < kQ8AieRows; ++row) {
                    const size_t tile =
                        ((static_cast<size_t>(group) * kQ8AieColumns +
                          static_cast<size_t>(column)) * down_k_tiles +
                         static_cast<size_t>(kt)) * kQ8AieRows +
                        static_cast<size_t>(row);
                    append(down_tiles, tile);
                }
            }
        }
    }
    return destination == packed.size();
}

size_t q8_expert_v2_bytes() {
    const size_t first = q8_corrected_packed_projection_bytes(4096, 2048);
    const size_t down = q8_corrected_packed_projection_bytes(2048, 4096);
    return first && down ? 2 * first + down : 0;
}

bool pack_q8_expert_v2(const void * gate, size_t gate_bytes,
                       const void * up, size_t up_bytes,
                       const void * down, size_t down_bytes,
                       std::vector<uint8_t> & packed, std::string * error) {
    std::vector<uint8_t> gate_tiles, up_tiles, down_tiles;
    if (!pack_q8_gemm_corrected_bf16(gate, gate_bytes, 4096, 2048,
                                     gate_tiles, error) ||
        !pack_q8_gemm_corrected_bf16(up, up_bytes, 4096, 2048,
                                     up_tiles, error) ||
        !pack_q8_gemm_corrected_bf16(down, down_bytes, 2048, 4096,
                                     down_tiles, error)) return false;
    constexpr int gate_k_tiles = 4096 / kQ8TileK;
    constexpr int down_k_tiles = 2048 / kQ8TileK;
    constexpr int down_groups = 4096 / kQ8OutputsPerPass;
    packed.resize(q8_expert_v2_bytes());
    size_t destination = 0;
    auto append = [&](const std::vector<uint8_t> & source, size_t tile) {
        std::memcpy(packed.data() + destination,
                    source.data() + tile * kQ8CorrectedTileBytes,
                    kQ8CorrectedTileBytes);
        destination += kQ8CorrectedTileBytes;
    };
    for (int column = 0; column < kQ8AieColumns; ++column) {
        for (int kt = 0; kt < gate_k_tiles; ++kt) {
            for (int row = 0; row < kQ8AieRows; ++row) {
                const size_t tile =
                    (static_cast<size_t>(column) * gate_k_tiles +
                     static_cast<size_t>(kt)) * kQ8AieRows +
                    static_cast<size_t>(row);
                append(gate_tiles, tile);
            }
            for (int row = 0; row < kQ8AieRows; ++row) {
                const size_t tile =
                    (static_cast<size_t>(column) * gate_k_tiles +
                     static_cast<size_t>(kt)) * kQ8AieRows +
                    static_cast<size_t>(row);
                append(up_tiles, tile);
            }
        }
        for (int group = 0; group < down_groups; ++group) {
            for (int kt = 0; kt < down_k_tiles; ++kt) {
                for (int row = 0; row < kQ8AieRows; ++row) {
                    const size_t tile =
                        ((static_cast<size_t>(group) * kQ8AieColumns +
                          static_cast<size_t>(column)) * down_k_tiles +
                         static_cast<size_t>(kt)) * kQ8AieRows +
                        static_cast<size_t>(row);
                    append(down_tiles, tile);
                }
            }
        }
    }
    return destination == packed.size();
}

bool q8_gemm_raw_reference(const void * raw, size_t raw_bytes,
                           const float * input, int k, int n,
                           float * output) {
    const size_t expected = q8_projection_bytes(k, n);
    if (!raw || !input || !output || !expected || raw_bytes < expected)
        return false;
    const auto * bytes = static_cast<const uint8_t *>(raw);
    const int blocks_per_output = k / kQ8BlockWeights;
    for (int out = 0; out < n; ++out) {
        float sum = 0.0f;
        for (int block = 0; block < blocks_per_output; ++block) {
            const uint8_t * q = bytes +
                (static_cast<size_t>(out) *
                     static_cast<size_t>(blocks_per_output) +
                 static_cast<size_t>(block)) *
                kQ8BlockBytes;
            uint16_t scale_bits = 0;
            std::memcpy(&scale_bits, q, sizeof(scale_bits));
            const float scale = fp16_to_float(scale_bits);
            for (int lane = 0; lane < kQ8BlockWeights; ++lane) {
                sum += input[block * kQ8BlockWeights + lane] *
                    static_cast<float>(static_cast<int8_t>(q[2 + lane])) * scale;
            }
        }
        output[out] = sum;
    }
    return true;
}

bool q8_gemm_packed_bf16_reference(const void * packed, size_t packed_bytes,
                                   const float * input, int k, int n,
                                   float * output) {
    const size_t expected = q8_packed_projection_bytes(k, n);
    if (!packed || !input || !output || !expected || packed_bytes < expected)
        return false;
    const auto * bytes = static_cast<const uint8_t *>(packed);
    const int k_tiles = k / kQ8TileK;
    for (int out = 0; out < n; ++out) {
        const int group = out / kQ8OutputsPerPass;
        const int within = out % kQ8OutputsPerPass;
        const int row = within / (kQ8AieColumns * kQ8TileN);
        const int column = (within / kQ8TileN) % kQ8AieColumns;
        const int lane = within % kQ8TileN;
        float sum = 0.0f;
        for (int kt = 0; kt < k_tiles; ++kt) {
            size_t tile = static_cast<size_t>(group);
            tile = tile * kQ8AieColumns + static_cast<size_t>(column);
            tile = tile * static_cast<size_t>(k_tiles) +
                   static_cast<size_t>(kt);
            tile = tile * kQ8AieRows + static_cast<size_t>(row);
            const auto * values = reinterpret_cast<const uint16_t *>(
                bytes + tile * kQ8PackedTileBytes);
            for (int i = 0; i < kQ8TileK; ++i) {
                sum += input[kt * kQ8TileK + i] * bf16_to_float(
                    values[static_cast<size_t>(i) * kQ8TileN +
                           static_cast<size_t>(lane)]);
            }
        }
        output[out] = sum;
    }
    return true;
}

}  // namespace ember::xdna2
