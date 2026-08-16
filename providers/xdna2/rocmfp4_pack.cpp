#include "rocmfp4_pack.h"

#include "rocmfp2_pack.h"

#include <cstring>
#include <limits>

namespace ember::xdna2 {
namespace {

bool checked_projection_size(int k, int n, size_t * size) {
    if (k <= 0 || n <= 0 || k % kRocmfp4BlockWeights != 0) return false;
    const size_t blocks = static_cast<size_t>(k) / kRocmfp4BlockWeights;
    if (static_cast<size_t>(n) > std::numeric_limits<size_t>::max() / blocks ||
        static_cast<size_t>(n) * blocks >
            std::numeric_limits<size_t>::max() / kRocmfp4BlockBytes) {
        return false;
    }
    *size = static_cast<size_t>(n) * blocks * kRocmfp4BlockBytes;
    return true;
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

int8_t decode_code(uint8_t code) {
    code &= 0x0fu;
    const int magnitude_code = code & 7;
    const int magnitude = magnitude_code <= 4
        ? magnitude_code : 2 * magnitude_code - 4;
    return static_cast<int8_t>((code & 8u) ? -magnitude : magnitude);
}

uint8_t block_code(const uint8_t * block, int index) {
    return index < 16
        ? static_cast<uint8_t>(block[index] & 0x0fu)
        : static_cast<uint8_t>((block[index - 16] >> 4) & 0x0fu);
}

size_t tile_count(int k, int n) {
    return (static_cast<size_t>(k) / kRocmfp4TileK) *
           (static_cast<size_t>(n) / kRocmfp4TileN);
}

}  // namespace

bool rocmfp4_supported_shape(int k, int n) {
    return k > 0 && n > 0 && k % kRocmfp4TileK == 0 &&
           n % kRocmfp4OutputsPerPass == 0;
}

size_t rocmfp4_projection_bytes(int k, int n) {
    size_t result = 0;
    return checked_projection_size(k, n, &result) ? result : 0;
}

size_t rocmfp4_packed_projection_bytes(int k, int n) {
    if (!rocmfp4_supported_shape(k, n)) return 0;
    const size_t tiles = tile_count(k, n);
    if (tiles > std::numeric_limits<size_t>::max() /
                    kRocmfp4PackedTileBytes) {
        return 0;
    }
    return tiles * kRocmfp4PackedTileBytes;
}

bool pack_rocmfp4_gemm(const void * raw, size_t raw_bytes, int k, int n,
                       std::vector<uint8_t> & packed, std::string * error) {
    size_t expected = 0;
    const size_t packed_bytes = rocmfp4_packed_projection_bytes(k, n);
    if (!raw || !checked_projection_size(k, n, &expected) ||
        packed_bytes == 0) {
        if (error) *error = "ROCMFP4 AIE packing requires K%128=0 and N%2048=0";
        return false;
    }
    if (raw_bytes < expected) {
        if (error) *error = "ROCMFP4 projection is shorter than its shape";
        return false;
    }

    const auto * source = static_cast<const uint8_t *>(raw);
    const int k_tiles = k / kRocmfp4TileK;
    const int blocks_per_row = k / kRocmfp4BlockWeights;
    const int output_groups = n / kRocmfp4OutputsPerPass;
    packed.assign(packed_bytes, 0);
    size_t tile_index = 0;
    for (int group = 0; group < output_groups; ++group) {
        for (int column = 0; column < kRocmfp4AieColumns; ++column) {
            for (int kt = 0; kt < k_tiles; ++kt) {
                for (int row = 0; row < kRocmfp4AieRows;
                     ++row, ++tile_index) {
                    uint8_t * tile = packed.data() +
                        tile_index * kRocmfp4PackedTileBytes;
                    uint8_t * scales = tile + kRocmfp4CodeTileBytes;
                    for (int lane = 0; lane < kRocmfp4TileN; ++lane) {
                        const int output = group * kRocmfp4OutputsPerPass +
                            row * kRocmfp4AieColumns * kRocmfp4TileN +
                            column * kRocmfp4TileN + lane;
                        for (int block = 0;
                             block < kRocmfp4TileK / kRocmfp4BlockWeights;
                             ++block) {
                            const size_t source_block =
                                static_cast<size_t>(output) *
                                    static_cast<size_t>(blocks_per_row) +
                                static_cast<size_t>(kt) *
                                    (kRocmfp4TileK / kRocmfp4BlockWeights) +
                                static_cast<size_t>(block);
                            const uint8_t * q = source +
                                source_block * kRocmfp4BlockBytes;
                            const size_t metadata =
                                static_cast<size_t>(block) * kRocmfp4TileN +
                                static_cast<size_t>(lane);
                            const uint16_t scale =
                                float_to_bf16(ue4m3_to_float(q[16]));
                            std::memcpy(scales + metadata * sizeof(scale),
                                        &scale, sizeof(scale));
                            for (int i = 0; i < kRocmfp4BlockWeights; ++i) {
                                const int input =
                                    block * kRocmfp4BlockWeights + i;
                                uint8_t & pair = tile[
                                    static_cast<size_t>(input) *
                                        (kRocmfp4TileN / 2) +
                                    static_cast<size_t>(lane / 2)];
                                pair |= static_cast<uint8_t>(
                                    block_code(q, i) << (4 * (lane & 1)));
                            }
                        }
                    }
                }
            }
        }
    }
    return tile_index == tile_count(k, n);
}

size_t rocmfp4_expert_v7_bytes() {
    const size_t first = rocmfp4_packed_projection_bytes(4096, 2048);
    const size_t down = rocmfp4_packed_projection_bytes(2048, 4096);
    return first && down ? 2 * first + down : 0;
}

bool pack_rocmfp4_expert_v7(const void * gate, size_t gate_bytes,
                            const void * up, size_t up_bytes,
                            const void * down, size_t down_bytes,
                            std::vector<uint8_t> & packed,
                            std::string * error) {
    std::vector<uint8_t> gate_tiles, up_tiles, down_tiles;
    if (!pack_rocmfp4_gemm(gate, gate_bytes, 4096, 2048,
                           gate_tiles, error) ||
        !pack_rocmfp4_gemm(up, up_bytes, 4096, 2048,
                           up_tiles, error) ||
        !pack_rocmfp4_gemm(down, down_bytes, 2048, 4096,
                           down_tiles, error)) {
        return false;
    }
    constexpr int gate_k_tiles = 4096 / kRocmfp4TileK;
    constexpr int down_k_tiles = 2048 / kRocmfp4TileK;
    constexpr int down_groups = 4096 / kRocmfp4OutputsPerPass;
    packed.resize(rocmfp4_expert_v7_bytes());
    size_t destination = 0;
    auto append_tile = [&](const std::vector<uint8_t> & source,
                           size_t source_tile) {
        std::memcpy(packed.data() + destination,
                    source.data() +
                        source_tile * kRocmfp4PackedTileBytes,
                    kRocmfp4PackedTileBytes);
        destination += kRocmfp4PackedTileBytes;
    };
    for (int column = 0; column < kRocmfp4AieColumns; ++column) {
        for (int kt = 0; kt < gate_k_tiles; ++kt) {
            for (int row = 0; row < kRocmfp4AieRows; ++row) {
                const size_t source_tile =
                    (static_cast<size_t>(column) * gate_k_tiles +
                     static_cast<size_t>(kt)) * kRocmfp4AieRows +
                    static_cast<size_t>(row);
                append_tile(gate_tiles, source_tile);
            }
            for (int row = 0; row < kRocmfp4AieRows; ++row) {
                const size_t source_tile =
                    (static_cast<size_t>(column) * gate_k_tiles +
                     static_cast<size_t>(kt)) * kRocmfp4AieRows +
                    static_cast<size_t>(row);
                append_tile(up_tiles, source_tile);
            }
        }
        for (int group = 0; group < down_groups; ++group) {
            for (int kt = 0; kt < down_k_tiles; ++kt) {
                for (int row = 0; row < kRocmfp4AieRows; ++row) {
                    const size_t source_tile =
                        ((static_cast<size_t>(group) *
                              kRocmfp4AieColumns +
                          static_cast<size_t>(column)) * down_k_tiles +
                         static_cast<size_t>(kt)) * kRocmfp4AieRows +
                        static_cast<size_t>(row);
                    append_tile(down_tiles, source_tile);
                }
            }
        }
    }
    return destination == packed.size();
}

bool rocmfp4_gemm_raw_reference(const void * raw, size_t raw_bytes,
                                const float * input, int k, int n,
                                float scale, float * output) {
    const size_t expected = rocmfp4_projection_bytes(k, n);
    if (!raw || !input || !output || !expected || raw_bytes < expected)
        return false;
    const auto * bytes = static_cast<const uint8_t *>(raw);
    const int blocks_per_row = k / kRocmfp4BlockWeights;
    for (int out = 0; out < n; ++out) {
        float sum = 0.0f;
        for (int block = 0; block < blocks_per_row; ++block) {
            const uint8_t * q = bytes +
                (static_cast<size_t>(out) *
                     static_cast<size_t>(blocks_per_row) +
                 static_cast<size_t>(block)) * kRocmfp4BlockBytes;
            const float block_scale = ue4m3_to_float(q[16]);
            for (int i = 0; i < kRocmfp4BlockWeights; ++i) {
                sum += input[block * kRocmfp4BlockWeights + i] *
                       static_cast<float>(decode_code(block_code(q, i))) *
                       block_scale;
            }
        }
        output[out] = sum * scale;
    }
    return true;
}

bool rocmfp4_gemm_packed_reference(const void * packed, size_t packed_bytes,
                                   const float * input, int k, int n,
                                   float scale, float * output) {
    const size_t expected = rocmfp4_packed_projection_bytes(k, n);
    if (!packed || !input || !output || !expected || packed_bytes < expected)
        return false;
    const auto * bytes = static_cast<const uint8_t *>(packed);
    const int k_tiles = k / kRocmfp4TileK;
    for (int out = 0; out < n; ++out) {
        const int group = out / kRocmfp4OutputsPerPass;
        const int within = out % kRocmfp4OutputsPerPass;
        const int row = within /
            (kRocmfp4AieColumns * kRocmfp4TileN);
        const int column =
            (within / kRocmfp4TileN) % kRocmfp4AieColumns;
        const int lane = within % kRocmfp4TileN;
        float sum = 0.0f;
        for (int kt = 0; kt < k_tiles; ++kt) {
            size_t tile_index = static_cast<size_t>(group);
            tile_index = tile_index * kRocmfp4AieColumns +
                         static_cast<size_t>(column);
            tile_index = tile_index * static_cast<size_t>(k_tiles) +
                         static_cast<size_t>(kt);
            tile_index = tile_index * kRocmfp4AieRows +
                         static_cast<size_t>(row);
            const uint8_t * tile = bytes +
                tile_index * kRocmfp4PackedTileBytes;
            const uint8_t * scales = tile + kRocmfp4CodeTileBytes;
            for (int block = 0;
                 block < kRocmfp4TileK / kRocmfp4BlockWeights; ++block) {
                const size_t metadata =
                    static_cast<size_t>(block) * kRocmfp4TileN +
                    static_cast<size_t>(lane);
                uint16_t scale_bits = 0;
                std::memcpy(&scale_bits,
                            scales + metadata * sizeof(scale_bits),
                            sizeof(scale_bits));
                const float block_scale = bf16_to_float(scale_bits);
                for (int i = 0; i < kRocmfp4BlockWeights; ++i) {
                    const int local_input =
                        block * kRocmfp4BlockWeights + i;
                    const uint8_t pair = tile[
                        static_cast<size_t>(local_input) *
                            (kRocmfp4TileN / 2) +
                        static_cast<size_t>(lane / 2)];
                    const uint8_t code = static_cast<uint8_t>(
                        (pair >> (4 * (lane & 1))) & 0x0fu);
                    sum += input[kt * kRocmfp4TileK + local_input] *
                           static_cast<float>(decode_code(code)) *
                           block_scale;
                }
            }
        }
        output[out] = sum * scale;
    }
    return true;
}

}  // namespace ember::xdna2
