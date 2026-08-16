#include "rocmfp2_pack.h"

#include <cstring>
#include <limits>

namespace ember::xdna2 {
namespace {

bool checked_projection_size(int k, int n, size_t * size) {
    if (k <= 0 || n <= 0 || k % kRocmfp2BlockWeights != 0) return false;
    const size_t blocks_per_row = (size_t)k / kRocmfp2BlockWeights;
    if ((size_t)n > std::numeric_limits<size_t>::max() / blocks_per_row ||
        (size_t)n * blocks_per_row >
            std::numeric_limits<size_t>::max() / kRocmfp2BlockBytes) {
        return false;
    }
    *size = (size_t)n * blocks_per_row * kRocmfp2BlockBytes;
    return true;
}

float decode(const uint8_t * block, int index) {
    const uint8_t code =
        (uint8_t)((block[index >> 2] >> (2 * (index & 3))) & 3u);
    return (float)code * ue4m3_to_float(block[8]) -
           ue4m3_to_float(block[9]);
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

size_t v4_tile_count(int k, int n) {
    return (static_cast<size_t>(k) / kGemvTileK) *
           (static_cast<size_t>(n) / kGemvTileN);
}

}  // namespace

bool rocmfp2_supported_shape(int k, int n) {
    return k > 0 && n > 0 && k % kGemvTileK == 0 &&
           n % kOutputsPerArrayPass == 0;
}

size_t rocmfp2_projection_bytes(int k, int n) {
    size_t size = 0;
    return checked_projection_size(k, n, &size) ? size : 0;
}

float ue4m3_to_float(uint8_t value) {
    if (value > 0x7e) return 0.0f;
    const unsigned exponent = value >> 3;
    const unsigned mantissa = value & 7u;
    if (exponent == 0) return (float)mantissa * 0x1p-10f;
    static constexpr float powers[16] = {
        0.0f, 0x1p-10f, 0x1p-9f, 0x1p-8f,
        0x1p-7f, 0x1p-6f, 0x1p-5f, 0x1p-4f,
        0x1p-3f, 0x1p-2f, 0x1p-1f, 0x1p0f,
        0x1p1f, 0x1p2f, 0x1p3f, 0x1p4f,
    };
    return (float)(8u + mantissa) * powers[exponent];
}

bool pack_rocmfp2_gemv(const void * raw, size_t raw_bytes, int k, int n,
                       std::vector<uint8_t> & packed, std::string * error) {
    size_t expected = 0;
    if (!raw || !checked_projection_size(k, n, &expected) ||
        !rocmfp2_supported_shape(k, n)) {
        if (error) *error = "ROCMFP2 GEMV requires K%128=0 and N%2048=0";
        return false;
    }
    if (raw_bytes < expected) {
        if (error) *error = "ROCMFP2 projection is shorter than its shape";
        return false;
    }

    const auto * source = static_cast<const uint8_t *>(raw);
    const int k_tiles = k / kGemvTileK;
    const int blocks_per_row = k / kRocmfp2BlockWeights;
    const int output_groups = n / kOutputsPerArrayPass;
    packed.resize(expected);
    size_t destination = 0;

    // Array-pass -> column -> K tile -> row -> output lane -> four blocks.
    // Four row tiles form one memory-tile object, which the AIE object-fifo
    // link splits across the four compute rows without a host-side gather.
    for (int group = 0; group < output_groups; ++group) {
        for (int column = 0; column < kAieColumns; ++column) {
            for (int kt = 0; kt < k_tiles; ++kt) {
                for (int row = 0; row < kAieRows; ++row) {
                    for (int lane = 0; lane < kGemvTileN; ++lane) {
                        const int output = group * kOutputsPerArrayPass +
                                           row * kAieColumns * kGemvTileN +
                                           column * kGemvTileN + lane;
                        const size_t source_offset =
                            ((size_t)output * (size_t)blocks_per_row +
                             (size_t)kt * (kGemvTileK / kRocmfp2BlockWeights)) *
                            kRocmfp2BlockBytes;
                        constexpr size_t row_tile_bytes =
                            (kGemvTileK / kRocmfp2BlockWeights) *
                            kRocmfp2BlockBytes;
                        std::memcpy(packed.data() + destination,
                                    source + source_offset, row_tile_bytes);
                        destination += row_tile_bytes;
                    }
                }
            }
        }
    }
    return destination == expected;
}

size_t rocmfp2_v4_projection_bytes(int k, int n) {
    if (!rocmfp2_supported_shape(k, n)) return 0;
    const size_t tiles = v4_tile_count(k, n);
    if (tiles > std::numeric_limits<size_t>::max() / kV4PackedTileBytes)
        return 0;
    return tiles * kV4PackedTileBytes;
}

bool pack_rocmfp2_gemv_v4(const void * raw, size_t raw_bytes, int k, int n,
                          std::vector<uint8_t> & packed,
                          std::string * error) {
    size_t expected = 0;
    const size_t packed_bytes = rocmfp2_v4_projection_bytes(k, n);
    if (!raw || !checked_projection_size(k, n, &expected) ||
        packed_bytes == 0) {
        if (error) *error = "ROCMFP2 Gen4 GEMV requires K%128=0 and N%2048=0";
        return false;
    }
    if (raw_bytes < expected) {
        if (error) *error = "ROCMFP2 projection is shorter than its shape";
        return false;
    }

    const auto * source = static_cast<const uint8_t *>(raw);
    const int k_tiles = k / kGemvTileK;
    const int blocks_per_row = k / kRocmfp2BlockWeights;
    const int output_groups = n / kOutputsPerArrayPass;
    packed.assign(packed_bytes, 0);
    size_t tile_index = 0;

    // Preserve the array-pass/column/K-tile/row distribution used by Gen1-3,
    // but transpose each 128x64 tile to K-major vectors. The code plane stores
    // two unsigned FP2 codes per byte as uint4 nibbles, so AIE2P can use one
    // native cast+unpack for all 64 output lanes. Scale and offset are exact
    // BF16 values: UE4M3 has only three mantissa bits.
    for (int group = 0; group < output_groups; ++group) {
        for (int column = 0; column < kAieColumns; ++column) {
            for (int kt = 0; kt < k_tiles; ++kt) {
                for (int row = 0; row < kAieRows; ++row, ++tile_index) {
                    uint8_t * tile = packed.data() + tile_index * kV4PackedTileBytes;
                    uint8_t * scales = tile + kV4CodeTileBytes;
                    uint8_t * offsets =
                        tile + kV4CodeTileBytes + kV4MetadataPlaneBytes;
                    for (int lane = 0; lane < kGemvTileN; ++lane) {
                        const int output = group * kOutputsPerArrayPass +
                                           row * kAieColumns * kGemvTileN +
                                           column * kGemvTileN + lane;
                        for (int block = 0;
                             block < kGemvTileK / kRocmfp2BlockWeights;
                             ++block) {
                            const size_t source_block =
                                static_cast<size_t>(output) *
                                    static_cast<size_t>(blocks_per_row) +
                                static_cast<size_t>(kt) *
                                    (kGemvTileK / kRocmfp2BlockWeights) +
                                static_cast<size_t>(block);
                            const uint8_t * q = source +
                                source_block * kRocmfp2BlockBytes;
                            const size_t metadata =
                                static_cast<size_t>(block) * kGemvTileN +
                                static_cast<size_t>(lane);
                            const uint16_t scale =
                                float_to_bf16(ue4m3_to_float(q[8]));
                            const uint16_t offset =
                                float_to_bf16(ue4m3_to_float(q[9]));
                            std::memcpy(scales + metadata * sizeof(scale),
                                        &scale, sizeof(scale));
                            std::memcpy(offsets + metadata * sizeof(offset),
                                        &offset, sizeof(offset));
                            for (int i = 0; i < kRocmfp2BlockWeights; ++i) {
                                const uint8_t code = static_cast<uint8_t>(
                                    (q[i >> 2] >> (2 * (i & 3))) & 3u);
                                const int input = block * kRocmfp2BlockWeights + i;
                                uint8_t & pair = tile[
                                    static_cast<size_t>(input) *
                                        (kGemvTileN / kRocmfp2V4CodesPerByte) +
                                    static_cast<size_t>(lane / 2)];
                                pair |= static_cast<uint8_t>(
                                    code << (4 * (lane & 1)));
                            }
                        }
                    }
                }
            }
        }
    }
    return tile_index == v4_tile_count(k, n);
}

size_t rocmfp2_expert_v5_bytes() {
    return 3 * rocmfp2_v4_projection_bytes(4096, 2048);
}

bool pack_rocmfp2_expert_v5(const void * gate, size_t gate_bytes,
                            const void * up, size_t up_bytes,
                            const void * down, size_t down_bytes,
                            std::vector<uint8_t> & packed,
                            std::string * error) {
    std::vector<uint8_t> gate_v4, up_v4, down_v4;
    if (!pack_rocmfp2_gemv_v4(gate, gate_bytes, 4096, 2048,
                              gate_v4, error) ||
        !pack_rocmfp2_gemv_v4(up, up_bytes, 4096, 2048,
                              up_v4, error) ||
        !pack_rocmfp2_gemv_v4(down, down_bytes, 2048, 4096,
                              down_v4, error)) {
        return false;
    }
    constexpr int gate_k_tiles = 4096 / kGemvTileK;
    constexpr int down_k_tiles = 2048 / kGemvTileK;
    constexpr int down_groups = 4096 / kOutputsPerArrayPass;
    packed.resize(rocmfp2_expert_v5_bytes());
    size_t destination = 0;
    auto append_tile = [&](const std::vector<uint8_t> & source,
                           size_t tile_index) {
        std::memcpy(packed.data() + destination,
                    source.data() + tile_index * kV4PackedTileBytes,
                    kV4PackedTileBytes);
        destination += kV4PackedTileBytes;
    };
    for (int column = 0; column < kAieColumns; ++column) {
        for (int kt = 0; kt < gate_k_tiles; ++kt) {
            for (int row = 0; row < kAieRows; ++row) {
                const size_t tile =
                    (static_cast<size_t>(column) *
                         static_cast<size_t>(gate_k_tiles) +
                     static_cast<size_t>(kt)) * static_cast<size_t>(kAieRows) +
                    static_cast<size_t>(row);
                append_tile(gate_v4, tile);
            }
            for (int row = 0; row < kAieRows; ++row) {
                const size_t tile =
                    (static_cast<size_t>(column) *
                         static_cast<size_t>(gate_k_tiles) +
                     static_cast<size_t>(kt)) * static_cast<size_t>(kAieRows) +
                    static_cast<size_t>(row);
                append_tile(up_v4, tile);
            }
        }
        for (int group = 0; group < down_groups; ++group) {
            for (int kt = 0; kt < down_k_tiles; ++kt) {
                for (int row = 0; row < kAieRows; ++row) {
                    const size_t tile =
                        ((static_cast<size_t>(group) *
                              static_cast<size_t>(kAieColumns) +
                          static_cast<size_t>(column)) *
                             static_cast<size_t>(down_k_tiles) +
                         static_cast<size_t>(kt)) *
                            static_cast<size_t>(kAieRows) +
                        static_cast<size_t>(row);
                    append_tile(down_v4, tile);
                }
            }
        }
    }
    return destination == packed.size();
}

bool rocmfp2_gemv_raw_reference(const void * raw, size_t raw_bytes,
                                const float * input, int k, int n,
                                float scale, float * output) {
    const size_t expected = rocmfp2_projection_bytes(k, n);
    if (!raw || !input || !output || expected == 0 || raw_bytes < expected)
        return false;
    const auto * bytes = static_cast<const uint8_t *>(raw);
    const int blocks_per_row = k / kRocmfp2BlockWeights;
    for (int out = 0; out < n; ++out) {
        float sum = 0.0f;
        for (int block = 0; block < blocks_per_row; ++block) {
            const uint8_t * q = bytes +
                (static_cast<size_t>(out) * static_cast<size_t>(blocks_per_row) +
                 static_cast<size_t>(block)) *
                    static_cast<size_t>(kRocmfp2BlockBytes);
            for (int i = 0; i < kRocmfp2BlockWeights; ++i)
                sum += input[block * kRocmfp2BlockWeights + i] * decode(q, i);
        }
        output[out] = sum * scale;
    }
    return true;
}

bool rocmfp2_gemv_packed_reference(const void * packed, size_t packed_bytes,
                                   const float * input, int k, int n,
                                   float scale, float * output) {
    const size_t expected = rocmfp2_projection_bytes(k, n);
    if (!packed || !input || !output || !rocmfp2_supported_shape(k, n) ||
        packed_bytes < expected) return false;
    const auto * bytes = static_cast<const uint8_t *>(packed);
    const int k_tiles = k / kGemvTileK;
    for (int out = 0; out < n; ++out) {
        const int group = out / kOutputsPerArrayPass;
        const int within = out % kOutputsPerArrayPass;
        const int row = within / (kAieColumns * kGemvTileN);
        const int column = (within / kGemvTileN) % kAieColumns;
        const int lane = within % kGemvTileN;
        float sum = 0.0f;
        for (int kt = 0; kt < k_tiles; ++kt) {
            size_t tile_index = static_cast<size_t>(group);
            tile_index = tile_index * static_cast<size_t>(kAieColumns) +
                         static_cast<size_t>(column);
            tile_index = tile_index * static_cast<size_t>(k_tiles) +
                         static_cast<size_t>(kt);
            tile_index = tile_index * static_cast<size_t>(kAieRows) +
                         static_cast<size_t>(row);
            const size_t tile = tile_index * kPackedTileBytes;
            const uint8_t * row_blocks = bytes + tile +
                (size_t)lane * (kGemvTileK / kRocmfp2BlockWeights) *
                    kRocmfp2BlockBytes;
            for (int block = 0; block < kGemvTileK / kRocmfp2BlockWeights;
                 ++block) {
                const uint8_t * q = row_blocks + block * kRocmfp2BlockBytes;
                for (int i = 0; i < kRocmfp2BlockWeights; ++i) {
                    const int input_index = kt * kGemvTileK +
                                            block * kRocmfp2BlockWeights + i;
                    sum += input[input_index] * decode(q, i);
                }
            }
        }
        output[out] = sum * scale;
    }
    return true;
}

bool rocmfp2_gemv_v4_packed_reference(const void * packed,
                                      size_t packed_bytes,
                                      const float * input, int k, int n,
                                      float scale, float * output) {
    const size_t expected = rocmfp2_v4_projection_bytes(k, n);
    if (!packed || !input || !output || expected == 0 ||
        packed_bytes < expected) return false;
    const auto * bytes = static_cast<const uint8_t *>(packed);
    const int k_tiles = k / kGemvTileK;
    for (int out = 0; out < n; ++out) {
        const int group = out / kOutputsPerArrayPass;
        const int within = out % kOutputsPerArrayPass;
        const int row = within / (kAieColumns * kGemvTileN);
        const int column = (within / kGemvTileN) % kAieColumns;
        const int lane = within % kGemvTileN;
        float sum = 0.0f;
        for (int kt = 0; kt < k_tiles; ++kt) {
            size_t tile_index = static_cast<size_t>(group);
            tile_index = tile_index * static_cast<size_t>(kAieColumns) +
                         static_cast<size_t>(column);
            tile_index = tile_index * static_cast<size_t>(k_tiles) +
                         static_cast<size_t>(kt);
            tile_index = tile_index * static_cast<size_t>(kAieRows) +
                         static_cast<size_t>(row);
            const uint8_t * tile = bytes + tile_index * kV4PackedTileBytes;
            const uint8_t * scales = tile + kV4CodeTileBytes;
            const uint8_t * offsets =
                tile + kV4CodeTileBytes + kV4MetadataPlaneBytes;
            for (int block = 0;
                 block < kGemvTileK / kRocmfp2BlockWeights; ++block) {
                const size_t metadata =
                    static_cast<size_t>(block) * kGemvTileN +
                    static_cast<size_t>(lane);
                uint16_t scale_bits = 0;
                uint16_t offset_bits = 0;
                std::memcpy(&scale_bits, scales + metadata * sizeof(scale_bits),
                            sizeof(scale_bits));
                std::memcpy(&offset_bits,
                            offsets + metadata * sizeof(offset_bits),
                            sizeof(offset_bits));
                const float block_scale = bf16_to_float(scale_bits);
                const float block_offset = bf16_to_float(offset_bits);
                for (int i = 0; i < kRocmfp2BlockWeights; ++i) {
                    const int local_input = block * kRocmfp2BlockWeights + i;
                    const uint8_t pair = tile[
                        static_cast<size_t>(local_input) *
                            (kGemvTileN / kRocmfp2V4CodesPerByte) +
                        static_cast<size_t>(lane / 2)];
                    const uint8_t code = static_cast<uint8_t>(
                        (pair >> (4 * (lane & 1))) & 0x0fu);
                    sum += input[kt * kGemvTileK + local_input] *
                           (static_cast<float>(code) * block_scale -
                            block_offset);
                }
            }
        }
        output[out] = sum * scale;
    }
    return true;
}

}  // namespace ember::xdna2
