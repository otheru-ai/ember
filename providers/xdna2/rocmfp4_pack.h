// ROCMFP4_FAST weight pre-tiling for the XDNA2 DSpark prototype.
//
// GGUF stores one output-major 32-weight block as 16 packed signed-codebook
// nibbles followed by a UE4M3 scale byte. The AIE graph consumes 128x64
// K-major tiles. Packing is a lossless permutation plus exact UE4M3-to-BF16
// scale expansion; it does not requantize the trained draft weights.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ember::xdna2 {

constexpr int kRocmfp4BlockWeights = 32;
constexpr int kRocmfp4BlockBytes = 17;
constexpr int kRocmfp4TileK = 128;
constexpr int kRocmfp4TileN = 64;
constexpr int kRocmfp4AieRows = 4;
constexpr int kRocmfp4AieColumns = 8;
constexpr int kRocmfp4OutputsPerPass =
    kRocmfp4TileN * kRocmfp4AieRows * kRocmfp4AieColumns;
constexpr size_t kRocmfp4CodeTileBytes =
    (size_t)kRocmfp4TileK * kRocmfp4TileN / 2;
constexpr size_t kRocmfp4ScalePlaneBytes =
    (size_t)(kRocmfp4TileK / kRocmfp4BlockWeights) *
    kRocmfp4TileN * sizeof(uint16_t);
constexpr size_t kRocmfp4PackedTileBytes =
    kRocmfp4CodeTileBytes + kRocmfp4ScalePlaneBytes;

bool rocmfp4_supported_shape(int k, int n);
size_t rocmfp4_projection_bytes(int k, int n);
size_t rocmfp4_packed_projection_bytes(int k, int n);

bool pack_rocmfp4_gemm(const void * raw, size_t raw_bytes, int k, int n,
                       std::vector<uint8_t> & packed,
                       std::string * error = nullptr);

size_t rocmfp4_expert_v7_bytes();
bool pack_rocmfp4_expert_v7(const void * gate, size_t gate_bytes,
                            const void * up, size_t up_bytes,
                            const void * down, size_t down_bytes,
                            std::vector<uint8_t> & packed,
                            std::string * error = nullptr);

bool rocmfp4_gemm_raw_reference(const void * raw, size_t raw_bytes,
                                const float * input, int k, int n,
                                float scale, float * output);
bool rocmfp4_gemm_packed_reference(const void * packed, size_t packed_bytes,
                                   const float * input, int k, int n,
                                   float scale, float * output);

}  // namespace ember::xdna2
