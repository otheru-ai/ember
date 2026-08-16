// ROCMFP2 weight pre-tiling shared by the XRT provider and host tests.
//
// GGML stores an expert projection output-major: each output row contains
// consecutive 32-weight ROCMFP2 blocks. The XDNA2 design consumes 128x64
// tiles distributed over the 4x8 AIE array. This module performs only that
// lossless permutation; it never requantizes model weights.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ember::xdna2 {

constexpr int kRocmfp2BlockWeights = 32;
constexpr int kRocmfp2BlockBytes = 10;
constexpr int kGemvTileK = 128;
constexpr int kGemvTileN = 64;
constexpr int kAieRows = 4;
constexpr int kAieColumns = 8;
constexpr int kOutputsPerArrayPass = kGemvTileN * kAieRows * kAieColumns;
constexpr size_t kPackedTileBytes =
    (size_t)kGemvTileN * (kGemvTileK / kRocmfp2BlockWeights) *
    kRocmfp2BlockBytes;

// Gen4 trades a bounded 2x cache expansion for vector-native decoding. Each
// pair of FP2 codes occupies the low/high nibble of one byte, followed by
// block-major BF16 scale and offset planes. A tile remains small enough for a
// compute tile to expand its 128x64 weights into a 16 KiB BF16 scratchpad.
constexpr int kRocmfp2V4CodesPerByte = 2;
constexpr size_t kV4CodeTileBytes =
    (size_t)kGemvTileK * kGemvTileN / kRocmfp2V4CodesPerByte;
constexpr size_t kV4MetadataPlaneBytes =
    (size_t)(kGemvTileK / kRocmfp2BlockWeights) * kGemvTileN *
    sizeof(uint16_t);
constexpr size_t kV4PackedTileBytes =
    kV4CodeTileBytes + 2 * kV4MetadataPlaneBytes;

bool rocmfp2_supported_shape(int k, int n);
size_t rocmfp2_projection_bytes(int k, int n);

bool pack_rocmfp2_gemv(const void * raw, size_t raw_bytes, int k, int n,
                       std::vector<uint8_t> & packed,
                       std::string * error = nullptr);

size_t rocmfp2_v4_projection_bytes(int k, int n);
bool pack_rocmfp2_gemv_v4(const void * raw, size_t raw_bytes, int k, int n,
                          std::vector<uint8_t> & packed,
                          std::string * error = nullptr);

float ue4m3_to_float(uint8_t value);

// Deterministic references used to validate the lossless permutation and the
// AIE kernel's arithmetic contract. Both accumulate in F32.
bool rocmfp2_gemv_raw_reference(const void * raw, size_t raw_bytes,
                                const float * input, int k, int n,
                                float scale, float * output);
bool rocmfp2_gemv_packed_reference(const void * packed, size_t packed_bytes,
                                   const float * input, int k, int n,
                                   float scale, float * output);
bool rocmfp2_gemv_v4_packed_reference(const void * packed,
                                      size_t packed_bytes,
                                      const float * input, int k, int n,
                                      float scale, float * output);

}  // namespace ember::xdna2
