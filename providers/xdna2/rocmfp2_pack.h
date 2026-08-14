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

bool rocmfp2_supported_shape(int k, int n);
size_t rocmfp2_projection_bytes(int k, int n);

bool pack_rocmfp2_gemv(const void * raw, size_t raw_bytes, int k, int n,
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

}  // namespace ember::xdna2
