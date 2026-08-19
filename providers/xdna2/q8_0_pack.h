// Q8_0 pre-tiling for the XDNA2 DSpark dense/shared path.
//
// DSpark's attention, projections, and shared experts use ordinary GGML
// Q8_0: one F16 scale followed by 32 signed bytes.  Gen1 dequantizes to BF16
// cache tiles as an IRON-compatible baseline.  Gen2 stores every dequantized
// weight as a BF16 high term plus a BF16 residual. Both planes MAC directly
// into FP32 on AIE2P. This is deliberately wider than raw Q8: Peano lowers a
// vector FP32 multiply to BF16 inputs on this target, so applying an exact F32
// scale after a 32-input dot silently loses the precision being preserved.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ember::xdna2 {

constexpr int kQ8BlockWeights = 32;
constexpr int kQ8BlockBytes = 34;
constexpr int kQ8TileK = 128;
constexpr int kQ8TileN = 64;
constexpr int kQ8AieRows = 4;
constexpr int kQ8AieColumns = 8;
constexpr int kQ8OutputsPerRow = kQ8TileN * kQ8AieColumns;
constexpr int kQ8OutputsPerPass = kQ8TileN * kQ8AieRows * kQ8AieColumns;
constexpr size_t kQ8PackedTileBytes =
    static_cast<size_t>(kQ8TileK) * kQ8TileN * sizeof(uint16_t);
constexpr size_t kQ8CorrectedTileBytes = 2 * kQ8PackedTileBytes;

bool q8_supported_shape(int k, int n);
int q8_projection_rows(int n);
size_t q8_projection_bytes(int k, int n);
size_t q8_packed_projection_bytes(int k, int n);

bool pack_q8_gemm_bf16(const void * raw, size_t raw_bytes, int k, int n,
                       std::vector<uint8_t> & packed,
                       std::string * error = nullptr);

size_t q8_corrected_packed_projection_bytes(int k, int n);
bool pack_q8_gemm_corrected_bf16(const void * raw, size_t raw_bytes,
                                 int k, int n,
                                 std::vector<uint8_t> & packed,
                                 std::string * error = nullptr);
bool pack_q8_projection_corrected_bf16(const void * raw, size_t raw_bytes,
                                       int k, int n,
                                       std::vector<uint8_t> & packed,
                                       std::string * error = nullptr);
size_t q8_projection_task_packed_bytes(int k, int n);
// Expand a projection into a wider fixed-overlay row layout. Source and
// destination must have the same output-group count; newly active rows are
// zero-filled, so the original output lanes remain byte-exact.
bool pad_q8_projection_rows(const std::vector<uint8_t> & source,
                            int k, int source_n, int destination_n,
                            std::vector<uint8_t> & destination,
                            std::string * error = nullptr);
// Fuse two same-input, one-group projections into adjacent rows of one
// resident-overlay invocation. This is used for DSpark Q-a + KV.
bool concat_q8_projection_rows(const std::vector<uint8_t> & first,
                               int k, int first_n,
                               const std::vector<uint8_t> & second,
                               int second_n,
                               std::vector<uint8_t> & destination,
                               std::string * error = nullptr);
// Pack independent output groups that consume different activation rows into
// one descriptor sequence. DSpark output-A uses 8 x (4096 -> 1024), each
// zero-padded to the resident overlay's 2048-output group width.
bool pack_q8_grouped_projection_corrected_bf16(
    const void * raw, size_t raw_bytes, int k, int group_n, int groups,
    int padded_group_n, std::vector<uint8_t> & packed,
    std::string * error = nullptr);
bool pack_q8_gemm_m32_corrected_bf16(const void * raw, size_t raw_bytes,
                                     int k, int n,
                                     std::vector<uint8_t> & packed,
                                     std::string * error = nullptr);
size_t q8_gemm_m32_packed_bytes(int k, int n);

size_t q8_expert_v1_bytes();
bool pack_q8_expert_v1(const void * gate, size_t gate_bytes,
                       const void * up, size_t up_bytes,
                       const void * down, size_t down_bytes,
                       std::vector<uint8_t> & packed,
                       std::string * error = nullptr);

size_t q8_expert_v2_bytes();
bool pack_q8_expert_v2(const void * gate, size_t gate_bytes,
                       const void * up, size_t up_bytes,
                       const void * down, size_t down_bytes,
                       std::vector<uint8_t> & packed,
                       std::string * error = nullptr);

bool q8_gemm_raw_reference(const void * raw, size_t raw_bytes,
                           const float * input, int k, int n,
                           float * output);
bool q8_gemm_packed_bf16_reference(const void * packed, size_t packed_bytes,
                                   const float * input, int k, int n,
                                   float * output);

}  // namespace ember::xdna2
