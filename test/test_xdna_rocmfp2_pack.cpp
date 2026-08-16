#include "rocmfp2_pack.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) do {                                            \
    if (cond) { ++g_pass; } else {                                      \
        ++g_fail; std::fprintf(stderr, "FAIL: %s\n", msg);             \
    }                                                                   \
} while (0)

static void fill_projection(std::vector<uint8_t> & raw, int k, int n) {
    const int blocks_per_row = k / ember::xdna2::kRocmfp2BlockWeights;
    for (int output = 0; output < n; ++output) {
        for (int block = 0; block < blocks_per_row; ++block) {
            uint8_t * q = raw.data() +
                ((size_t)output * (size_t)blocks_per_row + (size_t)block) *
                ember::xdna2::kRocmfp2BlockBytes;
            for (int byte = 0; byte < 8; ++byte) {
                q[byte] = (uint8_t)((output * 17 + block * 29 + byte * 43) & 0xff);
            }
            q[8] = (uint8_t)(8 + ((output + block) % 80));
            q[9] = (uint8_t)(1 + ((output * 3 + block) % 72));
        }
    }
}

int main() {
    using namespace ember::xdna2;

    CHECK(ue4m3_to_float(0) == 0.0f, "UE4M3 zero");
    CHECK(ue4m3_to_float(7) == 7.0f / 1024.0f, "UE4M3 denormal");
    CHECK(ue4m3_to_float(8) == 8.0f / 1024.0f, "UE4M3 first normal");
    CHECK(ue4m3_to_float(0x7e) == 224.0f, "UE4M3 largest finite");
    CHECK(ue4m3_to_float(0x7f) == 0.0f, "UE4M3 invalid maps to zero");

    CHECK(rocmfp2_supported_shape(128, 2048), "minimum array shape accepted");
    CHECK(rocmfp2_supported_shape(4096, 4096), "DeepSeek fused shape accepted");
    CHECK(!rocmfp2_supported_shape(127, 2048), "unaligned K rejected");
    CHECK(!rocmfp2_supported_shape(128, 1024), "partial array N rejected");

    constexpr int k = 256;
    constexpr int n = 2048;
    const size_t bytes = rocmfp2_projection_bytes(k, n);
    std::vector<uint8_t> raw(bytes);
    fill_projection(raw, k, n);

    std::vector<uint8_t> packed;
    std::string error;
    CHECK(pack_rocmfp2_gemv(raw.data(), raw.size(), k, n, packed, &error),
          "valid projection packs");
    CHECK(packed.size() == raw.size(), "packing is a lossless permutation");

    std::vector<float> input(k);
    for (int i = 0; i < k; ++i) input[(size_t)i] = (float)((i % 19) - 9) / 17.0f;
    std::vector<float> raw_output(n);
    std::vector<float> packed_output(n);
    CHECK(rocmfp2_gemv_raw_reference(raw.data(), raw.size(), input.data(), k, n,
                                     0.375f, raw_output.data()),
          "raw reference runs");
    CHECK(rocmfp2_gemv_packed_reference(packed.data(), packed.size(), input.data(),
                                        k, n, 0.375f, packed_output.data()),
          "packed reference runs");
    bool equal = true;
    for (int i = 0; i < n; ++i) {
        if (raw_output[(size_t)i] != packed_output[(size_t)i]) {
            equal = false;
            break;
        }
    }
    CHECK(equal, "packed order preserves every GEMV result exactly");

    std::vector<uint8_t> packed_v4;
    CHECK(pack_rocmfp2_gemv_v4(raw.data(), raw.size(), k, n,
                               packed_v4, &error),
          "valid projection packs for Gen4");
    CHECK(packed_v4.size() == rocmfp2_v4_projection_bytes(k, n),
          "Gen4 packing reports its expanded size");
    CHECK(packed_v4.size() == raw.size() * 2,
          "Gen4 vector layout is exactly twice the ROCMFP2 source size");
    std::vector<float> packed_v4_output(n);
    CHECK(rocmfp2_gemv_v4_packed_reference(
              packed_v4.data(), packed_v4.size(), input.data(), k, n,
              0.375f, packed_v4_output.data()),
          "Gen4 packed reference runs");
    bool v4_equal = true;
    for (int i = 0; i < n; ++i) {
        if (raw_output[(size_t)i] != packed_v4_output[(size_t)i]) {
            v4_equal = false;
            break;
        }
    }
    CHECK(v4_equal, "Gen4 nibble/BF16 layout preserves every GEMV result exactly");

    std::vector<uint8_t> rejected;
    CHECK(!pack_rocmfp2_gemv(raw.data(), raw.size() - 1, k, n,
                             rejected, &error),
          "truncated projection rejected");
    CHECK(!pack_rocmfp2_gemv_v4(raw.data(), raw.size() - 1, k, n,
                                rejected, &error),
          "truncated Gen4 projection rejected");

    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
