#include "rocmfp4_pack.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) do {                                        \
    if (cond) { ++g_pass; } else {                                  \
        ++g_fail; std::fprintf(stderr, "FAIL: %s\n", msg);         \
    }                                                               \
} while (0)

static void fill_projection(std::vector<uint8_t> & raw, int k, int n) {
    using namespace ember::xdna2;
    const int blocks_per_row = k / kRocmfp4BlockWeights;
    for (int output = 0; output < n; ++output) {
        for (int block = 0; block < blocks_per_row; ++block) {
            uint8_t * q = raw.data() +
                (static_cast<size_t>(output) *
                     static_cast<size_t>(blocks_per_row) +
                 static_cast<size_t>(block)) * kRocmfp4BlockBytes;
            for (int byte = 0; byte < 16; ++byte) {
                const uint8_t low = static_cast<uint8_t>(
                    (output + block * 3 + byte * 5) & 0x0f);
                const uint8_t high = static_cast<uint8_t>(
                    (output * 7 + block + byte * 11) & 0x0f);
                q[byte] = static_cast<uint8_t>(low | (high << 4));
            }
            q[16] = static_cast<uint8_t>(8 + ((output + block) % 96));
        }
    }
}

int main() {
    using namespace ember::xdna2;
    CHECK(rocmfp4_supported_shape(128, 2048),
          "minimum whole-array shape accepted");
    CHECK(rocmfp4_supported_shape(4096, 4096),
          "DSpark expert shapes accepted");
    CHECK(!rocmfp4_supported_shape(127, 2048),
          "unaligned K rejected");
    CHECK(!rocmfp4_supported_shape(128, 1024),
          "partial array N rejected");

    constexpr int k = 256;
    constexpr int n = 2048;
    std::vector<uint8_t> raw(rocmfp4_projection_bytes(k, n));
    fill_projection(raw, k, n);
    std::vector<uint8_t> packed;
    std::string error;
    CHECK(pack_rocmfp4_gemm(raw.data(), raw.size(), k, n,
                            packed, &error),
          "valid ROCMFP4 projection packs");
    CHECK(packed.size() == rocmfp4_packed_projection_bytes(k, n),
          "packed size matches tile contract");
    CHECK(packed.size() * 17 == raw.size() * 18,
          "BF16 scale expansion costs exactly one byte per block");

    std::vector<float> input(k);
    for (int i = 0; i < k; ++i)
        input[static_cast<size_t>(i)] = static_cast<float>((i % 23) - 11) / 19.0f;
    std::vector<float> raw_output(n);
    std::vector<float> cpu_output(n);
    std::vector<float> packed_output(n);
    CHECK(rocmfp4_gemm_raw_reference(
              raw.data(), raw.size(), input.data(), k, n,
              0.625f, raw_output.data()),
          "raw ROCMFP4 reference runs");
    CHECK(rocmfp4_gemm_cpu(
              raw.data(), raw.size(), input.data(), k, n,
              0.625f, cpu_output.data()),
          "host ROCMFP4 decoder runs");
    CHECK(rocmfp4_gemm_packed_reference(
              packed.data(), packed.size(), input.data(), k, n,
              0.625f, packed_output.data()),
          "packed ROCMFP4 reference runs");
    bool equal = true;
    for (int i = 0; i < n; ++i) {
        if (raw_output[static_cast<size_t>(i)] !=
            packed_output[static_cast<size_t>(i)]) {
            equal = false;
            break;
        }
    }
    CHECK(equal, "packing preserves every signed-codebook GEMM result");
    float cpu_max_abs = 0.0f;
    for (int i = 0; i < n; ++i) {
        cpu_max_abs = std::max(
            cpu_max_abs,
            std::fabs(raw_output[static_cast<size_t>(i)] -
                      cpu_output[static_cast<size_t>(i)]));
    }
    std::printf("host ROCMFP4 max abs: %.9g\n", cpu_max_abs);
    CHECK(cpu_max_abs <= 3.0e-4f,
          "host ROCMFP4 decoder preserves scalar arithmetic");

    std::vector<uint8_t> rejected;
    CHECK(!pack_rocmfp4_gemm(raw.data(), raw.size() - 1, k, n,
                             rejected, &error),
          "truncated projection rejected");

    const size_t first = rocmfp4_projection_bytes(4096, 2048);
    const size_t down = rocmfp4_projection_bytes(2048, 4096);
    std::vector<uint8_t> gate(first, 0x11);
    std::vector<uint8_t> up(first, 0x22);
    std::vector<uint8_t> down_raw(down, 0x33);
    std::vector<uint8_t> expert;
    CHECK(pack_rocmfp4_expert_v7(
              gate.data(), gate.size(), up.data(), up.size(),
              down_raw.data(), down_raw.size(), expert, &error),
          "Gen7 DSpark expert packs");
    CHECK(expert.size() == rocmfp4_expert_v7_bytes(),
          "Gen7 expert has exact packed size");
    CHECK(!pack_rocmfp4_expert_v7(
              gate.data(), gate.size() - 1, up.data(), up.size(),
              down_raw.data(), down_raw.size(), expert, &error),
          "short Gen7 gate projection rejected");

    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
