// GPU-free oracle for the exact mixed-signedness I4 arithmetic proposed for
// gfx1151 ROCMI4 kernels. This tests integer identity only; production dispatch
// remains on the existing exact I8 path.
#include "rocmi4_exact.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace {

int g_pass;
int g_fail;

#define CHECK(condition, message)                                              \
    do {                                                                       \
        if (condition) {                                                       \
            ++g_pass;                                                          \
        } else {                                                               \
            ++g_fail;                                                          \
            std::printf("FAIL: %s\n", message);                               \
        }                                                                      \
    } while (0)

constexpr std::size_t kTileK = 32;
constexpr std::size_t kPackedValues = 8;

int sign_extend_i4(std::uint32_t nibble) {
    nibble &= 0x0fu;
    return nibble < 8u ? static_cast<int>(nibble)
                       : static_cast<int>(nibble) - 16;
}

std::uint32_t pack_signed_i4(const std::array<std::int8_t, kTileK> & values,
                             std::size_t base) {
    std::uint32_t packed = 0;
    for (std::size_t i = 0; i < kPackedValues; ++i) {
        packed |= (static_cast<std::uint32_t>(values[base + i]) & 0x0fu)
                  << (4 * i);
    }
    return packed;
}

std::uint32_t pack_low_u4(const std::array<std::int8_t, kTileK> & values,
                          std::size_t base) {
    std::uint32_t words[2] = {0, 0};
    for (std::size_t i = 0; i < kPackedValues; ++i) {
        words[i / 4] |= static_cast<std::uint32_t>(
                            static_cast<std::uint8_t>(values[base + i]))
                        << (8 * (i % 4));
    }
    return rocmi4_pack_q8x8_low_u4(words[0], words[1]);
}

std::uint32_t pack_high_i4(const std::array<std::int8_t, kTileK> & values,
                           std::size_t base) {
    std::uint32_t words[2] = {0, 0};
    for (std::size_t i = 0; i < kPackedValues; ++i) {
        words[i / 4] |= static_cast<std::uint32_t>(
                            static_cast<std::uint8_t>(values[base + i]))
                        << (8 * (i % 4));
    }
    return rocmi4_pack_q8x8_high_i4(words[0], words[1]);
}

int dot8_signed_unsigned_i4(std::uint32_t weights,
                            std::uint32_t activations) {
    int sum = 0;
    for (std::size_t i = 0; i < kPackedValues; ++i) {
        const int weight = sign_extend_i4(weights >> (4 * i));
        const int activation =
            static_cast<int>((activations >> (4 * i)) & 0x0fu);
        sum += weight * activation;
    }
    return sum;
}

int dot8_signed_signed_i4(std::uint32_t weights,
                          std::uint32_t activations) {
    int sum = 0;
    for (std::size_t i = 0; i < kPackedValues; ++i) {
        const int weight = sign_extend_i4(weights >> (4 * i));
        const int activation = sign_extend_i4(activations >> (4 * i));
        sum += weight * activation;
    }
    return sum;
}

int direct_dot_k32(const std::array<std::int8_t, kTileK> & weights,
                   const std::array<std::int8_t, kTileK> & activations) {
    int sum = 0;
    for (std::size_t i = 0; i < kTileK; ++i) {
        sum += static_cast<int>(weights[i]) *
               static_cast<int>(activations[i]);
    }
    return sum;
}

int decomposed_dot_k32(const std::array<std::int8_t, kTileK> & weights,
                       const std::array<std::int8_t, kTileK> & activations) {
    int sum = 0;
    for (std::size_t base = 0; base < kTileK; base += kPackedValues) {
        const std::uint32_t packed_weights = pack_signed_i4(weights, base);
        const int low =
            dot8_signed_unsigned_i4(packed_weights,
                                    pack_low_u4(activations, base));
        const int high =
            dot8_signed_signed_i4(packed_weights,
                                  pack_high_i4(activations, base));
        sum += low + 16 * high;
    }
    return sum;
}

std::uint32_t next_random(std::uint32_t & state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

void check_endpoints() {
    constexpr std::array<int, 7> endpoints = {
        -128, -127, -1, 0, 1, 126, 127,
    };
    constexpr std::array<int, 7> expected_low = {
        0, 1, 15, 0, 1, 14, 15,
    };
    constexpr std::array<int, 7> expected_high = {
        -8, -8, -1, 0, 0, 7, 7,
    };

    bool ok = true;
    for (std::size_t i = 0; i < endpoints.size(); ++i) {
        const rocmi4_q8_i4_parts parts =
            rocmi4_q8_decompose_i4(static_cast<std::int8_t>(endpoints[i]));
        ok = ok && static_cast<int>(parts.low_u4) == expected_low[i] &&
             static_cast<int>(parts.high_i4) == expected_high[i] &&
             rocmi4_q8_recompose_i4(parts) == endpoints[i];
    }
    CHECK(ok, "q8 endpoints have the exact low-u4/high-i4 representation");
}

void check_all_q8_values() {
    bool ok = true;
    for (int activation = -128; activation <= 127; ++activation) {
        const rocmi4_q8_i4_parts parts =
            rocmi4_q8_decompose_i4(static_cast<std::int8_t>(activation));
        ok = ok && parts.low_u4 <= 15u && parts.high_i4 >= -8 &&
             parts.high_i4 <= 7 &&
             rocmi4_q8_recompose_i4(parts) == activation;
    }
    CHECK(ok, "all 256 signed q8 values recompose as low + 16*high");
}

void check_fragment_packing() {
    bool ok = true;
    for (int activation = -128; activation <= 127; ++activation) {
        const std::uint32_t byte = static_cast<std::uint32_t>(
            static_cast<std::uint8_t>(activation));
        const std::uint32_t word = byte * UINT32_C(0x01010101);
        const rocmi4_q8_i4_parts parts =
            rocmi4_q8_decompose_i4(static_cast<std::int8_t>(activation));
        const std::uint32_t low = static_cast<std::uint32_t>(parts.low_u4) *
                                  UINT32_C(0x11111111);
        const std::uint32_t high =
            (static_cast<std::uint32_t>(parts.high_i4) & 0x0fu) *
            UINT32_C(0x11111111);
        ok = ok && rocmi4_pack_q8x8_low_u4(word, word) == low &&
             rocmi4_pack_q8x8_high_i4(word, word) == high;
    }
    CHECK(ok, "q8 fragment packing preserves all low/high nibble bit patterns");
}

void check_runtime_opt_in() {
    CHECK(!rocmi4_w4a8_iu4_requested(nullptr) &&
              !rocmi4_w4a8_iu4_requested("") &&
              !rocmi4_w4a8_iu4_requested("0") &&
              !rocmi4_w4a8_iu4_requested("true") &&
              !rocmi4_w4a8_iu4_requested("1 ") &&
              rocmi4_w4a8_iu4_requested("1"),
          "only the exact runtime value 1 requests the W4A8 IU4 experiment");
}

void check_exhaustive_scalar_products() {
    bool ok = true;
    for (int weight = -8; weight <= 7; ++weight) {
        for (int activation = -128; activation <= 127; ++activation) {
            const rocmi4_q8_i4_parts parts =
                rocmi4_q8_decompose_i4(static_cast<std::int8_t>(activation));
            const int direct = weight * activation;
            const int decomposed =
                weight * static_cast<int>(parts.low_u4) +
                16 * weight * static_cast<int>(parts.high_i4);
            ok = ok && direct == decomposed;
        }
    }
    CHECK(ok, "all 4096 signed-I4 by signed-q8 scalar products are exact");
}

void check_adversarial_tiles() {
    std::array<std::int8_t, kTileK> weights{};
    std::array<std::int8_t, kTileK> activations{};

    weights.fill(-8);
    activations.fill(-128);
    CHECK(direct_dot_k32(weights, activations) == 32768 &&
          decomposed_dot_k32(weights, activations) == 32768,
          "maximum positive K32 tile is exact and fits int32");

    activations.fill(127);
    CHECK(direct_dot_k32(weights, activations) == -32512 &&
          decomposed_dot_k32(weights, activations) == -32512,
          "large negative K32 tile is exact and fits int32");

    for (std::size_t i = 0; i < kTileK; ++i) {
        weights[i] = static_cast<std::int8_t>(
            static_cast<int>(i % 16u) - 8);
        activations[i] = static_cast<std::int8_t>(
            i % 4u == 0u ? -128
                         : (i % 4u == 1u ? -1 : (i % 4u == 2u ? 0 : 127)));
    }
    CHECK(direct_dot_k32(weights, activations) ==
              decomposed_dot_k32(weights, activations),
          "alternating endpoint K32 tile is exact through packed DOT8 oracles");
}

void check_random_tiles() {
    constexpr int trials = 32768;
    std::uint32_t state = 0x6d2b79f5u;
    bool ok = true;
    for (int trial = 0; trial < trials; ++trial) {
        std::array<std::int8_t, kTileK> weights{};
        std::array<std::int8_t, kTileK> activations{};
        for (std::size_t i = 0; i < kTileK; ++i) {
            weights[i] = static_cast<std::int8_t>(
                static_cast<int>(next_random(state) & 0x0fu) - 8);
            activations[i] = static_cast<std::int8_t>(
                next_random(state) & 0xffu);
        }
        if (direct_dot_k32(weights, activations) !=
            decomposed_dot_k32(weights, activations)) {
            ok = false;
            break;
        }
    }
    CHECK(ok, "32768 deterministic random K32 tiles match the direct q8 dot");
}

} // namespace

int main() {
    check_endpoints();
    check_all_q8_values();
    check_fragment_packing();
    check_runtime_opt_in();
    check_exhaustive_scalar_products();
    check_adversarial_tiles();
    check_random_tiles();

    std::printf("%s: %d passed, %d failed\n",
                g_fail == 0 ? "PASS" : "FAIL", g_pass, g_fail);
    return g_fail != 0;
}
