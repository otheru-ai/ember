// GPU-free oracle for the exact mixed-signedness I4 arithmetic proposed for
// gfx1151 ROCMI4 kernels. This tests integer identity only; production dispatch
// remains on the existing exact I8 path.
#include "rocmi4_exact.h"
#include "w4a4_grid.h"

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

int prepacked_dot_k32(const std::array<std::int8_t, kTileK> & weights,
                      const std::array<std::int8_t, kTileK> & activations) {
    std::array<std::uint32_t, kTileK/4> q8_words{};
    for (std::size_t k = 0; k < kTileK; ++k) {
        q8_words[k/4] |= static_cast<std::uint32_t>(
                             static_cast<std::uint8_t>(activations[k]))
                         << (8*(k % 4));
    }
    const rocmi4_q8x16_i4_prepack first = rocmi4_prepack_q8x16_i4(
        q8_words[0], q8_words[1], q8_words[2], q8_words[3]);
    const rocmi4_q8x16_i4_prepack second = rocmi4_prepack_q8x16_i4(
        q8_words[4], q8_words[5], q8_words[6], q8_words[7]);
    const std::array<std::uint32_t, 8> lds = {
        first.high_i4[0], first.high_i4[1],
        second.high_i4[0], second.high_i4[1],
        first.low_u4[0], first.low_u4[1],
        second.low_u4[0], second.low_u4[1],
    };

    int sum = 0;
    for (std::size_t base = 0; base < kTileK; base += kPackedValues) {
        const std::size_t word = base/kPackedValues;
        const std::uint32_t packed_weights = pack_signed_i4(weights, base);
        sum += dot8_signed_unsigned_i4(packed_weights, lds[4 + word]) +
               16*dot8_signed_signed_i4(packed_weights, lds[word]);
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

void check_activation_prepack_layout() {
    std::array<std::int8_t, kTileK> activations{};
    std::array<std::uint32_t, kTileK/4> q8_words{};
    constexpr std::array<int, 8> seeds = {
        -128, -97, -17, -1, 0, 23, 106, 127,
    };
    for (std::size_t k = 0; k < kTileK; ++k) {
        activations[k] = static_cast<std::int8_t>(
            seeds[k % seeds.size()] + static_cast<int>(k/seeds.size()));
        q8_words[k/4] |= static_cast<std::uint32_t>(
                             static_cast<std::uint8_t>(activations[k]))
                         << (8*(k % 4));
    }

    const rocmi4_q8x16_i4_prepack first = rocmi4_prepack_q8x16_i4(
        q8_words[0], q8_words[1], q8_words[2], q8_words[3]);
    const rocmi4_q8x16_i4_prepack second = rocmi4_prepack_q8x16_i4(
        q8_words[4], q8_words[5], q8_words[6], q8_words[7]);
    const std::array<std::uint32_t, 8> lds = {
        first.high_i4[0], first.high_i4[1],
        second.high_i4[0], second.high_i4[1],
        first.low_u4[0], first.low_u4[1],
        second.low_u4[0], second.low_u4[1],
    };

    bool ok = true;
    for (std::size_t k = 0; k < kTileK; ++k) {
        const std::size_t word = k/kPackedValues;
        const std::size_t nibble = k % kPackedValues;
        const int high = sign_extend_i4(lds[word] >> (4*nibble));
        const int low = static_cast<int>(
            (lds[4 + word] >> (4*nibble)) & 0x0fu);
        ok = ok && low + 16*high == static_cast<int>(activations[k]);
    }
    CHECK(ok,
          "cooperative K32 activation prepack stores contiguous high then low fragments");
}

void check_split_half_weight_fragment_order() {
    std::array<std::int8_t, kTileK> weights{};
    std::array<std::uint8_t, kTileK / 2> storage{};
    for (std::size_t k = 0; k < kTileK; ++k) {
        weights[k] = static_cast<std::int8_t>(
            k < storage.size() ? static_cast<int>(k) - 8
                               : 7 - static_cast<int>(k - storage.size()));
    }
    for (std::size_t k = 0; k < storage.size(); ++k) {
        storage[k] = static_cast<std::uint8_t>(
            (static_cast<std::uint8_t>(weights[k]) & 0x0fu) |
            ((static_cast<std::uint8_t>(weights[k + storage.size()]) & 0x0fu) << 4));
    }

    bool ok = true;
    for (std::size_t half = 0; half < 2; ++half) {
        std::uint32_t first4 = 0;
        std::uint32_t next4 = 0;
        for (std::size_t byte = 0; byte < 4; ++byte) {
            first4 |= static_cast<std::uint32_t>(storage[8 * half + byte]) << (8 * byte);
            next4 |= static_cast<std::uint32_t>(storage[8 * half + 4 + byte]) << (8 * byte);
        }
        const std::uint32_t low = rocmi4_pack_split_half_low_i4(first4, next4);
        const std::uint32_t high = rocmi4_pack_split_half_high_i4(first4, next4);
        for (std::size_t nibble = 0; nibble < kPackedValues; ++nibble) {
            const std::size_t low_k = 8 * half + nibble;
            const std::size_t high_k = storage.size() + low_k;
            ok = ok && sign_extend_i4(low >> (4 * nibble)) == weights[low_k] &&
                 sign_extend_i4(high >> (4 * nibble)) == weights[high_k];
        }
    }
    CHECK(ok, "split-half ROCMI4 storage packs a K-contiguous gfx1151 A fragment");
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

void check_w4a4_grid_arithmetic() {
    CHECK(rocmi4_w4a4_scale(0.0f) == 0.0f &&
              rocmi4_w4a4_pack4(0.0f, 0.0f, 0.0f, 0.0f, 0.0f) == 0u,
          "W4A4 zero block keeps both scale and inverse scale at zero");

    const float d_inv = 7.0f / 7.0f;
    const std::uint32_t packed = rocmi4_w4a4_pack4(
        -8.0f, 7.0f, -9.0f, 9.0f, d_inv);
    CHECK((packed & 0x0fu) == 8u && ((packed >> 8) & 0x0fu) == 7u &&
              ((packed >> 16) & 0x0fu) == 8u &&
              ((packed >> 24) & 0x0fu) == 7u,
          "W4A4 clamp admits signed -8/+7 and clamps both directions");
    CHECK(rocmi4_w4a4_quantize_value(-7.0f, d_inv) == -7,
          "W4A4 symmetric scale never emits -8 for -amax");
    CHECK(rocmi4_w4a4_scale(7.0f) == 7.0f / (7.0f * 16.0f),
          "W4A4 scale preserves the sixteen-fold denominator");

    // quantize_mmq_q8_1 uses lane^1: each lane's four byte values occupy the
    // low nibbles while its partner's values occupy the high nibbles. This is
    // a stride-4-in-K pairing, distinct from the stride-16 ROCMI4 prepack.
    const std::uint32_t lane0 = rocmi4_w4a4_pack4(0, 1, 2, 3, 1.0f);
    const std::uint32_t lane1 = rocmi4_w4a4_pack4(4, 5, 6, 7, 1.0f);
    const std::uint32_t folded0 = rocmi4_w4a4_fold(lane0, lane1);
    const std::uint32_t lane2 = rocmi4_w4a4_pack4(-1, -2, -3, -4, 1.0f);
    const std::uint32_t lane3 = rocmi4_w4a4_pack4(-5, -6, -7, -8, 1.0f);
    const std::uint32_t folded1 = rocmi4_w4a4_fold(lane2, lane3);
    bool round_trip = true;
    for (std::size_t i = 0; i < 4; ++i) {
        round_trip = round_trip && ((folded0 >> (8 * i)) & 0x0fu) == i &&
                     ((folded0 >> (8 * i + 4)) & 0x0fu) == i + 4;
        const int low = static_cast<int>((folded1 >> (8 * i)) & 0x0fu);
        const int high = static_cast<int>((folded1 >> (8 * i + 4)) & 0x0fu);
        round_trip = round_trip && low == 15 - static_cast<int>(i) &&
                     high == 11 - static_cast<int>(i);
    }
    CHECK(round_trip, "W4A4 fold round-trips both lanes across two groups");
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
          decomposed_dot_k32(weights, activations) == 32768 &&
          prepacked_dot_k32(weights, activations) == 32768,
          "maximum positive K32 tile is exact and fits int32");

    activations.fill(127);
    CHECK(direct_dot_k32(weights, activations) == -32512 &&
          decomposed_dot_k32(weights, activations) == -32512 &&
          prepacked_dot_k32(weights, activations) == -32512,
          "large negative K32 tile is exact and fits int32");

    for (std::size_t i = 0; i < kTileK; ++i) {
        weights[i] = static_cast<std::int8_t>(
            static_cast<int>(i % 16u) - 8);
        activations[i] = static_cast<std::int8_t>(
            i % 4u == 0u ? -128
                         : (i % 4u == 1u ? -1 : (i % 4u == 2u ? 0 : 127)));
    }
    CHECK(direct_dot_k32(weights, activations) ==
              decomposed_dot_k32(weights, activations) &&
          direct_dot_k32(weights, activations) ==
              prepacked_dot_k32(weights, activations),
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
                decomposed_dot_k32(weights, activations) ||
            direct_dot_k32(weights, activations) !=
                prepacked_dot_k32(weights, activations)) {
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
    check_activation_prepack_layout();
    check_split_half_weight_fragment_order();
    check_runtime_opt_in();
    check_w4a4_grid_arithmetic();
    check_exhaustive_scalar_products();
    check_adversarial_tiles();
    check_random_tiles();

    std::printf("%s: %d passed, %d failed\n",
                g_fail == 0 ? "PASS" : "FAIL", g_pass, g_fail);
    return g_fail != 0;
}
