#include "qwen4exp_yarn_reference.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

namespace yarn = dflash::qwen4exp::yarn_reference;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(condition, message) do {                                      \
    if (condition) { ++g_pass; } else {                                     \
        ++g_fail; std::fprintf(stderr, "FAIL: %s\n", message);             \
    }                                                                        \
} while (0)

static bool close_enough(double actual, double expected,
                         double tolerance = 1.0e-10) {
    return std::fabs(actual - expected) <= tolerance;
}

static void test_official_parameters() {
    const yarn::Parameters parameters = yarn::official_one_million_parameters();
    CHECK(parameters.head_dimension == 256,
          "released Qwen attention head dimension is pinned");
    CHECK(close_enough(parameters.partial_rotary_factor, 0.25),
          "released Qwen partial rotary factor is pinned");
    CHECK(close_enough(parameters.rope_theta, 10000000.0),
          "released Qwen rope theta is pinned");
    CHECK(close_enough(parameters.factor, 4.0),
          "official 1M extension uses a static factor of four");
    CHECK(parameters.original_max_position_embeddings == 262144,
          "official YaRN origin is the native 256K context");
    const std::array<std::uint32_t, 3> expected_sections = {11, 11, 10};
    CHECK(parameters.mrope_sections == expected_sections,
          "released interleaved M-RoPE sections are pinned");
    CHECK(!parameters.attention_factor.has_value() &&
          !parameters.mscale.has_value() &&
          !parameters.mscale_all_dim.has_value(),
          "open weights do not invent cloud-only YaRN scaling metadata");

    yarn::DerivedParameters derived;
    std::string error;
    CHECK(yarn::derive(parameters, derived, error),
          "official YaRN parameters validate");
    CHECK(derived.rotary_dimension == 64 && derived.frequency_count == 32,
          "partial RoPE covers exactly 64 dimensions");
    CHECK(close_enough(derived.correction_low, 14.0) &&
          close_enough(derived.correction_high, 22.0),
          "Transformers beta defaults give the pinned correction range");
    CHECK(close_enough(derived.attention_scaling, 1.138629436111989),
          "default attention scaling is 1 + 0.1 ln(factor)");
}

static void test_inverse_frequency_boundaries() {
    const yarn::Parameters parameters = yarn::official_one_million_parameters();
    std::vector<float> frequencies;
    std::string error;
    CHECK(yarn::inverse_frequencies(parameters, frequencies, nullptr, error),
          "official inverse frequencies compute");
    CHECK(frequencies.size() == 32, "inverse frequency count is exact");

    // Independently generated from Transformers _compute_yarn_parameters at
    // revision 36deb0b53ed0863f4b4dfdea23dcaec7f3df3701.
    const std::array<std::pair<std::size_t, double>, 9> expected = {{
        {0, 1.0},
        {13, 0.0014330126577988267},
        {14, 0.0008659643353894353},
        {15, 0.00047423981595784426},
        {16, 0.00025693507632240653},
        {21, 0.000008759770935285792},
        {22, 0.000003849816039291909},
        {23, 0.000002326430149190128},
        {31, 0.00000004137042708340921},
    }};
    for (const auto & item : expected) {
        CHECK(close_enough(frequencies[item.first], item.second, 1.0e-12),
              "inverse frequency matches the pinned host vector");
    }
    CHECK(close_enough(frequencies[14],
                       1.0 / std::pow(10000000.0, 28.0 / 64.0), 2.0e-11),
          "lower boundary is fully extrapolated");
    CHECK(close_enough(frequencies[22],
                       0.25 / std::pow(10000000.0, 44.0 / 64.0), 2.0e-12),
          "upper boundary is fully interpolated");
}

static void test_interleaved_mrope() {
    const yarn::Parameters parameters = yarn::official_one_million_parameters();
    const std::array<float, 3> positions = {999999.0F, 12345.0F, 67.0F};
    std::vector<float> frequencies;
    std::string error;
    CHECK(yarn::interleaved_frequencies(parameters, positions, frequencies, error),
          "interleaved M-RoPE frequencies compute");
    CHECK(close_enough(frequencies[0], 999999.0),
          "frequency zero takes temporal position");
    CHECK(close_enough(frequencies[1], 7460.0390625, 1.0e-6),
          "frequency one takes height position");
    CHECK(close_enough(frequencies[2], 24.466665267944336, 2.0e-5),
          "frequency two takes width position");
    CHECK(close_enough(frequencies[29], 0.000007590402674395591, 1.0e-12),
          "last width frequency is index 29");
    CHECK(close_enough(frequencies[30], 0.06846042722463608, 1.0e-8),
          "index 30 returns to temporal position");
    CHECK(close_enough(frequencies[31], 0.0005107179167680442, 1.0e-10),
          "index 31 is the final height frequency");

    const std::array<float, 3> text_position = {524288.0F, 524288.0F, 524288.0F};
    std::vector<float> text_frequencies;
    CHECK(yarn::interleaved_frequencies(parameters, text_position,
                                        text_frequencies, error),
          "text-only equal-axis positions compute");
    std::vector<float> inverse;
    CHECK(yarn::inverse_frequencies(parameters, inverse, nullptr, error),
          "text-only comparison inverse frequencies compute");
    bool all_equal = true;
    for (std::size_t index = 0; index < inverse.size(); ++index) {
        all_equal = all_equal && close_enough(
            text_frequencies[index], inverse[index] * 524288.0F, 1.0e-6);
    }
    CHECK(all_equal, "equal T/H/W ids reduce interleaved M-RoPE to text RoPE");
}

static void test_cosine_sine_and_validation() {
    yarn::Parameters parameters = yarn::official_one_million_parameters();
    std::vector<float> cosine;
    std::vector<float> sine;
    std::string error;
    CHECK(yarn::cosine_sine(parameters, {0.0, 0.0, 0.0},
                            cosine, sine, error),
          "position-zero embedding computes");
    CHECK(cosine.size() == 64 && sine.size() == 64,
          "Qwen duplicates 32 phases into 64 rotary dimensions");
    bool scaled_identity = true;
    for (std::size_t index = 0; index < cosine.size(); ++index) {
        scaled_identity = scaled_identity &&
            close_enough(cosine[index], 1.138629436111989, 1.0e-7) &&
            close_enough(sine[index], 0.0);
    }
    CHECK(scaled_identity, "YaRN attention scaling multiplies both cos and sin");
    bool duplicated = true;
    for (std::size_t index = 0; index < 32; ++index) {
        duplicated = duplicated && close_enough(cosine[index], cosine[index + 32]) &&
                     close_enough(sine[index], sine[index + 32]);
    }
    CHECK(duplicated, "phase vectors use Transformers cat(freqs, freqs) layout");

    parameters.mscale = 2.0;
    parameters.mscale_all_dim = 1.0;
    yarn::DerivedParameters derived;
    CHECK(yarn::derive(parameters, derived, error) &&
          close_enough(derived.attention_scaling,
                       (1.0 + 0.2 * std::log(4.0)) /
                       (1.0 + 0.1 * std::log(4.0))),
          "explicit mscale pair uses the Transformers ratio");
    parameters.attention_factor = 1.25;
    CHECK(yarn::derive(parameters, derived, error) &&
          close_enough(derived.attention_scaling, 1.25),
          "explicit attention factor has highest precedence");

    parameters = yarn::official_one_million_parameters();
    parameters.mrope_sections = {12, 10, 9};
    CHECK(!yarn::derive(parameters, derived, error) && !error.empty(),
          "invalid M-RoPE section coverage fails closed");
    parameters = yarn::official_one_million_parameters();
    parameters.beta_fast = 0.5;
    CHECK(!yarn::derive(parameters, derived, error) && !error.empty(),
          "reversed beta range fails closed");
}

int main() {
    test_official_parameters();
    test_inverse_frequency_boundaries();
    test_interleaved_mrope();
    test_cosine_sine_and_validation();
    std::printf("qwen4exp YaRN reference: %d passed, %d failed\n",
                g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
