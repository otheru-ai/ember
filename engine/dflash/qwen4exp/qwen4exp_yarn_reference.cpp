#include "qwen4exp_yarn_reference.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace dflash::qwen4exp::yarn_reference {
namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;

double get_mscale(double scale, double multiplier = 1.0) {
    if (scale <= 1.0) {
        return 1.0;
    }
    return 0.1 * multiplier * std::log(scale) + 1.0;
}

bool finite_positive(double value) {
    return std::isfinite(value) && value > 0.0;
}

}  // namespace

Parameters official_one_million_parameters() {
    return Parameters{};
}

bool derive(const Parameters & parameters,
            DerivedParameters & derived,
            std::string & error) {
    error.clear();
    if (parameters.head_dimension == 0 ||
        !finite_positive(parameters.partial_rotary_factor) ||
        parameters.partial_rotary_factor > 1.0) {
        error = "YaRN head dimension and partial rotary factor are invalid";
        return false;
    }
    if (!finite_positive(parameters.rope_theta) ||
        !finite_positive(parameters.factor) ||
        parameters.factor < 1.0 ||
        parameters.original_max_position_embeddings == 0) {
        error = "YaRN theta, factor, and original context must be positive";
        return false;
    }
    if (!finite_positive(parameters.beta_fast) ||
        !finite_positive(parameters.beta_slow) ||
        parameters.beta_fast < parameters.beta_slow) {
        error = "YaRN beta_fast must be greater than or equal to beta_slow";
        return false;
    }

    const double raw_dimension =
        static_cast<double>(parameters.head_dimension) *
        parameters.partial_rotary_factor;
    const double rounded_dimension = std::round(raw_dimension);
    if (std::fabs(raw_dimension - rounded_dimension) > 1.0e-9 ||
        rounded_dimension < 2.0 ||
        rounded_dimension > static_cast<double>(std::numeric_limits<std::size_t>::max())) {
        error = "YaRN partial rotary dimension must be an even integer";
        return false;
    }
    const std::size_t rotary_dimension =
        static_cast<std::size_t>(rounded_dimension);
    if ((rotary_dimension % 2U) != 0U) {
        error = "YaRN partial rotary dimension must be an even integer";
        return false;
    }
    const std::size_t frequency_count = rotary_dimension / 2U;
    std::uint64_t section_sum = 0;
    for (std::uint32_t section : parameters.mrope_sections) {
        section_sum += section;
    }
    if (section_sum != frequency_count) {
        error = "Qwen4Exp M-RoPE sections must cover every rotary frequency";
        return false;
    }

    const auto correction_dimension = [&](double rotations) {
        return static_cast<double>(rotary_dimension) *
               std::log(static_cast<double>(parameters.original_max_position_embeddings) /
                        (rotations * kTwoPi)) /
               (2.0 * std::log(parameters.rope_theta));
    };
    double low = correction_dimension(parameters.beta_fast);
    double high = correction_dimension(parameters.beta_slow);
    if (parameters.truncate_correction_range) {
        low = std::floor(low);
        high = std::ceil(high);
    }
    low = std::max(low, 0.0);
    high = std::min(high, static_cast<double>(rotary_dimension - 1U));

    double attention_scaling = 1.0;
    if (parameters.attention_factor.has_value()) {
        attention_scaling = *parameters.attention_factor;
    } else if (parameters.mscale.has_value() &&
               parameters.mscale_all_dim.has_value() &&
               *parameters.mscale != 0.0 &&
               *parameters.mscale_all_dim != 0.0) {
        attention_scaling =
            get_mscale(parameters.factor, *parameters.mscale) /
            get_mscale(parameters.factor, *parameters.mscale_all_dim);
    } else {
        attention_scaling = get_mscale(parameters.factor);
    }
    if (!finite_positive(attention_scaling)) {
        error = "YaRN attention scaling must be positive and finite";
        return false;
    }

    derived.rotary_dimension = rotary_dimension;
    derived.frequency_count = frequency_count;
    derived.correction_low = low;
    derived.correction_high = high;
    derived.attention_scaling = attention_scaling;
    return true;
}

bool inverse_frequencies(const Parameters & parameters,
                         std::vector<float> & frequencies,
                         DerivedParameters * derived_out,
                         std::string & error) {
    DerivedParameters derived;
    if (!derive(parameters, derived, error)) {
        frequencies.clear();
        return false;
    }

    double ramp_high = derived.correction_high;
    if (derived.correction_low == ramp_high) {
        ramp_high += 0.001;
    }
    frequencies.resize(derived.frequency_count);
    for (std::size_t index = 0; index < derived.frequency_count; ++index) {
        // Transformers creates this entire tensor as torch.float.  Keeping the
        // arithmetic in float32 here is part of the oracle contract.
        const float exponent =
            (2.0F * static_cast<float>(index)) /
            static_cast<float>(derived.rotary_dimension);
        const float extrapolated =
            1.0F / std::pow(static_cast<float>(parameters.rope_theta), exponent);
        const float interpolated =
            extrapolated / static_cast<float>(parameters.factor);
        const float ramp = std::clamp(
            (static_cast<float>(index) - static_cast<float>(derived.correction_low)) /
                static_cast<float>(ramp_high - derived.correction_low),
            0.0F, 1.0F);
        frequencies[index] =
            interpolated * ramp + extrapolated * (1.0F - ramp);
    }
    if (derived_out != nullptr) {
        *derived_out = derived;
    }
    return true;
}

bool interleaved_frequencies(const Parameters & parameters,
                             const std::array<float, kPositionAxes> & positions,
                             std::vector<float> & frequencies,
                             std::string & error) {
    std::vector<float> inverse;
    if (!inverse_frequencies(parameters, inverse, nullptr, error)) {
        frequencies.clear();
        return false;
    }
    for (float position : positions) {
        if (!std::isfinite(position)) {
            error = "Qwen4Exp M-RoPE positions must be finite";
            frequencies.clear();
            return false;
        }
    }

    // This is intentionally phrased like Qwen4Exp's implementation: begin
    // with all-T, then overwrite the H and W stride-three slices.  In
    // particular, mrope_sections[0] does not define a contiguous chunk.
    frequencies.resize(inverse.size());
    for (std::size_t index = 0; index < inverse.size(); ++index) {
        frequencies[index] = inverse[index] * positions[0];
    }
    for (std::size_t axis = 1; axis < kPositionAxes; ++axis) {
        const std::size_t limit = std::min(
            inverse.size(),
            static_cast<std::size_t>(parameters.mrope_sections[axis]) *
                kPositionAxes);
        for (std::size_t index = axis; index < limit;
             index += kPositionAxes) {
            frequencies[index] = inverse[index] * positions[axis];
        }
    }
    return true;
}

bool cosine_sine(const Parameters & parameters,
                 const std::array<float, kPositionAxes> & positions,
                 std::vector<float> & cosine,
                 std::vector<float> & sine,
                 std::string & error) {
    DerivedParameters derived;
    std::vector<float> inverse;
    if (!inverse_frequencies(parameters, inverse, &derived, error)) {
        cosine.clear();
        sine.clear();
        return false;
    }
    std::vector<float> selected;
    if (!interleaved_frequencies(parameters, positions, selected, error)) {
        cosine.clear();
        sine.clear();
        return false;
    }

    cosine.resize(derived.rotary_dimension);
    sine.resize(derived.rotary_dimension);
    for (std::size_t index = 0; index < derived.frequency_count; ++index) {
        const float cos_value = std::cos(selected[index]) *
                                static_cast<float>(derived.attention_scaling);
        const float sin_value = std::sin(selected[index]) *
                                static_cast<float>(derived.attention_scaling);
        cosine[index] = cos_value;
        cosine[index + derived.frequency_count] = cos_value;
        sine[index] = sin_value;
        sine[index + derived.frequency_count] = sin_value;
    }
    return true;
}

}  // namespace dflash::qwen4exp::yarn_reference
