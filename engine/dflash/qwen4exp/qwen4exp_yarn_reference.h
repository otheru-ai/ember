// GPU-free YaRN and interleaved M-RoPE oracle for Qwen3.8-Flash-Next.
//
// The released checkpoint is native to 262,144 tokens.  Its official model
// card recommends static YaRN only when a deployment needs a longer context;
// importantly, that changes every sequence, including short ones.  This file
// keeps the extension math separate from the runtime graph so loader and HIP
// implementations can be checked against a small, deterministic host oracle.
//
// Sources pinned for audit:
// - Qwen/Qwen3.8-Flash-Next config and README, revision
//   f5d08274bafd880402bd16f5e3e6c514136ec06c.
// - huggingface/transformers Qwen4ExpTextRotaryEmbedding and
//   _compute_yarn_parameters, revision
//   36deb0b53ed0863f4b4dfdea23dcaec7f3df3701.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace dflash::qwen4exp::yarn_reference {

constexpr std::size_t kPositionAxes = 3;

struct Parameters {
    std::uint32_t head_dimension = 256;
    double partial_rotary_factor = 0.25;
    double rope_theta = 10000000.0;
    double factor = 4.0;
    std::uint32_t original_max_position_embeddings = 262144;
    double beta_fast = 32.0;
    double beta_slow = 1.0;
    bool truncate_correction_range = true;
    std::array<std::uint32_t, kPositionAxes> mrope_sections = {11, 11, 10};

    // Transformers uses an explicit attention_factor first.  Otherwise, when
    // both mscale values are truthy it uses their ratio; with neither present
    // it uses get_mscale(factor).  The open checkpoint specifies none of them.
    std::optional<double> attention_factor;
    std::optional<double> mscale;
    std::optional<double> mscale_all_dim;
};

struct DerivedParameters {
    std::size_t rotary_dimension = 0;
    std::size_t frequency_count = 0;
    double correction_low = 0.0;
    double correction_high = 0.0;
    double attention_scaling = 1.0;
};

// The exact static-YaRN parameters in the official open-weight model card for
// extension toward 1,000,000 tokens.  The factor remains 4.0 (not 1M/262144).
Parameters official_one_million_parameters();

// Derives and validates all values used by the frequency calculation.
bool derive(const Parameters & parameters,
            DerivedParameters & derived,
            std::string & error);

// Returns rotary_dimension/2 inverse frequencies.  Like Transformers, the
// first correction dimensions extrapolate, the last interpolate by factor,
// and the dimensions between them use a linear ramp.
bool inverse_frequencies(const Parameters & parameters,
                         std::vector<float> & frequencies,
                         DerivedParameters * derived,
                         std::string & error);

// Qwen4Exp first computes three independent T/H/W frequency planes, then
// chooses their dimensions in the interleaved [T,H,W,...,T,H] layout.  Text
// supplies equal position ids on all three axes and therefore reduces to RoPE.
bool interleaved_frequencies(const Parameters & parameters,
                             const std::array<float, kPositionAxes> & positions,
                             std::vector<float> & frequencies,
                             std::string & error);

// Matches Qwen4ExpTextRotaryEmbedding.forward: duplicate the selected half
// frequencies and multiply both cos and sin by YaRN attention_scaling.
bool cosine_sine(const Parameters & parameters,
                 const std::array<float, kPositionAxes> & positions,
                 std::vector<float> & cosine,
                 std::vector<float> & sine,
                 std::string & error);

}  // namespace dflash::qwen4exp::yarn_reference
