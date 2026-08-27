#include "qwen4exp_reference.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace dflash::qwen4exp::reference {
namespace {

double sigmoid(double value) {
    if (value >= 0.0) {
        return 1.0 / (1.0 + std::exp(-value));
    }
    const double exponent = std::exp(value);
    return exponent / (1.0 + exponent);
}

double silu(double value) {
    return value * sigmoid(value);
}

void require_finite(double value, const char * name) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(name);
    }
}

std::vector<double> matrix_vector(const std::vector<double> & matrix,
                                  std::size_t rows,
                                  std::size_t columns,
                                  const std::vector<double> & vector) {
    if (columns != vector.size() ||
        (columns != 0 && rows > std::numeric_limits<std::size_t>::max() / columns) ||
        matrix.size() != rows * columns) {
        throw std::invalid_argument("invalid row-major matrix dimensions");
    }
    std::vector<double> output(rows, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            output[row] += matrix[row * columns + column] * vector[column];
        }
    }
    return output;
}

}  // namespace

const PleHashParameters & released_ple_hash_parameters() {
    // Direct checkpoint constants used by the official Transformers revision
    // 36deb0b53ed0863f4b4dfdea23dcaec7f3df3701. llama.cpp PR #27742's
    // qwen4exp loader/hash path is the GGUF-side provenance for the unsigned
    // storage and token-major 2-gram/3-gram head ordering.
    static const PleHashParameters parameters = {
        248046,
        {23703573157769ULL, 20109073645365ULL, 8052911324071ULL},
        {
            20000003ULL, 20000023ULL, 20000033ULL, 20000047ULL,
            20000059ULL, 20000063ULL, 20000069ULL, 20000077ULL,
            20000081ULL, 20000093ULL, 20000107ULL, 20000147ULL,
            20000153ULL, 20000159ULL, 20000161ULL, 20000171ULL,
        },
        {
            0ULL, 20000003ULL, 40000026ULL, 60000059ULL,
            80000106ULL, 100000165ULL, 120000228ULL, 140000297ULL,
            160000374ULL, 180000455ULL, 200000548ULL, 220000655ULL,
            240000802ULL, 260000955ULL, 280001114ULL, 300001275ULL,
        },
    };
    return parameters;
}

std::vector<std::uint64_t> ple_hash_indices(
    const std::vector<std::uint64_t> & input_ids,
    const std::vector<std::uint64_t> & previous_context,
    const PleHashParameters & parameters) {
    for (std::size_t head = 0; head < kPleHeadCount; ++head) {
        if (parameters.head_vocab_sizes[head] == 0) {
            throw std::invalid_argument("PLE head vocabulary size must be nonzero");
        }
    }

    std::array<std::uint64_t, kPleNgramSize - 1> history = {
        parameters.eos_token_id, parameters.eos_token_id,
    };
    const std::size_t retained = std::min(previous_context.size(), history.size());
    for (std::size_t i = 0; i < retained; ++i) {
        history[history.size() - retained + i] =
            previous_context[previous_context.size() - retained + i];
    }

    std::vector<std::uint64_t> tokens;
    tokens.reserve(history.size() + input_ids.size());
    tokens.insert(tokens.end(), history.begin(), history.end());
    tokens.insert(tokens.end(), input_ids.begin(), input_ids.end());

    std::vector<std::uint64_t> rows;
    rows.reserve(input_ids.size() * kPleHeadCount);
    for (std::size_t token_index = history.size(); token_index < tokens.size(); ++token_index) {
        std::array<std::uint64_t, kPleNgramSize> context{};
        context[0] = tokens[token_index];
        bool cut = false;
        for (std::size_t shift = 1; shift < kPleNgramSize; ++shift) {
            const std::uint64_t predecessor = cut
                ? parameters.eos_token_id
                : tokens[token_index - shift];
            context[shift] = predecessor;
            if (predecessor == parameters.eos_token_id) {
                cut = true;
            }
        }

        for (std::size_t order = 2; order <= kPleNgramSize; ++order) {
            std::uint64_t mixed = context[0] * parameters.multipliers[0];
            for (std::size_t position = 1; position < order; ++position) {
                mixed ^= context[position] * parameters.multipliers[position];
            }
            const std::size_t base = (order - 2) * kPleHeadsPerNgram;
            for (std::size_t local_head = 0; local_head < kPleHeadsPerNgram;
                 ++local_head) {
                const std::size_t head = base + local_head;
                rows.push_back(mixed % parameters.head_vocab_sizes[head] +
                               parameters.head_offsets[head]);
            }
        }
    }
    return rows;
}

std::vector<double> qsa_pool_complete_blocks(
    const std::vector<double> & raw_keys,
    std::size_t key_dimension,
    const std::vector<std::size_t> & visible_token_indices,
    std::size_t block_size) {
    // Qwen4ExpTextQSAIndexer.forward in Transformers revision
    // 36deb0b53ed0863f4b4dfdea23dcaec7f3df3701 pools only floor(V / r)
    // groups and leaves the V % r causal tail unpooled.
    if (key_dimension == 0 || block_size == 0 ||
        raw_keys.size() % key_dimension != 0) {
        throw std::invalid_argument("invalid QSA key or block dimensions");
    }
    const std::size_t key_count = raw_keys.size() / key_dimension;
    for (std::size_t index : visible_token_indices) {
        if (index >= key_count) {
            throw std::out_of_range("visible QSA token index exceeds key count");
        }
    }
    const std::size_t complete_blocks = visible_token_indices.size() / block_size;
    std::vector<double> pooled(complete_blocks * key_dimension, 0.0);
    for (std::size_t block = 0; block < complete_blocks; ++block) {
        for (std::size_t member = 0; member < block_size; ++member) {
            const std::size_t token = visible_token_indices[block * block_size + member];
            for (std::size_t dim = 0; dim < key_dimension; ++dim) {
                pooled[block * key_dimension + dim] +=
                    raw_keys[token * key_dimension + dim];
            }
        }
        for (std::size_t dim = 0; dim < key_dimension; ++dim) {
            pooled[block * key_dimension + dim] /= static_cast<double>(block_size);
        }
    }
    return pooled;
}

std::vector<double> qsa_score_blocks(
    const std::vector<double> & query_heads,
    std::size_t head_count,
    std::size_t head_dimension,
    const std::vector<double> & pooled_keys) {
    if (head_count == 0 || head_dimension == 0 ||
        head_count > std::numeric_limits<std::size_t>::max() / head_dimension ||
        query_heads.size() != head_count * head_dimension ||
        pooled_keys.size() % head_dimension != 0) {
        throw std::invalid_argument("invalid QSA score dimensions");
    }
    const std::size_t block_count = pooled_keys.size() / head_dimension;
    std::vector<double> scores(block_count, 0.0);
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dimension));
    for (std::size_t block = 0; block < block_count; ++block) {
        for (std::size_t head = 0; head < head_count; ++head) {
            double dot = 0.0;
            for (std::size_t dim = 0; dim < head_dimension; ++dim) {
                dot += query_heads[head * head_dimension + dim] *
                       pooled_keys[block * head_dimension + dim];
            }
            scores[block] += std::max(0.0, dot);
        }
        scores[block] *= scale;
    }
    return scores;
}

std::vector<std::size_t> qsa_select_tokens(
    const std::vector<double> & complete_block_scores,
    const std::vector<std::size_t> & visible_token_indices,
    std::size_t token_budget,
    std::size_t block_size) {
    if (block_size == 0 || token_budget == 0 || token_budget % block_size != 0) {
        throw std::invalid_argument("QSA token budget must contain whole blocks");
    }
    const std::size_t complete_blocks = visible_token_indices.size() / block_size;
    if (complete_block_scores.size() != complete_blocks) {
        throw std::invalid_argument("QSA score count does not match complete blocks");
    }
    for (double score : complete_block_scores) {
        require_finite(score, "QSA block scores must be finite");
    }

    std::vector<std::size_t> ranking(complete_blocks);
    std::iota(ranking.begin(), ranking.end(), std::size_t{0});
    std::stable_sort(ranking.begin(), ranking.end(),
                     [&](std::size_t left, std::size_t right) {
                         if (complete_block_scores[left] != complete_block_scores[right]) {
                             return complete_block_scores[left] > complete_block_scores[right];
                         }
                         return left < right;
                     });
    const std::size_t blocks_to_select =
        std::min(complete_blocks, token_budget / block_size);

    std::vector<std::size_t> selected;
    const std::size_t tail_size = visible_token_indices.size() % block_size;
    selected.reserve(blocks_to_select * block_size + tail_size);
    for (std::size_t rank = 0; rank < blocks_to_select; ++rank) {
        const std::size_t block = ranking[rank];
        const auto begin = visible_token_indices.begin() +
                           static_cast<std::ptrdiff_t>(block * block_size);
        selected.insert(selected.end(), begin,
                        begin + static_cast<std::ptrdiff_t>(block_size));
    }
    const auto tail_begin = visible_token_indices.begin() +
                            static_cast<std::ptrdiff_t>(complete_blocks * block_size);
    selected.insert(selected.end(), tail_begin, visible_token_indices.end());
    return selected;
}

std::vector<double> gated_residual_init(const std::vector<double> & input) {
    std::vector<double> output;
    output.reserve(kResidualStreams * input.size());
    for (std::size_t stream = 0; stream < kResidualStreams; ++stream) {
        output.insert(output.end(), input.begin(), input.end());
    }
    return output;
}

GatedResidualRead gated_residual_read(
    const std::vector<double> & hyper_input,
    std::size_t hidden_size,
    const std::vector<double> & norm_weight_delta,
    std::size_t low_rank,
    const std::vector<double> & down_matrix,
    const std::vector<double> & up_matrix,
    const std::vector<double> & injection_matrix,
    double epsilon) {
    // Direct tensor-free form of Qwen4ExpTextGatedResidual.forward in
    // Transformers revision 36deb0b53ed0863f4b4dfdea23dcaec7f3df3701.
    if (hidden_size == 0 || low_rank == 0 || epsilon <= 0.0 ||
        hidden_size > std::numeric_limits<std::size_t>::max() / kResidualStreams) {
        throw std::invalid_argument("invalid gated-residual dimensions");
    }
    const std::size_t width = kResidualStreams * hidden_size;
    if (hyper_input.size() != width || norm_weight_delta.size() != width) {
        throw std::invalid_argument("gated-residual vector width mismatch");
    }

    GatedResidualRead result;
    result.normalized.resize(width);
    for (std::size_t stream = 0; stream < kResidualStreams; ++stream) {
        double mean_square = 0.0;
        for (std::size_t dim = 0; dim < hidden_size; ++dim) {
            const double value = hyper_input[stream * hidden_size + dim];
            mean_square += value * value;
        }
        mean_square /= static_cast<double>(hidden_size);
        const double inverse_rms = 1.0 / std::sqrt(mean_square + epsilon);
        for (std::size_t dim = 0; dim < hidden_size; ++dim) {
            const std::size_t index = stream * hidden_size + dim;
            result.normalized[index] = hyper_input[index] * inverse_rms *
                                       (1.0 + norm_weight_delta[index]);
        }
    }

    std::vector<double> low =
        matrix_vector(down_matrix, low_rank, width, result.normalized);
    for (double & value : low) {
        value = silu(value / static_cast<double>(kResidualStreams));
    }
    result.mix_weights = matrix_vector(up_matrix, width, low_rank, low);
    for (double & value : result.mix_weights) {
        value = sigmoid(value);
    }
    result.mixed.assign(hidden_size, 0.0);
    for (std::size_t stream = 0; stream < kResidualStreams; ++stream) {
        for (std::size_t dim = 0; dim < hidden_size; ++dim) {
            const std::size_t index = stream * hidden_size + dim;
            result.mixed[dim] += result.mix_weights[index] * result.normalized[index] /
                                 static_cast<double>(kResidualStreams);
        }
    }

    if (!injection_matrix.empty()) {
        result.injection_weights = matrix_vector(
            injection_matrix, kResidualStreams, width, result.normalized);
        for (double & value : result.injection_weights) {
            value = 2.0 * sigmoid(value / static_cast<double>(kResidualStreams));
        }
    }
    return result;
}

std::vector<double> gated_residual_inject(
    const std::vector<double> & residual,
    const std::vector<double> & block_output,
    const std::vector<double> & injection_weights) {
    if (block_output.empty() ||
        block_output.size() > std::numeric_limits<std::size_t>::max() / kResidualStreams ||
        residual.size() != kResidualStreams * block_output.size() ||
        injection_weights.size() != kResidualStreams) {
        throw std::invalid_argument("invalid gated-residual injection dimensions");
    }
    std::vector<double> output = residual;
    for (std::size_t stream = 0; stream < kResidualStreams; ++stream) {
        for (std::size_t dim = 0; dim < block_output.size(); ++dim) {
            output[stream * block_output.size() + dim] +=
                block_output[dim] * injection_weights[stream];
        }
    }
    return output;
}

GdnScalarTrace gdn_scalar_recurrence(
    const std::vector<GdnScalarStep> & steps,
    double initial_state,
    double norm_weight,
    double l2_epsilon,
    double rms_epsilon) {
    if (l2_epsilon <= 0.0 || rms_epsilon <= 0.0) {
        throw std::invalid_argument("GDN normalization epsilon must be positive");
    }
    require_finite(initial_state, "GDN initial state must be finite");
    require_finite(norm_weight, "GDN norm weight must be finite");

    // Direct scalarization of torch_recurrent_gated_delta_rule and
    // Qwen4ExpTextRMSNormGated in Transformers revision
    // 36deb0b53ed0863f4b4dfdea23dcaec7f3df3701.
    GdnScalarTrace trace;
    trace.states.reserve(steps.size());
    trace.core_outputs.reserve(steps.size());
    trace.gated_outputs.reserve(steps.size());
    double state = initial_state;
    for (const GdnScalarStep & step : steps) {
        require_finite(step.query, "GDN query must be finite");
        require_finite(step.key, "GDN key must be finite");
        require_finite(step.value, "GDN value must be finite");
        require_finite(step.log_decay, "GDN log decay must be finite");
        require_finite(step.beta, "GDN beta must be finite");
        require_finite(step.output_gate, "GDN output gate must be finite");

        const double query = step.query /
                             std::sqrt(step.query * step.query + l2_epsilon);
        const double key = step.key /
                           std::sqrt(step.key * step.key + l2_epsilon);
        state *= std::exp(step.log_decay);
        const double memory = state * key;
        const double delta = (step.value - memory) * step.beta;
        state += key * delta;
        const double core = state * query;
        const double normalized = core / std::sqrt(core * core + rms_epsilon);
        const double output = norm_weight * normalized * sigmoid(step.output_gate);

        trace.states.push_back(state);
        trace.core_outputs.push_back(core);
        trace.gated_outputs.push_back(output);
    }
    return trace;
}

}  // namespace dflash::qwen4exp::reference
