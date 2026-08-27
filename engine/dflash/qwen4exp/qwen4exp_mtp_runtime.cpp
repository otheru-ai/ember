#include "qwen4exp_mtp.h"

namespace dflash::common {

bool qwen4exp_replay_target_q1(
        void * opaque, Qwen4ExpState & state, const Qwen4ExpReplayRow & row,
        std::vector<float> & logits, std::string & error) {
    if (!opaque) {
        error = "Qwen4Exp MTP replay has no target weights";
        return false;
    }
    const auto * weights = static_cast<const Qwen4ExpWeights *>(opaque);
    return qwen4exp_step_q1_mrope(*weights, state, row.token,
                                   row.mrope_position, logits, error);
}

bool qwen4exp_verify_target_batch(
        void * opaque, Qwen4ExpState & state,
        const std::vector<Qwen4ExpReplayRow> & rows,
        Qwen4ExpMtpVerifyOutput & output, std::string & error) {
    auto * weights = static_cast<Qwen4ExpWeights *>(opaque);
    if (!weights || rows.empty()) {
        error = "invalid Qwen4Exp target batch adapter";
        return false;
    }
    output = {};
    std::vector<int32_t> tokens;
    std::vector<std::array<int32_t, 3>> positions;
    tokens.reserve(rows.size());
    positions.reserve(rows.size());
    for (const Qwen4ExpReplayRow & row : rows) {
        tokens.push_back(row.token);
        positions.push_back(row.mrope_position);
    }
    return qwen4exp_step_batch_mrope(*weights, state, tokens, positions,
                                     output.row_logits, output.row_hc, error);
}

} // namespace dflash::common
