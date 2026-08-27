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

} // namespace dflash::common
