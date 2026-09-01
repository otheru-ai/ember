#include "qwen4exp_backend.h"

#include "../common/activation_dump.h"
#include "qwen4exp_activation_dump.h"
#include "qwen4exp_frontier.h"
#include "qwen4exp_vision.h"

#include "common/errors.h"
#include "common/sampler.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <unordered_set>

namespace dflash::common {
namespace {
using Clock = std::chrono::steady_clock;
double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
bool valid_slot(int slot) { return slot >= 0 && slot < ModelBackend::kMaxSlots; }
uint64_t add_saturating(uint64_t a, uint64_t b) {
    return b > UINT64_MAX - a ? UINT64_MAX : a + b;
}
uint64_t float_vector_bytes(const std::vector<float> & values) {
    return static_cast<uint64_t>(values.capacity()) * sizeof(float);
}
// One post-snapshot token can detach every GDN layer plus one partial slab for
// each QSA K/V/index cache. Reserve this before entering the layer loop so an
// otherwise recoverable prefix-cache miss cannot become a UMA OOM midway.
constexpr uint64_t kStepCowHeadroom = 130ULL << 20;
constexpr uint64_t kMtpQsaBytesPerToken = (512ULL + 512ULL + 128ULL) * 4ULL;

int32_t argmax_logits(const std::vector<float> & logits) {
    return logits.empty() ? -1 : static_cast<int32_t>(std::distance(
        logits.begin(), std::max_element(logits.begin(), logits.end())));
}

bool numerics_evidence_enabled() {
    const char * value = std::getenv("DFLASH_QWEN_NUMERICS_EVIDENCE");
    return value && std::strcmp(value, "1") == 0;
}

void log_numerics_top2(const std::vector<float> & logits,
                       const GenerateRequest & request,
                       const GenerateResult & result,
                       const char * phase, int emitted) {
    if (!numerics_evidence_enabled() || logits.size() < 2) return;
    size_t first = 0;
    size_t second = 1;
    if (logits[second] > logits[first]) std::swap(first, second);
    for (size_t index = 2; index < logits.size(); ++index) {
        if (logits[index] > logits[first]) {
            second = first;
            first = index;
        } else if (logits[index] > logits[second]) {
            second = index;
        }
    }
    std::fprintf(stderr,
                 "[qwen-numerics] event=top2 phase=%s emitted=%d mode=%s "
                 "force_exact=%s prompt_tokens=%zu top1_id=%zu "
                 "top1=%.9g top2_id=%zu top2=%.9g margin=%.9g\n",
                 phase, emitted, result.prefill_mode.c_str(),
                 request.force_exact_prefill ? "true" : "false",
                 request.prompt.size(), first,
                 static_cast<double>(logits[first]), second,
                 static_cast<double>(logits[second]),
                 static_cast<double>(logits[first] - logits[second]));
}

bool parse_mtp_depth(const char * text, int & depth) {
    if (!text || !text[0]) { depth = 3; return true; }
    char * end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (!end || end == text || *end != '\0' || value < 1 || value > 4)
        return false;
    depth = static_cast<int>(value);
    return true;
}

bool prepare_prompt_positions(
        const GenerateRequest & request,
        std::array<std::vector<int32_t>, 3> & positions,
        int64_t & rope_delta, std::string & error) {
    size_t run_index = 0;
    size_t cursor = 0;
    std::vector<Qwen4ExpMropeRun> runs;
    while (cursor < request.prompt.size()) {
        if (run_index < request.vision.size() &&
            request.vision[run_index].prompt_offset == static_cast<int>(cursor)) {
            const VisionEmbeddingRun & run = request.vision[run_index++];
            if (run.grid_t != 1 || run.grid_h <= 0 || run.grid_w <= 0 ||
                run.grid_h % 2 != 0 || run.grid_w % 2 != 0 ||
                run.embedding_width !=
                    Qwen4ExpVisionContract::output_hidden_size ||
                !run.token_ids.empty() ||
                run.embeddings.size() % Qwen4ExpVisionContract::output_hidden_size != 0) {
                error = "invalid Qwen4Exp vision run contract";
                return false;
            }
            const size_t rows = run.embeddings.size() /
                Qwen4ExpVisionContract::output_hidden_size;
            const size_t expected = static_cast<size_t>(run.grid_h /
                Qwen4ExpVisionContract::spatial_merge_size) *
                static_cast<size_t>(run.grid_w /
                Qwen4ExpVisionContract::spatial_merge_size);
            if (rows != expected || rows > request.prompt.size() - cursor) {
                error = "Qwen4Exp image rows do not match the merged grid";
                return false;
            }
            for (size_t row = 0; row < rows; ++row) {
                if (request.prompt[cursor + row] !=
                    static_cast<int32_t>(Qwen4ExpVisionContract::image_token_id)) {
                    error = "Qwen4Exp vision run must cover image_pad tokens";
                    return false;
                }
            }
            runs.push_back({rows, true,
                            {static_cast<uint32_t>(run.grid_t),
                             static_cast<uint32_t>(run.grid_h),
                             static_cast<uint32_t>(run.grid_w)}});
            cursor += rows;
            continue;
        }
        if (run_index < request.vision.size() &&
            request.vision[run_index].prompt_offset < static_cast<int>(cursor)) {
            error = "Qwen4Exp vision runs overlap or are out of order";
            return false;
        }
        const size_t next = run_index < request.vision.size()
            ? static_cast<size_t>(request.vision[run_index].prompt_offset)
            : request.prompt.size();
        if (next < cursor || next > request.prompt.size()) {
            error = "Qwen4Exp vision run lies outside the prompt";
            return false;
        }
        runs.push_back({next - cursor, false, {}});
        cursor = next;
        continue;
    }
    if (run_index != request.vision.size()) {
        error = "Qwen4Exp vision run lies outside the prompt";
        return false;
    }
    return qwen4exp_assign_mrope_positions(runs, request.prompt.size(),
                                            positions, rope_delta, error);
}
} // namespace

Qwen4ExpBackend::Qwen4ExpBackend(const Qwen4ExpBackendConfig & config)
    : config_(config) {}

Qwen4ExpBackend::~Qwen4ExpBackend() { shutdown(); }

bool Qwen4ExpBackend::init() {
    const auto init_begin = Clock::now();
    // Backend construction is retried in some embedding processes. Clear a
    // prior architecture's diagnostic so every false return below owns the
    // message observed by the factory and C ABI bridge.
    set_last_error("");
    if (!config_.model_path || !config_.model_path[0]) {
        set_last_error("Qwen4Exp model path is empty");
        return false;
    }
    const char * activation_dump = std::getenv("DFLASH_QWEN_ACT_DUMP");
    if (activation_dump && activation_dump[0]) {
        activation_dump_path_ = activation_dump;
        if (activation_dump_path_.front() != '/' ||
            activation_dump_path_.back() == '/') {
            set_last_error(
                "DFLASH_QWEN_ACT_DUMP must be an absolute output file path");
            return false;
        }
    }
    backend_ = ggml_backend_cuda_init(config_.device.gpu);
    if (!backend_) {
        set_last_error("failed to initialize Qwen4Exp HIP backend");
        return false;
    }
    snapshot_backend_ = ggml_backend_init_by_name("cpu", nullptr);
    if (!snapshot_backend_) {
        set_last_error("failed to initialize Qwen4Exp snapshot backend");
        return false;
    }
    const auto hip_end = Clock::now();
    const int max_ctx = config_.max_ctx > 0 ? config_.max_ctx : 8192;
    std::string error;
    if (!load_qwen4exp_gguf(config_.model_path, backend_, max_ctx,
                            config_.enable_yarn,
                            weights_, error)) {
        set_last_error("Qwen4Exp model load failed: " +
                       (error.empty() ? std::string("no loader diagnostic")
                                      : error));
        return false;
    }
    const auto target_end = Clock::now();
    state_budget_bytes_ = weights_.state_budget_bytes;
    const char * mtp_path = std::getenv("DFLASH_QWEN_MTP");
    const char * mtp_depth = std::getenv("DFLASH_QWEN_MTP_DEPTH");
    if ((!mtp_path || !mtp_path[0]) && mtp_depth && mtp_depth[0]) {
        set_last_error("DFLASH_QWEN_MTP_DEPTH requires DFLASH_QWEN_MTP");
        return false;
    }
    if (mtp_path && mtp_path[0]) {
        if (!parse_mtp_depth(mtp_depth, mtp_depth_)) {
            set_last_error("DFLASH_QWEN_MTP_DEPTH must be an integer from 1 to 4");
            return false;
        }
        if (!load_qwen4exp_mtp_gguf(mtp_path, backend_, mtp_weights_, error)) {
            set_last_error("Qwen4Exp MTP load failed: " +
                           (error.empty() ? std::string("no loader diagnostic")
                                          : error));
            return false;
        }
        const Qwen4ExpMemoryPlan target_plan = qwen4exp_memory_plan(
            weights_.resident_weight_bytes, max_ctx);
        const uint64_t mtp_cache = add_saturating(
            static_cast<uint64_t>(max_ctx) * kMtpQsaBytesPerToken,
            static_cast<uint64_t>(max_ctx) * 3ULL * sizeof(int32_t));
        const uint64_t mtp_total = add_saturating(
            mtp_weights_.resident_weight_bytes, mtp_cache);
        if (!target_plan.fits || mtp_total >
                target_plan.capacity_bytes - target_plan.total_bytes) {
            set_last_error("Qwen4Exp MTP companion and independent QSA cache "
                           "do not fit the 128-GiB UMA memory plan");
            return false;
        }
        state_budget_bytes_ -= mtp_weights_.resident_weight_bytes;
        if (!qwen4exp_frontier_mtp_create(
                mtp_weights_, weights_.yarn, error)) {
            set_last_error("Qwen4Exp MTP frontier initialization failed: " +
                           (error.empty() ? std::string("no frontier diagnostic")
                                          : error));
            return false;
        }
    }
    const auto mtp_end = Clock::now();
    const char * dispatch_evidence =
        std::getenv("DFLASH_ROCMI4_W4A8_DISPATCH_EVIDENCE");
    if (dispatch_evidence && std::strcmp(dispatch_evidence, "1") == 0 &&
        !qwen4exp_frontier_run_rocmi4_dispatch_controls(weights_, error)) {
        set_last_error("Qwen4Exp ROCMI4 dispatch controls failed: " + error);
        return false;
    }
    if (numerics_evidence_enabled() &&
        !qwen4exp_frontier_run_projection_numerics_control(weights_, error)) {
        set_last_error("Qwen4Exp projection numerics control failed: " + error);
        return false;
    }
    std::fprintf(stderr,
                 "[qwen4exp] text AR initialized: layers=48 ctx=%d yarn=%s; "
                 "vision=lazy mtp=%s mtp_depth=%d "
                 "mtp_verify=native-layer-major "
                 "prefill_batch=causal-q16 cache_variants=3 "
                 "ple_projection_batch=q5-q16 "
                 "hc_mixer=persistent-q1-q5-q16 "
                 "qsa_projection_batch=q5-q16 "
                 "mtp_qsa=%s "
                 "final_head=fused-hc-vocab-q1-q5-q16 "
                 "activation_dump=%s\n",
                 max_ctx, weights_.yarn.enabled ? "factor-4" : "off",
                 mtp_depth_ ? "opt-in" : "off", mtp_depth_,
                 mtp_weights_.frontier_qsa ? "persistent-q1" :
                     (mtp_depth_ ? "scalar-debug" : "off"),
                 activation_dump_path_.empty() ? "off" : "on");
    std::fprintf(stderr,
                 "[qwen-load] component=backend hip_init_ms=%.3f "
                 "target_init_ms=%.3f mtp_init_ms=%.3f total_ms=%.3f\n",
                 std::chrono::duration<double, std::milli>(hip_end - init_begin).count(),
                 std::chrono::duration<double, std::milli>(target_end - hip_end).count(),
                 std::chrono::duration<double, std::milli>(mtp_end - target_end).count(),
                 std::chrono::duration<double, std::milli>(mtp_end - init_begin).count());
    return true;
}

bool Qwen4ExpBackend::encode_vision_image(
        const uint8_t * encoded, size_t encoded_size,
        int prompt_offset,
        EncodedVisionImage & out, std::string & error) {
    (void)prompt_offset;
    // The released text and BF16 mmproj artifacts are separate. Constructing
    // this only for an image request preserves text-only residency on 128-GiB
    // UMA instead of eagerly consuming the tower's allocation.
    std::lock_guard<std::mutex> lock(vision_provider_mu_);
    if (!vision_provider_)
        vision_provider_ = std::make_unique<Qwen4ExpLazyVisionProvider>(
            config_.device.gpu);
    return vision_provider_->encode(encoded, encoded_size, out, error);
}

int32_t Qwen4ExpBackend::sample(const GenerateRequest & request,
                               const std::vector<int32_t> & history) const {
    if (request.do_sample || request.sampler.needs_logit_processing())
        return sample_logits(logits_.data(), static_cast<int>(logits_.size()),
                             request.sampler, history, rng_);
    return static_cast<int32_t>(std::distance(
        logits_.begin(), std::max_element(logits_.begin(), logits_.end())));
}

uint64_t Qwen4ExpBackend::state_storage_bytes(
        int replacement_slot, const Qwen4ExpSnapshot * replacement,
        const MtpSnapshot * mtp_replacement) const {
    std::unordered_set<const void *> seen;
    uint64_t total = state_.account_bytes(seen);
    total = add_saturating(total, float_vector_bytes(logits_));
    if (mtp_depth_) {
        total = add_saturating(total, mtp_state_.account_bytes(seen));
        total = add_saturating(total, float_vector_bytes(mtp_target_hc_));
    }
    for (int slot = 0; slot < kMaxSlots; ++slot) {
        const Qwen4ExpSnapshot * snapshot =
            slot == replacement_slot ? replacement
                                     : &snapshots_[static_cast<size_t>(slot)];
        if (!snapshot || !snapshot->used) continue;
        total = add_saturating(total, snapshot->state.account_bytes(seen));
        total = add_saturating(total, float_vector_bytes(snapshot->logits));
        if (mtp_depth_) {
            const MtpSnapshot * mtp_snapshot = slot == replacement_slot
                ? mtp_replacement
                : &mtp_snapshots_[static_cast<size_t>(slot)];
            if (mtp_snapshot && mtp_snapshot->used) {
                total = add_saturating(
                    total, mtp_snapshot->state.account_bytes(seen));
                total = add_saturating(
                    total, float_vector_bytes(mtp_snapshot->target_hc));
            }
        }
    }
    return total;
}

bool Qwen4ExpBackend::step_memory_available(std::string & error) const {
    const uint64_t used = state_storage_bytes();
    if (used > state_budget_bytes_ ||
        kStepCowHeadroom > state_budget_bytes_ - used) {
        error = "Qwen4Exp q=1 state would exceed the 128-GiB UMA budget; "
                "reduce context or free prefix snapshots";
        return false;
    }
    return true;
}

GenerateResult Qwen4ExpBackend::run(const GenerateRequest & request,
                                    const DaemonIO & io,
                                    int prompt_offset) {
    GenerateResult result;
    result.prefill_mode = "exact-q1";
    result.prefill_reason = mtp_depth_ ? "qwen4exp_text_plus_mtp_sync"
                                       : "qwen4exp_text_runtime";
    if (request.n_gen < 0 || prompt_offset < 0 ||
        prompt_offset > static_cast<int>(request.prompt.size())) {
        result.fail(GenerateErrorCode::BackendSpecific,
                    "invalid Qwen4Exp generation request");
        return result;
    }
    if (request.token_mask) {
        result.fail(GenerateErrorCode::SamplingUnsupported,
                    "Qwen4Exp constrained decoding is not implemented");
        return result;
    }
    if (prompt_offset == 0 && request.prompt.empty()) {
        result.fail(GenerateErrorCode::PrefillFailed,
                    "Qwen4Exp requires at least one prompt token");
        return result;
    }
    if (prompt_offset == 0 && !activation_dump_path_.empty() &&
        !request.vision.empty()) {
        result.fail(GenerateErrorCode::PrefillFailed,
                    "Qwen4Exp activation extraction accepts text-only prompts");
        return result;
    }
    const int remaining = static_cast<int>(request.prompt.size()) - prompt_offset;
    if (state_.cur_pos + remaining + request.n_gen > weights_.max_ctx) {
        result.fail(GenerateErrorCode::ContextOverflow,
                    "Qwen4Exp request exceeds configured context");
        return result;
    }
    std::string error;
    std::array<std::vector<int32_t>, 3> prompt_positions;
    int64_t rope_delta = 0;
    if (!prepare_prompt_positions(request, prompt_positions, rope_delta, error)) {
        result.fail(GenerateErrorCode::PrefillFailed, error);
        return result;
    }
    const auto prefill_start = Clock::now();
    size_t vision_index = 0;
    int i = prompt_offset;
    const int prompt_size = static_cast<int>(request.prompt.size());
    while (i < prompt_size) {
        if (!step_memory_available(error)) {
            result.fail(GenerateErrorCode::ContextOverflow, error);
            return result;
        }
        while (vision_index < request.vision.size() &&
               request.vision[vision_index].prompt_offset +
                   static_cast<int>(request.vision[vision_index].embeddings.size() /
                                    Qwen4ExpVisionContract::output_hidden_size) <= i)
            ++vision_index;
        const VisionEmbeddingRun * vision =
            vision_index < request.vision.size() &&
            request.vision[vision_index].prompt_offset <= i
                ? &request.vision[vision_index] : nullptr;
        const bool capture_row = prompt_offset == 0 &&
            !activation_dump_path_.empty() && i + 1 == prompt_size;
        size_t batchable_rows = 0;
        if (weights_.frontier && !request.force_exact_prefill &&
            activation_dump_path_.empty() && !vision) {
            int barrier = prompt_size;
            if (vision_index < request.vision.size()) {
                barrier = std::min(
                    barrier, request.vision[vision_index].prompt_offset);
            }
            if (barrier > i)
                batchable_rows = static_cast<size_t>(barrier - i);
        }
        const size_t chunk_rows = qwen4exp_prefill_chunk_rows(
            batchable_rows, state_.cur_pos, request.snap_pos,
            request.force_exact_prefill);
        if (chunk_rows >= 2) {
            const int target_pos_before = state_.cur_pos;
            const std::vector<float> pre_chunk_target_hc = mtp_target_hc_;
            std::array<int32_t, 3> pre_chunk_target_position{};
            if (mtp_depth_ && target_pos_before > 0 &&
                !qwen4exp_mtp_chain_position(
                    state_, 0, pre_chunk_target_position, error)) {
                result.fail(GenerateErrorCode::PrefillFailed, error);
                return result;
            }
            const auto first = request.prompt.begin() + i;
            std::vector<int32_t> tokens(
                first, first + static_cast<std::ptrdiff_t>(chunk_rows));
            std::vector<std::array<int32_t, 3>> positions;
            positions.reserve(chunk_rows);
            for (size_t row = 0; row < chunk_rows; ++row) {
                const size_t absolute = static_cast<size_t>(i) + row;
                positions.push_back({prompt_positions[0][absolute],
                                     prompt_positions[1][absolute],
                                     prompt_positions[2][absolute]});
            }
            std::vector<Qwen4ExpMtpPromptSyncRow> sync_plan;
            if (mtp_depth_ && !qwen4exp_mtp_prompt_sync_plan(
                    target_pos_before, mtp_state_.cur_pos, chunk_rows,
                    sync_plan, error)) {
                result.fail(GenerateErrorCode::PrefillFailed, error);
                return result;
            }
            std::vector<std::vector<float>> target_row_hc;
            if (!qwen4exp_step_prefill_batch_mrope(
                    weights_, state_, tokens, positions, logits_,
                    target_row_hc, error)) {
                result.fail(GenerateErrorCode::PrefillFailed, error);
                return result;
            }
            if (target_row_hc.size() != chunk_rows) {
                result.fail(GenerateErrorCode::PrefillFailed,
                            "Qwen4Exp batched prefill returned incomplete HC rows");
                return result;
            }
            if (mtp_depth_) {
                std::vector<int32_t> sync_tokens;
                std::vector<std::vector<float>> sync_target_hc;
                std::vector<std::array<int32_t, 3>> sync_positions;
                sync_tokens.reserve(sync_plan.size());
                sync_target_hc.reserve(sync_plan.size());
                sync_positions.reserve(sync_plan.size());
                for (const Qwen4ExpMtpPromptSyncRow & sync : sync_plan) {
                    sync_tokens.push_back(tokens[sync.token_row]);
                    std::array<int32_t, 3> sync_position{};
                    if (!qwen4exp_mtp_prompt_sync_position(
                            sync, pre_chunk_target_position, positions,
                            sync_position, error)) {
                        result.fail(GenerateErrorCode::PrefillFailed, error);
                        return result;
                    }
                    sync_positions.push_back(sync_position);
                    if (sync.preceding_target_hc_row < 0) {
                        sync_target_hc.push_back(pre_chunk_target_hc);
                    } else {
                        const size_t preceding = static_cast<size_t>(
                            sync.preceding_target_hc_row);
                        sync_target_hc.push_back(target_row_hc[preceding]);
                    }
                }
                if (!qwen4exp_mtp_sync_cache_batch(
                        weights_, mtp_weights_, mtp_state_, sync_tokens,
                        sync_target_hc, sync_positions, error)) {
                    result.fail(
                        GenerateErrorCode::PrefillFailed,
                        "Qwen4Exp batched MTP cache synchronization failed: " +
                            error);
                    return result;
                }
                mtp_target_hc_ = state_.hc;
            }
            i += static_cast<int>(chunk_rows);
            result.prefill_tokens += static_cast<int>(chunk_rows);
            result.prefill_mode = "exact-batched";
            result.prefill_reason = mtp_depth_
                ? "qwen4exp_text_frontier_plus_mtp_cache_sync"
                : "qwen4exp_text_frontier_layer_major";
            if (request.snap_pos == state_.cur_pos &&
                valid_slot(request.snap_slot)) {
                result.snapshot_saved = snapshot_save(request.snap_slot);
            }
            if (io.on_prefill_keepalive && !io.on_prefill_keepalive()) {
                result.fail(GenerateErrorCode::Incomplete,
                            "client disconnected during prefill");
                return result;
            }
            continue;
        }
        const std::array<int32_t, 3> position = {
            prompt_positions[0][static_cast<size_t>(i)],
            prompt_positions[1][static_cast<size_t>(i)],
            prompt_positions[2][static_cast<size_t>(i)],
        };
        const float * supplied_embedding = nullptr;
        if (vision) {
            const size_t row = static_cast<size_t>(i - vision->prompt_offset);
            supplied_embedding = vision->embeddings.data() +
                row * Qwen4ExpVisionContract::output_hidden_size;
        }
        if (mtp_depth_ && !mtp_target_hc_.empty()) {
            std::array<int32_t, 3> mtp_position{};
            if (!qwen4exp_mtp_chain_position(
                    state_, 0, mtp_position, error)) {
                result.fail(GenerateErrorCode::PrefillFailed, error);
                return result;
            }
            if (!qwen4exp_mtp_sync_cache_q1(
                    weights_, mtp_weights_, mtp_state_,
                    request.prompt[static_cast<size_t>(i)], supplied_embedding,
                    supplied_embedding ? Qwen4ExpVisionContract::output_hidden_size : 0U,
                    mtp_target_hc_.data(),
                    mtp_target_hc_.size(), mtp_position, error)) {
                result.fail(GenerateErrorCode::PrefillFailed,
                            "Qwen4Exp MTP prompt synchronization failed: " + error);
                return result;
            }
        }
        bool stepped = false;
        if (vision) {
            const size_t row = static_cast<size_t>(
                i - vision->prompt_offset);
            stepped = qwen4exp_step_q1_embedding(
                weights_, state_, request.prompt[static_cast<size_t>(i)],
                vision->embeddings.data() + row *
                    Qwen4ExpVisionContract::output_hidden_size,
                Qwen4ExpVisionContract::output_hidden_size,
                position, logits_, error);
        } else if (capture_row) {
            std::vector<float> capture;
            stepped = qwen4exp_step_q1_mrope_capture(
                weights_, state_, request.prompt[static_cast<size_t>(i)],
                position, logits_, capture, error);
            if (stepped) {
                dflash::common::ActivationDumpResult dump_result;
                stepped = dflash::common::append_activation_dump(
                    activation_dump_path_, capture,
                    kQwen4ExpActivationFloats, "Qwen4Exp",
                    dump_result, error);
                if (stepped) {
                    std::fprintf(stderr,
                                 "[qwen4exp] activation dump record=%llu "
                                 "offset=%llu path=%s\n",
                                 static_cast<unsigned long long>(
                                     dump_result.ordinal),
                                 static_cast<unsigned long long>(
                                     dump_result.byte_offset),
                                 activation_dump_path_.c_str());
                }
            }
        } else {
            stepped = qwen4exp_step_q1_mrope(
                weights_, state_, request.prompt[static_cast<size_t>(i)],
                position, logits_, error);
        }
        if (!stepped) {
            result.fail(GenerateErrorCode::PrefillFailed, error);
            return result;
        }
        if (mtp_depth_) mtp_target_hc_ = state_.hc;
        ++result.prefill_tokens;
        if (request.snap_pos == state_.cur_pos && valid_slot(request.snap_slot)) {
            result.snapshot_saved = snapshot_save(request.snap_slot);
        }
        if (io.on_prefill_keepalive && !io.on_prefill_keepalive()) {
            result.fail(GenerateErrorCode::Incomplete, "client disconnected during prefill");
            return result;
        }
        ++i;
    }
    result.prefill_s = seconds_since(prefill_start);
    log_numerics_top2(logits_, request, result, "prefill_seed", 0);
    if (logits_.empty() && request.n_gen > 0) {
        result.fail(GenerateErrorCode::DecodeSeedMissing,
                    "Qwen4Exp restore has no seed logits");
        return result;
    }

    std::vector<int32_t> history = request.prompt;
    const auto decode_start = Clock::now();
    Qwen4ExpMtpStats mtp_stats;
    int emitted = 0;
    const bool use_mtp = mtp_depth_ && !request.force_ar_decode &&
        !request.do_sample && !request.sampler.needs_logit_processing();
    const auto target_step = [&](int32_t token) {
        if (!step_memory_available(error)) return false;
        const int64_t decode_position =
            static_cast<int64_t>(state_.cur_pos) + rope_delta;
        if (decode_position < 0 ||
            decode_position > std::numeric_limits<int32_t>::max()) {
            error = "Qwen4Exp incremental M-RoPE position is out of range";
            return false;
        }
        const int32_t p = static_cast<int32_t>(decode_position);
        if (mtp_depth_) {
            std::array<int32_t, 3> mtp_position{};
            if (mtp_target_hc_.size() != 10240 ||
                !qwen4exp_mtp_chain_position(
                    state_, 0, mtp_position, error) ||
                !qwen4exp_mtp_sync_cache_q1(
                    weights_, mtp_weights_, mtp_state_, token, nullptr, 0,
                    mtp_target_hc_.data(), mtp_target_hc_.size(), mtp_position,
                    error)) return false;
        }
        if (!qwen4exp_step_q1_mrope(weights_, state_, token, {p, p, p},
                                    logits_, error)) return false;
        if (mtp_depth_) mtp_target_hc_ = state_.hc;
        return true;
    };
    while (emitted < request.n_gen && !io.cancelled) {
        if (request.capture_validation_logits)
            result.validation_logits.push_back(logits_);
        const int32_t token = sample(request, history);
        if (token == weights_.eos_id || token == weights_.eot_id) break;
        const int remaining_after_base = request.n_gen - emitted - 1;
        if (!use_mtp || remaining_after_base <= 0) {
            result.tokens.push_back(token);
            history.push_back(token);
            ++emitted;
            io.emit(token);
            if (io.cancelled) break;
            if (!target_step(token)) {
                result.fail(GenerateErrorCode::DecodeFailed, error);
                result.decode_s = seconds_since(decode_start);
                return result;
            }
            log_numerics_top2(
                logits_, request, result, "ar_frontier", emitted);
            continue;
        }
        if (mtp_target_hc_.size() != 10240) {
            result.fail(GenerateErrorCode::DecodeFailed,
                        "Qwen4Exp MTP target HC frontier is unavailable");
            result.decode_s = seconds_since(decode_start);
            return result;
        }

        const int proposal_depth = qwen4exp_mtp_effective_depth(
            mtp_depth_, remaining_after_base);
        Qwen4ExpMtpState proposal_state = mtp_state_;
        Qwen4ExpMtpState persistent_after_base;
        std::vector<float> proposal_hc = mtp_target_hc_;
        std::vector<float> proposal_logits;
        std::vector<int32_t> candidates;
        candidates.reserve(static_cast<size_t>(proposal_depth));
        int32_t proposal_input = token;
        const auto head_start = Clock::now();
        for (int depth = 0; depth < proposal_depth; ++depth) {
            std::vector<float> next_hc;
            std::array<int32_t, 3> mtp_position{};
            if (!qwen4exp_mtp_chain_position(
                    state_, static_cast<size_t>(depth), mtp_position, error))
                break;
            if (!qwen4exp_mtp_step_q1(
                    weights_, mtp_weights_, proposal_state, proposal_input,
                    nullptr, 0, proposal_hc.data(), proposal_hc.size(),
                    mtp_position, proposal_logits, next_hc, error)) break;
            if (depth == 0) persistent_after_base = proposal_state;
            const int32_t candidate = argmax_logits(proposal_logits);
            if (candidate < 0) {
                error = "Qwen4Exp MTP proposal produced empty logits";
                break;
            }
            candidates.push_back(candidate);
            proposal_input = candidate;
            proposal_hc = std::move(next_hc);
        }
        result.spec_head_s += seconds_since(head_start);
        if (static_cast<int>(candidates.size()) != proposal_depth) {
            result.fail(GenerateErrorCode::DecodeFailed,
                        "Qwen4Exp MTP proposal failed: " + error);
            result.decode_s = seconds_since(decode_start);
            return result;
        }
        result.spec_decode_ran = true;
        ++result.spec_cycles;

        std::vector<Qwen4ExpReplayRow> verify_rows;
        verify_rows.reserve(candidates.size() + 1);
        for (size_t row = 0; row <= candidates.size(); ++row) {
            const int64_t pos64 = static_cast<int64_t>(state_.cur_pos) +
                                  rope_delta + static_cast<int64_t>(row);
            if (pos64 < 0 || pos64 > std::numeric_limits<int32_t>::max()) {
                result.fail(GenerateErrorCode::DecodeFailed,
                            "Qwen4Exp MTP verify M-RoPE position is out of range");
                result.decode_s = seconds_since(decode_start);
                return result;
            }
            verify_rows.push_back({
                row == 0 ? token : candidates[row - 1],
                {static_cast<int32_t>(pos64), static_cast<int32_t>(pos64),
                 static_cast<int32_t>(pos64)}});
        }
        if (!step_memory_available(error)) {
            result.fail(GenerateErrorCode::ContextOverflow, error);
            result.decode_s = seconds_since(decode_start);
            return result;
        }
        const auto verify_start = Clock::now();
        Qwen4ExpMtpVerifyOutput verify_output;
        Qwen4ExpMtpVerifyResult verify_result;
        if (!qwen4exp_verify_bounded_batch(
                state_, logits_, verify_rows, candidates, weights_.eos_id,
                weights_.eot_id, qwen4exp_verify_target_batch,
                qwen4exp_replay_target_q1, &weights_, verify_output,
                verify_result, error)) {
            result.fail(GenerateErrorCode::DecodeFailed,
                        "Qwen4Exp MTP target verification failed: " + error);
            result.decode_s = seconds_since(decode_start);
            return result;
        }
        result.spec_verify_s += seconds_since(verify_start);

        // The first MTP call consumed the authoritative base pair. Advance
        // the persistent draft cache only over candidates the target accepted,
        // using target (not speculative) HC rows from the verified batch.
        mtp_state_ = std::move(persistent_after_base);
        for (size_t i = 0; i < verify_result.accepted_predictions; ++i) {
            const int32_t accepted_token = candidates[i];
            if (accepted_token == weights_.eos_id ||
                accepted_token == weights_.eot_id) break;
            if (!qwen4exp_mtp_sync_cache_q1(
                    weights_, mtp_weights_, mtp_state_, accepted_token,
                    nullptr, 0, verify_output.row_hc[i].data(),
                    verify_output.row_hc[i].size(),
                    verify_rows[i].mrope_position, error)) {
                result.fail(GenerateErrorCode::DecodeFailed,
                            "Qwen4Exp MTP accepted draft advance failed: " + error);
                result.decode_s = seconds_since(decode_start);
                return result;
            }
        }
        mtp_target_hc_ = state_.hc;
        if (!qwen4exp_mtp_frontier_valid(
                state_, mtp_state_, mtp_target_hc_, error)) {
            result.fail(GenerateErrorCode::DecodeFailed,
                        "Qwen4Exp MTP committed frontier is invalid: " + error);
            result.decode_s = seconds_since(decode_start);
            return result;
        }

        result.tokens.push_back(token);
        history.push_back(token);
        ++emitted;
        io.emit(token);
        for (size_t i = 0; i < verify_result.accepted_predictions &&
                           !io.cancelled; ++i) {
            const int32_t accepted_token = candidates[i];
            if (accepted_token == weights_.eos_id ||
                accepted_token == weights_.eot_id) break;
            result.tokens.push_back(accepted_token);
            history.push_back(accepted_token);
            ++emitted;
            io.emit(accepted_token);
        }
        const bool partial_replay = verify_result.replay.disposition !=
            Qwen4ExpReplayDisposition::FullAcceptance;
        if (!qwen4exp_mtp_record_round(
                mtp_stats, candidates.size(),
                verify_result.accepted_predictions, partial_replay)) {
            result.fail(GenerateErrorCode::DecodeFailed,
                        "Qwen4Exp MTP scheduler telemetry overflow");
            result.decode_s = seconds_since(decode_start);
            return result;
        }
        if (verify_result.terminal_prediction) break;
    }
    if (result.spec_decode_ran) result.accept_rate = mtp_stats.accept_rate();
    result.decode_s = seconds_since(decode_start);
    result.succeed();
    return result;
}

GenerateResult Qwen4ExpBackend::generate_impl(const GenerateRequest & request,
                                              const DaemonIO & io) {
    state_.clear();
    logits_.clear();
    mtp_state_.clear();
    mtp_target_hc_.clear();
    return run(request, io, 0);
}

GenerateResult Qwen4ExpBackend::restore_and_generate_impl(
        int slot, const GenerateRequest & request, const DaemonIO & io) {
    GenerateResult result;
    if (!valid_slot(slot) || !snapshots_[static_cast<size_t>(slot)].used) {
        result.fail(GenerateErrorCode::InvalidSnapshotSlot,
                    "invalid Qwen4Exp snapshot slot");
        return result;
    }
    const Qwen4ExpSnapshot & snapshot = snapshots_[static_cast<size_t>(slot)];
    if (snapshot.state.cur_pos < 0 ||
        snapshot.state.cur_pos > static_cast<int>(request.prompt.size())) {
        result.fail(GenerateErrorCode::ContextOverflow,
                    "Qwen4Exp snapshot frontier exceeds prompt");
        return result;
    }
    // Transactional restore copies K/V and raw index-K together; restoring an
    // attention cache with an empty indexer cache is explicitly forbidden.
    if (mtp_depth_) {
        const MtpSnapshot & mtp_snapshot =
            mtp_snapshots_[static_cast<size_t>(slot)];
        if (!mtp_snapshot.used) {
            result.fail(GenerateErrorCode::InvalidSnapshotSlot,
                        "Qwen4Exp snapshot has no matching MTP frontier");
            return result;
        }
        std::string validation_error;
        if (!qwen4exp_mtp_frontier_valid(
                snapshot.state, mtp_snapshot.state, mtp_snapshot.target_hc,
                validation_error)) {
            result.fail(GenerateErrorCode::InvalidSnapshotSlot,
                        "Qwen4Exp snapshot MTP frontier is invalid: " +
                            validation_error);
            return result;
        }
    }
    state_ = snapshot.state;
    logits_ = snapshot.logits;
    if (mtp_depth_) {
        const MtpSnapshot & mtp_snapshot =
            mtp_snapshots_[static_cast<size_t>(slot)];
        mtp_state_ = mtp_snapshot.state;
        mtp_target_hc_ = mtp_snapshot.target_hc;
    }
    return run(request, io, state_.cur_pos);
}

bool Qwen4ExpBackend::snapshot_save(int slot) {
    if (!valid_slot(slot) || state_.cur_pos <= 0 || logits_.empty()) return false;
    Qwen4ExpSnapshot snapshot;
    snapshot.used = true;
    snapshot.state = state_;
    snapshot.logits = logits_;
    MtpSnapshot mtp_snapshot;
    if (mtp_depth_) {
        std::string error;
        if (!qwen4exp_mtp_frontier_valid(
                state_, mtp_state_, mtp_target_hc_, error)) return false;
        mtp_snapshot.used = true;
        mtp_snapshot.state = mtp_state_;
        mtp_snapshot.target_hc = mtp_target_hc_;
    }
    if (state_storage_bytes(slot, &snapshot,
                            mtp_depth_ ? &mtp_snapshot : nullptr) >
        state_budget_bytes_)
        return false;
    snapshot_ref_release(slot);
    snapshots_[static_cast<size_t>(slot)] = std::move(snapshot);
    if (mtp_depth_)
        mtp_snapshots_[static_cast<size_t>(slot)] = std::move(mtp_snapshot);
    return true;
}

void Qwen4ExpBackend::snapshot_free(int slot) {
    if (valid_slot(slot)) {
        snapshot_ref_release(slot);
        snapshots_[static_cast<size_t>(slot)] = {};
        mtp_snapshots_[static_cast<size_t>(slot)] = {};
    }
}

bool Qwen4ExpBackend::snapshot_used(int slot) const {
    return valid_slot(slot) && snapshots_[static_cast<size_t>(slot)].used;
}

int Qwen4ExpBackend::snapshot_cur_pos(int slot) const {
    return snapshot_used(slot)
        ? snapshots_[static_cast<size_t>(slot)].state.cur_pos : 0;
}

ModelBackend::SnapshotRef Qwen4ExpBackend::snapshot_ref(int slot) const {
    SnapshotRef ref;
    if (!valid_slot(slot) || !snapshot_used(slot) || !snapshot_backend_)
        return ref;

    snapshot_ref_release(slot);
    const Qwen4ExpSnapshot & snapshot = snapshots_[static_cast<size_t>(slot)];
    const Qwen4ExpState & state = snapshot.state;
    constexpr size_t kHcValues = 10240;
    constexpr size_t kPleValues = 9 * 10240;
    constexpr size_t kGdnConvValues = 3 * 10240;
    constexpr size_t kGdnRecurrentValues = 48 * 128 * 128;
    const size_t rows = static_cast<size_t>(state.cur_pos);
    if (state.cur_pos <= 0 || state.cur_pos > weights_.max_ctx ||
        state.hc.size() != kHcValues || state.ple_conv.size() != kPleValues ||
        !weights_.output ||
        snapshot.logits.size() != static_cast<size_t>(weights_.output->ne[1]))
        return ref;
    for (const auto & axis : state.mrope_positions)
        if (axis.size() != rows) return ref;
    for (size_t layer = 0; layer < state.layers.size(); ++layer) {
        const Qwen4ExpLayerState & ls = state.layers[layer];
        if ((layer + 1) % 4 == 0) {
            if (ls.key.size() != rows * 512 ||
                ls.value.size() != rows * 512 ||
                ls.index_key.size() != rows * 128) return ref;
        } else if (!ls.conv || !ls.recurrent ||
                   ls.conv->size() != kGdnConvValues ||
                   ls.recurrent->size() != kGdnRecurrentValues) {
            return ref;
        }
    }
    if (mtp_depth_) {
        const MtpSnapshot & mtp = mtp_snapshots_[static_cast<size_t>(slot)];
        std::string error;
        if (!mtp.used || !qwen4exp_mtp_frontier_valid(
                state, mtp.state, mtp.target_hc, error)) return ref;
    }

    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * 128U + 4096U;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return ref;
    auto add_1d = [&](ggml_type type, int64_t count, const char * name) {
        ggml_tensor * tensor = ggml_new_tensor_1d(ctx, type, count);
        if (tensor) ggml_set_name(tensor, name);
        return tensor != nullptr;
    };
    auto add_rows = [&](ggml_type type, int64_t width, int64_t count,
                        const char * name) {
        ggml_tensor * tensor = ggml_new_tensor_2d(ctx, type, width, count);
        if (tensor) ggml_set_name(tensor, name);
        return tensor != nullptr;
    };
    bool ok = add_1d(GGML_TYPE_I32, 9, "qwen_snap_meta") &&
              add_1d(GGML_TYPE_F32, static_cast<int64_t>(kHcValues),
                     "qwen_snap_hc") &&
              add_1d(GGML_TYPE_F32, static_cast<int64_t>(kPleValues),
                     "qwen_snap_ple_conv") &&
              add_1d(GGML_TYPE_F32,
                     static_cast<int64_t>(snapshot.logits.size()),
                     "qwen_snap_logits");
    char name[64];
    for (int axis = 0; ok && axis < 3; ++axis) {
        std::snprintf(name, sizeof(name), "qwen_snap_mrope_%d", axis);
        ok = add_rows(GGML_TYPE_I32, 1, state.cur_pos, name);
    }
    for (int layer = 0; ok && layer < 48; ++layer) {
        if ((layer + 1) % 4 == 0) {
            std::snprintf(name, sizeof(name), "qwen_snap_qsa_key_%d", layer);
            ok = add_rows(GGML_TYPE_F32, 512, state.cur_pos, name);
            std::snprintf(name, sizeof(name), "qwen_snap_qsa_value_%d", layer);
            ok = ok && add_rows(GGML_TYPE_F32, 512, state.cur_pos, name);
            std::snprintf(name, sizeof(name), "qwen_snap_qsa_index_%d", layer);
            ok = ok && add_rows(GGML_TYPE_F32, 128, state.cur_pos, name);
        } else {
            std::snprintf(name, sizeof(name), "qwen_snap_gdn_conv_%d", layer);
            ok = add_1d(GGML_TYPE_F32,
                        static_cast<int64_t>(kGdnConvValues), name);
            std::snprintf(name, sizeof(name),
                          "qwen_snap_gdn_recurrent_%d", layer);
            ok = ok && add_1d(GGML_TYPE_F32,
                              static_cast<int64_t>(kGdnRecurrentValues), name);
        }
    }
    if (ok && mtp_depth_) {
        const Qwen4ExpMtpState & mtp =
            mtp_snapshots_[static_cast<size_t>(slot)].state;
        ok = add_rows(GGML_TYPE_F32, 512, mtp.cur_pos,
                      "qwen_snap_mtp_qsa_key") &&
             add_rows(GGML_TYPE_F32, 512, mtp.cur_pos,
                      "qwen_snap_mtp_qsa_value") &&
             add_rows(GGML_TYPE_F32, 128, mtp.cur_pos,
                      "qwen_snap_mtp_qsa_index") &&
             add_1d(GGML_TYPE_F32, static_cast<int64_t>(kHcValues),
                    "qwen_snap_mtp_hc") &&
             add_1d(GGML_TYPE_F32, static_cast<int64_t>(kHcValues),
                    "qwen_snap_mtp_target_hc");
        for (int axis = 0; ok && axis < 3; ++axis) {
            std::snprintf(name, sizeof(name), "qwen_snap_mtp_mrope_%d", axis);
            ok = add_rows(GGML_TYPE_I32, 1, mtp.cur_pos, name);
        }
    }
    if (!ok) {
        ggml_free(ctx);
        return ref;
    }

    ggml_backend_buffer_t buf =
        ggml_backend_alloc_ctx_tensors(ctx, snapshot_backend_);
    if (!buf) {
        ggml_free(ctx);
        return ref;
    }
    auto set_bytes = [&](const char * tensor_name, const void * data,
                         size_t bytes) {
        ggml_tensor * tensor = ggml_get_tensor(ctx, tensor_name);
        if (!tensor || ggml_nbytes(tensor) != bytes) return false;
        ggml_backend_tensor_set(tensor, data, 0, bytes);
        return true;
    };
    const int32_t meta[9] = {
        1, state.cur_pos, state.last_token, mtp_depth_,
        mtp_depth_ ? mtp_snapshots_[static_cast<size_t>(slot)].state.cur_pos : 0,
        state.ple_tokens[0], state.ple_tokens[1],
        mtp_depth_ && !mtp_snapshots_[static_cast<size_t>(slot)].state.hc.empty(),
        weights_.max_ctx};
    ok = set_bytes("qwen_snap_meta", meta, sizeof(meta)) &&
         set_bytes("qwen_snap_hc", state.hc.data(),
                   state.hc.size() * sizeof(float)) &&
         set_bytes("qwen_snap_ple_conv", state.ple_conv.data(),
                   state.ple_conv.size() * sizeof(float)) &&
         set_bytes("qwen_snap_logits", snapshot.logits.data(),
                   snapshot.logits.size() * sizeof(float));
    for (int axis = 0; ok && axis < 3; ++axis) {
        std::snprintf(name, sizeof(name), "qwen_snap_mrope_%d", axis);
        const auto & positions = state.mrope_positions[static_cast<size_t>(axis)];
        ok = set_bytes(name, positions.data(),
                       positions.size() * sizeof(int32_t));
    }
    std::vector<float> flattened;
    auto set_cow = [&](const char * tensor_name,
                       const Qwen4ExpCowBuffer & cow) {
        flattened.resize(cow.size());
        return cow.copy_to(flattened.data(), flattened.size()) &&
               set_bytes(tensor_name, flattened.data(),
                         flattened.size() * sizeof(float));
    };
    for (int layer = 0; ok && layer < 48; ++layer) {
        const Qwen4ExpLayerState & ls = state.layers[static_cast<size_t>(layer)];
        if ((layer + 1) % 4 == 0) {
            std::snprintf(name, sizeof(name), "qwen_snap_qsa_key_%d", layer);
            ok = set_cow(name, ls.key);
            std::snprintf(name, sizeof(name), "qwen_snap_qsa_value_%d", layer);
            ok = ok && set_cow(name, ls.value);
            std::snprintf(name, sizeof(name), "qwen_snap_qsa_index_%d", layer);
            ok = ok && set_cow(name, ls.index_key);
        } else {
            std::snprintf(name, sizeof(name), "qwen_snap_gdn_conv_%d", layer);
            ok = set_bytes(name, ls.conv->data(),
                           ls.conv->size() * sizeof(float));
            std::snprintf(name, sizeof(name),
                          "qwen_snap_gdn_recurrent_%d", layer);
            ok = ok && set_bytes(name, ls.recurrent->data(),
                                 ls.recurrent->size() * sizeof(float));
        }
    }
    if (ok && mtp_depth_) {
        const MtpSnapshot & mtp = mtp_snapshots_[static_cast<size_t>(slot)];
        ok = set_cow("qwen_snap_mtp_qsa_key", mtp.state.qsa.key) &&
             set_cow("qwen_snap_mtp_qsa_value", mtp.state.qsa.value) &&
             set_cow("qwen_snap_mtp_qsa_index", mtp.state.qsa.index_key);
        std::vector<float> zero_hc;
        const std::vector<float> * mtp_hc = &mtp.state.hc;
        if (mtp_hc->empty()) {
            zero_hc.assign(kHcValues, 0.0f);
            mtp_hc = &zero_hc;
        }
        ok = ok && set_bytes("qwen_snap_mtp_hc", mtp_hc->data(),
                             mtp_hc->size() * sizeof(float)) &&
             set_bytes("qwen_snap_mtp_target_hc", mtp.target_hc.data(),
                       mtp.target_hc.size() * sizeof(float));
        for (int axis = 0; ok && axis < 3; ++axis) {
            std::snprintf(name, sizeof(name), "qwen_snap_mtp_mrope_%d", axis);
            const auto & positions =
                mtp.state.mrope_positions[static_cast<size_t>(axis)];
            ok = set_bytes(name, positions.data(),
                           positions.size() * sizeof(int32_t));
        }
    }
    if (!ok) {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return ref;
    }
    SerializedSnapshot & serialized =
        serialized_snapshots_[static_cast<size_t>(slot)];
    serialized.ctx = ctx;
    serialized.buf = buf;
    ref.ctx = ctx;
    ref.buf = buf;
    ref.cur_pos = state.cur_pos;
    ref.last_tok = state.last_token;
    return ref;
}

void Qwen4ExpBackend::snapshot_ref_release(int slot) const {
    if (!valid_slot(slot)) return;
    SerializedSnapshot & serialized =
        serialized_snapshots_[static_cast<size_t>(slot)];
    if (serialized.buf) ggml_backend_buffer_free(serialized.buf);
    if (serialized.ctx) ggml_free(serialized.ctx);
    serialized = {};
}

bool Qwen4ExpBackend::snapshot_adopt(int slot, ggml_context * ctx,
                                    ggml_backend_buffer_t buf, int cur_pos,
                                    int32_t last_tok) {
    if (!valid_slot(slot) || !ctx || !buf || cur_pos <= 0 ||
        cur_pos > weights_.max_ctx) return false;
    auto tensor = [&](const char * name, ggml_type type, int64_t width,
                      int64_t rows) -> ggml_tensor * {
        ggml_tensor * value = ggml_get_tensor(ctx, name);
        if (!value || value->type != type || value->ne[0] != width ||
            value->ne[1] != rows || value->ne[2] != 1 || value->ne[3] != 1)
            return nullptr;
        return value;
    };
    auto read_f32 = [&](const char * name, size_t count,
                        std::vector<float> & out) {
        ggml_tensor * value = tensor(name, GGML_TYPE_F32,
                                    static_cast<int64_t>(count), 1);
        if (!value) return false;
        out.resize(count);
        ggml_backend_tensor_get(value, out.data(), 0,
                                count * sizeof(float));
        return true;
    };
    ggml_tensor * meta_tensor = tensor("qwen_snap_meta", GGML_TYPE_I32, 9, 1);
    if (!meta_tensor) return false;
    int32_t meta[9]{};
    ggml_backend_tensor_get(meta_tensor, meta, 0, sizeof(meta));
    if (meta[0] != 1 || meta[1] != cur_pos || meta[2] != last_tok ||
        meta[3] != mtp_depth_ || meta[8] != weights_.max_ctx ||
        (meta[7] != 0 && meta[7] != 1)) return false;

    Qwen4ExpSnapshot snapshot;
    snapshot.used = true;
    snapshot.state.cur_pos = cur_pos;
    snapshot.state.last_token = last_tok;
    snapshot.state.ple_tokens = {meta[5], meta[6]};
    if (!read_f32("qwen_snap_hc", 10240, snapshot.state.hc) ||
        !read_f32("qwen_snap_ple_conv", 9 * 10240,
                  snapshot.state.ple_conv)) return false;
    ggml_tensor * logits_tensor = ggml_get_tensor(ctx, "qwen_snap_logits");
    if (!weights_.output || !logits_tensor ||
        logits_tensor->type != GGML_TYPE_F32 ||
        logits_tensor->ne[0] != weights_.output->ne[1] ||
        logits_tensor->ne[1] != 1 ||
        logits_tensor->ne[2] != 1 || logits_tensor->ne[3] != 1) return false;
    snapshot.logits.resize(static_cast<size_t>(logits_tensor->ne[0]));
    ggml_backend_tensor_get(logits_tensor, snapshot.logits.data(), 0,
                            snapshot.logits.size() * sizeof(float));

    const size_t state_rows = static_cast<size_t>(cur_pos);
    char name[64];
    for (int axis = 0; axis < 3; ++axis) {
        std::snprintf(name, sizeof(name), "qwen_snap_mrope_%d", axis);
        ggml_tensor * value = tensor(name, GGML_TYPE_I32, 1, cur_pos);
        if (!value) return false;
        auto & positions =
            snapshot.state.mrope_positions[static_cast<size_t>(axis)];
        positions.resize(state_rows);
        ggml_backend_tensor_get(value, positions.data(), 0,
                                positions.size() * sizeof(int32_t));
    }
    auto read_cow = [&](const char * tensor_name, int64_t width, int64_t rows,
                        Qwen4ExpCowBuffer & out) {
        ggml_tensor * value = tensor(tensor_name, GGML_TYPE_F32, width, rows);
        if (!value) return false;
        std::vector<float> values(static_cast<size_t>(width * rows));
        ggml_backend_tensor_get(value, values.data(), 0,
                                values.size() * sizeof(float));
        out.append(values.data(), values.size());
        return true;
    };
    for (int layer = 0; layer < 48; ++layer) {
        Qwen4ExpLayerState & ls =
            snapshot.state.layers[static_cast<size_t>(layer)];
        if ((layer + 1) % 4 == 0) {
            std::snprintf(name, sizeof(name), "qwen_snap_qsa_key_%d", layer);
            if (!read_cow(name, 512, cur_pos, ls.key)) return false;
            std::snprintf(name, sizeof(name), "qwen_snap_qsa_value_%d", layer);
            if (!read_cow(name, 512, cur_pos, ls.value)) return false;
            std::snprintf(name, sizeof(name), "qwen_snap_qsa_index_%d", layer);
            if (!read_cow(name, 128, cur_pos, ls.index_key)) return false;
        } else {
            std::vector<float> values;
            std::snprintf(name, sizeof(name), "qwen_snap_gdn_conv_%d", layer);
            if (!read_f32(name, 3 * 10240, values)) return false;
            ls.conv = std::make_shared<std::vector<float>>(std::move(values));
            std::snprintf(name, sizeof(name),
                          "qwen_snap_gdn_recurrent_%d", layer);
            if (!read_f32(name, 48 * 128 * 128, values)) return false;
            ls.recurrent =
                std::make_shared<std::vector<float>>(std::move(values));
        }
    }

    MtpSnapshot mtp_snapshot;
    if (mtp_depth_) {
        if (meta[4] != cur_pos - 1) return false;
        mtp_snapshot.used = true;
        mtp_snapshot.state.cur_pos = meta[4];
        if (!read_cow("qwen_snap_mtp_qsa_key", 512, meta[4],
                      mtp_snapshot.state.qsa.key) ||
            !read_cow("qwen_snap_mtp_qsa_value", 512, meta[4],
                      mtp_snapshot.state.qsa.value) ||
            !read_cow("qwen_snap_mtp_qsa_index", 128, meta[4],
                      mtp_snapshot.state.qsa.index_key)) return false;
        std::vector<float> mtp_hc;
        if (!read_f32("qwen_snap_mtp_hc", 10240, mtp_hc) ||
            !read_f32("qwen_snap_mtp_target_hc", 10240,
                      mtp_snapshot.target_hc)) return false;
        if (meta[7]) mtp_snapshot.state.hc = std::move(mtp_hc);
        const size_t mtp_rows = static_cast<size_t>(meta[4]);
        for (int axis = 0; axis < 3; ++axis) {
            std::snprintf(name, sizeof(name), "qwen_snap_mtp_mrope_%d", axis);
            ggml_tensor * value = tensor(name, GGML_TYPE_I32, 1, meta[4]);
            if (!value) return false;
            auto & positions =
                mtp_snapshot.state.mrope_positions[static_cast<size_t>(axis)];
            positions.resize(mtp_rows);
            ggml_backend_tensor_get(value, positions.data(), 0,
                                    positions.size() * sizeof(int32_t));
        }
        std::string error;
        if (!qwen4exp_mtp_frontier_valid(
                snapshot.state, mtp_snapshot.state, mtp_snapshot.target_hc,
                error)) return false;
    }
    if (state_storage_bytes(slot, &snapshot,
                            mtp_depth_ ? &mtp_snapshot : nullptr) >
        state_budget_bytes_) return false;

    snapshot_free(slot);
    snapshots_[static_cast<size_t>(slot)] = std::move(snapshot);
    if (mtp_depth_)
        mtp_snapshots_[static_cast<size_t>(slot)] = std::move(mtp_snapshot);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

void Qwen4ExpBackend::shutdown() {
    {
        std::lock_guard<std::mutex> lock(vision_provider_mu_);
        vision_provider_.reset();
    }
    for (int slot = 0; slot < kMaxSlots; ++slot) snapshot_ref_release(slot);
    for (Qwen4ExpSnapshot & snapshot : snapshots_) snapshot = {};
    for (MtpSnapshot & snapshot : mtp_snapshots_) snapshot = {};
    state_.clear(); logits_.clear(); mtp_state_.clear(); mtp_target_hc_.clear();
    free_qwen4exp_mtp_weights(mtp_weights_);
    free_qwen4exp_weights(weights_);
    if (snapshot_backend_) {
        ggml_backend_free(snapshot_backend_);
        snapshot_backend_ = nullptr;
    }
    if (backend_) { ggml_backend_free(backend_); backend_ = nullptr; }
}

} // namespace dflash::common
