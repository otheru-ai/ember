// Single-session autoregressive text backend for Qwen3.8-Flash-Next.
// The MTP companion is opt-in and fail-closed; snapshots cover the
// complete text state, including raw index-K, QSA K/V, GDN recurrence/conv,
// PLE history/conv, and the HC/frontier state.

#pragma once

#include "common/model_backend.h"
#include "placement/placement_config.h"
#include "qwen4exp_internal.h"
#include "qwen4exp_mtp.h"
#include "qwen4exp_vision_provider.h"

#include <array>
#include <mutex>
#include <random>
#include <string>

namespace dflash::common {

struct Qwen4ExpBackendConfig {
    const char * model_path = nullptr;
    DevicePlacement device;
    int max_ctx = 0;
    bool enable_yarn = false;
};

class Qwen4ExpBackend final : public ModelBackend {
public:
    explicit Qwen4ExpBackend(const Qwen4ExpBackendConfig & config);
    ~Qwen4ExpBackend() override;
    bool init();
    bool validation_compare_production_prefill() const override {
        return true;
    }
    bool encode_vision_image(const uint8_t * encoded, size_t encoded_size,
                             EncodedVisionImage & out,
                             std::string & error) override;

    GenerateResult generate_impl(const GenerateRequest & request,
                                 const DaemonIO & io) override;
    GenerateResult restore_and_generate_impl(int slot,
                                             const GenerateRequest & request,
                                             const DaemonIO & io) override;
    bool snapshot_save(int slot) override;
    void snapshot_free(int slot) override;
    bool snapshot_used(int slot) const override;
    int snapshot_cur_pos(int slot) const override;
    SnapshotRef snapshot_ref(int slot) const override;
    void snapshot_ref_release(int slot) const override;
    bool snapshot_adopt(int slot, ggml_context * ctx,
                        ggml_backend_buffer_t buf, int cur_pos,
                        int32_t last_tok) override;
    void shutdown() override;

private:
    struct MtpSnapshot {
        bool used = false;
        Qwen4ExpMtpState state;
        std::vector<float> target_hc;
    };

    struct SerializedSnapshot {
        ggml_context * ctx = nullptr;
        ggml_backend_buffer_t buf = nullptr;
    };

    GenerateResult run(const GenerateRequest & request, const DaemonIO & io,
                       int prompt_offset);
    int32_t sample(const GenerateRequest & request,
                   const std::vector<int32_t> & history) const;
    uint64_t state_storage_bytes(int replacement_slot = -1,
                                 const Qwen4ExpSnapshot * replacement = nullptr,
                                 const MtpSnapshot * mtp_replacement = nullptr) const;
    bool step_memory_available(std::string & error) const;

    Qwen4ExpBackendConfig config_;
    ggml_backend_t backend_ = nullptr;
    ggml_backend_t snapshot_backend_ = nullptr;
    Qwen4ExpWeights weights_;
    Qwen4ExpState state_;
    std::vector<float> logits_;
    std::array<Qwen4ExpSnapshot, kMaxSlots> snapshots_;
    Qwen4ExpMtpWeights mtp_weights_;
    Qwen4ExpMtpState mtp_state_;
    std::vector<float> mtp_target_hc_;
    std::array<MtpSnapshot, kMaxSlots> mtp_snapshots_;
    mutable std::array<SerializedSnapshot, kMaxSlots> serialized_snapshots_;
    int mtp_depth_ = 0;
    uint64_t state_budget_bytes_ = 0;
    // Image preprocessing happens before requests enter the resident batch
    // coordinator. Batch workers therefore share this lazy provider directly;
    // serialize construction and use of its single mtmd context here.
    std::mutex vision_provider_mu_;
    std::unique_ptr<Qwen4ExpLazyVisionProvider> vision_provider_;
    std::string activation_dump_path_;
    mutable std::mt19937_64 rng_{0};
};

} // namespace dflash::common
