// XDNA2 asynchronous DSpark-forward provider ABI.
//
// The existing MoE provider is deliberately layer-local.  That is the wrong
// synchronization shape for heterogeneous speculative decode: 43 target-layer
// callbacks lose to the fused GPU verifier even when transport is cheap.  This
// ABI moves the complete three-layer DSpark support-model forward behind one
// submission.  A resident-session scheduler can therefore submit session B's
// proposal, verify session A on the GPU, then collect B without a per-layer
// CPU/GPU/NPU barrier.
//
// The provider returns normalized block hidden states, not token ids.  Ember
// retains the tied target LM/Markov/confidence heads, so the first hardware
// implementation does not have to duplicate the 129280-wide target output
// matrix.  Inputs are valid only during submit(); a successful provider must
// have copied or otherwise retained everything needed before submit returns.
// Optional tail fields preserve the v1 prefixes: providers must use
// struct_size before reading them. main_context is the GPU-preprojected
// replacement for ctx_features. context_kv goes one step further and contains
// all per-draft-layer normalized context KV rows; a new provider should prefer
// it when present, while an older v1 provider can continue using main_context.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#define EMBER_XDNA_DSPARK_PROVIDER_ABI_VERSION 1u
#define EMBER_XDNA_DSPARK_PROVIDER_SYMBOL \
    "ember_xdna_dspark_get_provider_v1"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ember_xdna_dspark_config_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    const char * draft_model_path;
    int32_t n_embd;
    int32_t n_target_layers;
    int32_t block_size;
    int32_t n_swa;
    // Optional tail: width of each per-layer context KV row.
    int32_t head_dim;
} ember_xdna_dspark_config_v1;

typedef struct ember_xdna_dspark_request_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    int32_t committed;
    int32_t ctx_len;
    int32_t n_embd;
    int32_t n_target_layers;
    int32_t block_size;
    const float * noise_embed;   // [block_size, n_embd]
    const float * ctx_features;  // optional [ctx_len, n_target_layers*n_embd]
    // Optional post-main_norm GPU pre-stage, [ctx_len, n_embd]. A request with
    // context must provide ctx_features, main_context, or context_kv. This tail
    // extension may be read only when struct_size covers the field.
    const float * main_context;
    // Optional normalized pre-RoPE context KV,
    // [n_target_layers, ctx_len, head_dim]. This tail may be read only when
    // struct_size covers the field. main_context remains populated so older v1
    // providers retain a complete fallback input.
    const float * context_kv;
} ember_xdna_dspark_request_v1;

typedef struct ember_xdna_dspark_result_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    float * hidden;             // [block_size, n_embd], output-RMSNorm state
    size_t hidden_capacity;     // float elements
    float * confidence_hidden;  // optional pre-output-RMSNorm state
    size_t confidence_capacity; // float elements
} ember_xdna_dspark_result_v1;

typedef struct ember_xdna_dspark_provider_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    const char * name;
    void * (*create)(const ember_xdna_dspark_config_v1 * config,
                     char * error, size_t error_capacity);
    // A successful submit owns all request data needed by the returned job.
    void * (*submit)(void * context,
                     const ember_xdna_dspark_request_v1 * request,
                     char * error, size_t error_capacity);
    int (*wait)(void * context, void * job,
                ember_xdna_dspark_result_v1 * result,
                char * error, size_t error_capacity);
    void (*cancel)(void * context, void * job);
    void (*destroy_job)(void * context, void * job);
    int (*healthy)(void * context);
    void (*destroy)(void * context);
} ember_xdna_dspark_provider_v1;

typedef const ember_xdna_dspark_provider_v1 *
    (*ember_xdna_dspark_provider_entry_v1)(void);

#ifdef __cplusplus
}
#endif

namespace dflash::common {

struct XdnaDSparkDraftConfig {
    std::string plugin_path;
    std::string draft_model_path;
    int n_embd = 0;
    int n_target_layers = 0;
    int block_size = 0;
    int n_swa = 0;
    int head_dim = 0;
    bool required = false;
};

struct XdnaDSparkDraftRequest {
    int committed = 0;
    int ctx_len = 0;
    const float * noise_embed = nullptr;
    const float * ctx_features = nullptr;
    const float * main_context = nullptr;
    const float * context_kv = nullptr;
};

struct XdnaDSparkDraftOutput {
    std::vector<float> hidden;
    std::vector<float> confidence_hidden;
};

class XdnaDSparkDraftJob {
public:
    ~XdnaDSparkDraftJob();
    XdnaDSparkDraftJob(XdnaDSparkDraftJob &&) noexcept;
    XdnaDSparkDraftJob & operator=(XdnaDSparkDraftJob &&) noexcept;
    XdnaDSparkDraftJob(const XdnaDSparkDraftJob &) = delete;
    XdnaDSparkDraftJob & operator=(const XdnaDSparkDraftJob &) = delete;

    bool wait(XdnaDSparkDraftOutput & output, std::string * error = nullptr);
    void cancel() noexcept;

private:
    struct Impl;
    explicit XdnaDSparkDraftJob(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    friend class XdnaDSparkDraftCompute;
};

class XdnaDSparkDraftCompute {
public:
    ~XdnaDSparkDraftCompute();
    XdnaDSparkDraftCompute(XdnaDSparkDraftCompute &&) noexcept;
    XdnaDSparkDraftCompute & operator=(XdnaDSparkDraftCompute &&) noexcept;
    XdnaDSparkDraftCompute(const XdnaDSparkDraftCompute &) = delete;
    XdnaDSparkDraftCompute & operator=(const XdnaDSparkDraftCompute &) = delete;

    bool healthy() const;
    bool failure_is_fatal() const;
    const char * name() const;
    std::unique_ptr<XdnaDSparkDraftJob> submit(
        const XdnaDSparkDraftRequest & request,
        std::string * error = nullptr);

private:
    struct Impl;
    explicit XdnaDSparkDraftCompute(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    friend std::unique_ptr<XdnaDSparkDraftCompute>
        make_xdna_dspark_draft_compute(const XdnaDSparkDraftConfig &,
                                       std::string *);
};

std::unique_ptr<XdnaDSparkDraftCompute> make_xdna_dspark_draft_compute(
    const XdnaDSparkDraftConfig & config, std::string * error = nullptr);

}  // namespace dflash::common
