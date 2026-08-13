#include "moe_expert_compute_xdna.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

namespace dflash::common {
namespace {

bool trace_enabled() {
    const char * value = std::getenv("DFLASH_MOE_XDNA_TRACE");
    return value && value[0] && std::strcmp(value, "0") != 0;
}

class XdnaMoeExpertCompute final : public MoeExpertCompute {
public:
    XdnaMoeExpertCompute(void * library,
                         const ember_xdna_moe_provider_v1 * provider,
                         void * context,
                         int min_tokens,
                         bool required)
        : library_(library), provider_(provider), context_(context),
          min_tokens_(std::max(1, min_tokens)), required_(required) {}

    ~XdnaMoeExpertCompute() override {
        if (trace_enabled()) {
            std::fprintf(stderr,
                         "[xdna-moe] provider=%s calls=%llu tokens=%llu wall_us=%llu\n",
                         provider_ && provider_->name ? provider_->name : "unknown",
                         (unsigned long long)calls_,
                         (unsigned long long)tokens_,
                         (unsigned long long)wall_us_);
        }
        if (provider_ && provider_->destroy && context_) {
            provider_->destroy(context_);
        }
#if !defined(_WIN32)
        if (library_) dlclose(library_);
#endif
    }

    bool healthy() const override {
        return healthy_ && provider_ && provider_->healthy &&
               provider_->healthy(context_) != 0;
    }

    bool accepts(int n_tokens, int n_selected) const override {
        return healthy() && n_tokens >= min_tokens_ && n_selected > 0;
    }

    bool failure_is_fatal() const override { return required_; }

    bool compute(const MoeExpertLayer & layer,
                 const float * input,
                 const int32_t * ids,
                 const float * weights,
                 int n_selected,
                 int n_embd,
                 int n_ff,
                 float * output) override {
        return compute_batch(layer, input, ids, weights, 1, n_selected,
                             n_embd, n_ff, output);
    }

    bool compute_batch(const MoeExpertLayer & layer,
                       const float * input,
                       const int32_t * ids,
                       const float * weights,
                       int n_tokens,
                       int n_selected,
                       int n_embd,
                       int n_ff,
                       float * output) override {
        if (!accepts(n_tokens, n_selected) || !input || !ids || !weights ||
            !output || layer.layer_idx < 0 || n_embd <= 0 || n_ff <= 0 ||
            (size_t)n_tokens > std::numeric_limits<size_t>::max() /
                                   (size_t)n_selected) {
            return false;
        }

        const size_t slots = (size_t)n_tokens * (size_t)n_selected;
        std::vector<int32_t> global_ids(slots);
        for (size_t i = 0; i < slots; ++i) {
            const int32_t local = ids[i];
            if (local < 0 || (size_t)local >= layer.cold_global_by_local.size()) {
                return false;
            }
            global_ids[i] = layer.cold_global_by_local[(size_t)local];
        }

        ember_xdna_moe_batch_v1 batch{};
        batch.abi_version = EMBER_XDNA_MOE_PROVIDER_ABI_VERSION;
        batch.struct_size = sizeof(batch);
        batch.layer_idx = layer.layer_idx;
        batch.n_tokens = n_tokens;
        batch.n_selected = n_selected;
        batch.n_embd = n_embd;
        batch.n_ff_exp = n_ff;
        batch.input = input;
        batch.expert_ids = global_ids.data();
        batch.router_weights = weights;
        batch.output = output;

        char error[512] = {};
        const auto start = std::chrono::steady_clock::now();
        const int ok = provider_->compute(context_, &batch, error, sizeof(error));
        const auto end = std::chrono::steady_clock::now();
        wall_us_ += (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                        end - start).count();
        calls_++;
        tokens_ += (uint64_t)n_tokens;
        if (!ok) {
            healthy_ = false;
            std::fprintf(stderr, "[xdna-moe] provider compute failed: %s\n",
                         error[0] ? error : "no provider error");
            return false;
        }
        return true;
    }

private:
    void * library_ = nullptr;
    const ember_xdna_moe_provider_v1 * provider_ = nullptr;
    void * context_ = nullptr;
    int min_tokens_ = 1;
    bool required_ = false;
    bool healthy_ = true;
    uint64_t calls_ = 0;
    uint64_t tokens_ = 0;
    uint64_t wall_us_ = 0;
};

}  // namespace

std::unique_ptr<MoeExpertCompute> make_xdna_moe_expert_compute(
    const XdnaMoeExpertComputeConfig & config,
    std::string * error) {
#if defined(_WIN32)
    if (error) *error = "XDNA MoE provider loading is not implemented on Windows";
    return nullptr;
#else
    if (config.plugin_path.empty() || config.model_path.empty() ||
        config.n_layer <= 0 || config.n_expert <= 0 ||
        config.n_expert_used <= 0 || config.n_embd <= 0 ||
        config.n_ff_exp <= 0) {
        if (error) *error = "invalid XDNA MoE provider configuration";
        return nullptr;
    }

    void * library = dlopen(config.plugin_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library) {
        if (error) *error = std::string("failed to load XDNA MoE provider: ") + dlerror();
        return nullptr;
    }
    dlerror();
    void * raw_symbol = dlsym(library, EMBER_XDNA_MOE_PROVIDER_SYMBOL);
    const char * symbol_error = dlerror();
    if (symbol_error || !raw_symbol) {
        if (error) {
            *error = std::string("XDNA MoE provider is missing ") +
                     EMBER_XDNA_MOE_PROVIDER_SYMBOL + ": " +
                     (symbol_error ? symbol_error : "symbol not found");
        }
        dlclose(library);
        return nullptr;
    }

    auto entry = reinterpret_cast<ember_xdna_moe_provider_entry_v1>(raw_symbol);
    const ember_xdna_moe_provider_v1 * provider = entry();
    if (!provider || provider->abi_version != EMBER_XDNA_MOE_PROVIDER_ABI_VERSION ||
        provider->struct_size < sizeof(ember_xdna_moe_provider_v1) ||
        !provider->create || !provider->compute || !provider->healthy ||
        !provider->destroy) {
        if (error) *error = "XDNA MoE provider has an incompatible ABI";
        dlclose(library);
        return nullptr;
    }

    ember_xdna_moe_config_v1 provider_config{};
    provider_config.abi_version = EMBER_XDNA_MOE_PROVIDER_ABI_VERSION;
    provider_config.struct_size = sizeof(provider_config);
    provider_config.model_path = config.model_path.c_str();
    provider_config.n_layer = config.n_layer;
    provider_config.n_expert = config.n_expert;
    provider_config.n_expert_used = config.n_expert_used;
    provider_config.n_embd = config.n_embd;
    provider_config.n_ff_exp = config.n_ff_exp;
    provider_config.swiglu_clamp = config.swiglu_clamp;
    char provider_error[512] = {};
    void * context = provider->create(&provider_config, provider_error,
                                      sizeof(provider_error));
    if (!context) {
        if (error) {
            *error = std::string("XDNA MoE provider initialization failed: ") +
                     (provider_error[0] ? provider_error : "no provider error");
        }
        dlclose(library);
        return nullptr;
    }

    return std::make_unique<XdnaMoeExpertCompute>(
        library, provider, context, config.min_tokens, config.required);
#endif
}

}  // namespace dflash::common
