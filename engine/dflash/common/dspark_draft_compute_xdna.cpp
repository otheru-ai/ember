#include "dspark_draft_compute_xdna.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

namespace dflash::common {
namespace {

struct ProviderState {
    void * library = nullptr;
    const ember_xdna_dspark_provider_v1 * provider = nullptr;
    void * context = nullptr;
    std::atomic<bool> healthy{true};

    ~ProviderState() {
        if (provider && provider->destroy && context) provider->destroy(context);
#if !defined(_WIN32)
        if (library) dlclose(library);
#endif
    }
};

bool checked_product(int a, int b, size_t * out) {
    if (a <= 0 || b <= 0) return false;
    const size_t ua = static_cast<size_t>(a);
    const size_t ub = static_cast<size_t>(b);
    if (ua > std::numeric_limits<size_t>::max() / ub) return false;
    *out = ua * ub;
    return true;
}

}  // namespace

struct XdnaDSparkDraftCompute::Impl {
    std::shared_ptr<ProviderState> state;
    XdnaDSparkDraftConfig config;
};

struct XdnaDSparkDraftJob::Impl {
    std::shared_ptr<ProviderState> state;
    void * job = nullptr;
    size_t output_elements = 0;
    bool completed = false;
};

XdnaDSparkDraftJob::XdnaDSparkDraftJob(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

XdnaDSparkDraftJob::~XdnaDSparkDraftJob() {
    if (!impl_ || !impl_->job) return;
    if (!impl_->completed && impl_->state->provider->cancel)
        impl_->state->provider->cancel(impl_->state->context, impl_->job);
    impl_->state->provider->destroy_job(impl_->state->context, impl_->job);
}

XdnaDSparkDraftJob::XdnaDSparkDraftJob(XdnaDSparkDraftJob &&) noexcept = default;
XdnaDSparkDraftJob & XdnaDSparkDraftJob::operator=(
    XdnaDSparkDraftJob &&) noexcept = default;

bool XdnaDSparkDraftJob::wait(XdnaDSparkDraftOutput & output,
                              std::string * error) {
    if (error) error->clear();
    if (!impl_ || !impl_->job || impl_->completed) {
        if (error) *error = "invalid or already completed XDNA DSpark job";
        return false;
    }
    output.hidden.resize(impl_->output_elements);
    output.confidence_hidden.resize(impl_->output_elements);
    ember_xdna_dspark_result_v1 result{};
    result.abi_version = EMBER_XDNA_DSPARK_PROVIDER_ABI_VERSION;
    result.struct_size = sizeof(result);
    result.hidden = output.hidden.data();
    result.hidden_capacity = output.hidden.size();
    result.confidence_hidden = output.confidence_hidden.data();
    result.confidence_capacity = output.confidence_hidden.size();
    char provider_error[512] = {};
    const int ok = impl_->state->provider->wait(
        impl_->state->context, impl_->job, &result,
        provider_error, sizeof(provider_error));
    impl_->completed = true;
    if (!ok) {
        impl_->state->healthy.store(false);
        output.hidden.clear();
        output.confidence_hidden.clear();
        if (error) {
            *error = std::string("XDNA DSpark wait failed: ") +
                (provider_error[0] ? provider_error : "no provider error");
        }
        return false;
    }
    return true;
}

void XdnaDSparkDraftJob::cancel() noexcept {
    if (!impl_ || !impl_->job || impl_->completed) return;
    if (impl_->state->provider->cancel)
        impl_->state->provider->cancel(impl_->state->context, impl_->job);
    impl_->completed = true;
}

XdnaDSparkDraftCompute::XdnaDSparkDraftCompute(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

XdnaDSparkDraftCompute::~XdnaDSparkDraftCompute() = default;
XdnaDSparkDraftCompute::XdnaDSparkDraftCompute(
    XdnaDSparkDraftCompute &&) noexcept = default;
XdnaDSparkDraftCompute & XdnaDSparkDraftCompute::operator=(
    XdnaDSparkDraftCompute &&) noexcept = default;

bool XdnaDSparkDraftCompute::healthy() const {
    return impl_ && impl_->state->healthy.load() &&
           impl_->state->provider->healthy(impl_->state->context) != 0;
}

bool XdnaDSparkDraftCompute::failure_is_fatal() const {
    return impl_ && impl_->config.required;
}

const char * XdnaDSparkDraftCompute::name() const {
    return impl_ && impl_->state->provider->name
        ? impl_->state->provider->name : "unknown";
}

std::unique_ptr<XdnaDSparkDraftJob> XdnaDSparkDraftCompute::submit(
    const XdnaDSparkDraftRequest & request, std::string * error) {
    if (error) error->clear();
    size_t output_elements = 0;
    if (!healthy() || request.committed < 0 || request.ctx_len < 0 ||
        request.ctx_len > impl_->config.n_swa || !request.noise_embed ||
        (request.ctx_len > 0 && !request.ctx_features &&
         !request.main_context && !request.context_kv) ||
        !checked_product(impl_->config.n_embd, impl_->config.block_size,
                         &output_elements)) {
        if (error) *error = "invalid XDNA DSpark request";
        return nullptr;
    }
    ember_xdna_dspark_request_v1 provider_request{};
    provider_request.abi_version = EMBER_XDNA_DSPARK_PROVIDER_ABI_VERSION;
    provider_request.struct_size = sizeof(provider_request);
    provider_request.committed = request.committed;
    provider_request.ctx_len = request.ctx_len;
    provider_request.n_embd = impl_->config.n_embd;
    provider_request.n_target_layers = impl_->config.n_target_layers;
    provider_request.block_size = impl_->config.block_size;
    provider_request.noise_embed = request.noise_embed;
    provider_request.ctx_features = request.ctx_features;
    provider_request.main_context = request.main_context;
    provider_request.context_kv = request.context_kv;
    char provider_error[512] = {};
    void * raw_job = impl_->state->provider->submit(
        impl_->state->context, &provider_request,
        provider_error, sizeof(provider_error));
    if (!raw_job) {
        impl_->state->healthy.store(false);
        if (error) {
            *error = std::string("XDNA DSpark submit failed: ") +
                (provider_error[0] ? provider_error : "no provider error");
        }
        return nullptr;
    }
    auto job = std::make_unique<XdnaDSparkDraftJob::Impl>();
    job->state = impl_->state;
    job->job = raw_job;
    job->output_elements = output_elements;
    return std::unique_ptr<XdnaDSparkDraftJob>(
        new XdnaDSparkDraftJob(std::move(job)));
}

std::unique_ptr<XdnaDSparkDraftCompute> make_xdna_dspark_draft_compute(
    const XdnaDSparkDraftConfig & config, std::string * error) {
    if (error) error->clear();
#if defined(_WIN32)
    (void)config;
    if (error) *error = "XDNA DSpark provider loading is not implemented on Windows";
    return nullptr;
#else
    if (config.plugin_path.empty() || config.draft_model_path.empty() ||
        config.n_embd <= 0 || config.n_target_layers <= 0 ||
        config.block_size <= 0 || config.n_swa <= 0 ||
        config.head_dim <= 0) {
        if (error) *error = "invalid XDNA DSpark provider configuration";
        return nullptr;
    }
    void * library = dlopen(config.plugin_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library) {
        if (error) *error = std::string("failed to load XDNA DSpark provider: ") +
                            dlerror();
        return nullptr;
    }
    dlerror();
    void * raw_symbol = dlsym(library, EMBER_XDNA_DSPARK_PROVIDER_SYMBOL);
    const char * symbol_error = dlerror();
    if (symbol_error || !raw_symbol) {
        if (error) {
            *error = std::string("XDNA DSpark provider is missing ") +
                     EMBER_XDNA_DSPARK_PROVIDER_SYMBOL + ": " +
                     (symbol_error ? symbol_error : "symbol not found");
        }
        dlclose(library);
        return nullptr;
    }
    auto entry = reinterpret_cast<ember_xdna_dspark_provider_entry_v1>(raw_symbol);
    const ember_xdna_dspark_provider_v1 * provider = entry();
    if (!provider ||
        provider->abi_version != EMBER_XDNA_DSPARK_PROVIDER_ABI_VERSION ||
        provider->struct_size < sizeof(ember_xdna_dspark_provider_v1) ||
        !provider->create || !provider->submit || !provider->wait ||
        !provider->destroy_job || !provider->healthy || !provider->destroy) {
        if (error) *error = "XDNA DSpark provider has an incompatible ABI";
        dlclose(library);
        return nullptr;
    }
    ember_xdna_dspark_config_v1 provider_config{};
    provider_config.abi_version = EMBER_XDNA_DSPARK_PROVIDER_ABI_VERSION;
    provider_config.struct_size = sizeof(provider_config);
    provider_config.draft_model_path = config.draft_model_path.c_str();
    provider_config.n_embd = config.n_embd;
    provider_config.n_target_layers = config.n_target_layers;
    provider_config.block_size = config.block_size;
    provider_config.n_swa = config.n_swa;
    provider_config.head_dim = config.head_dim;
    char provider_error[512] = {};
    void * context = provider->create(&provider_config, provider_error,
                                      sizeof(provider_error));
    if (!context) {
        if (error) {
            *error = std::string("XDNA DSpark provider initialization failed: ") +
                (provider_error[0] ? provider_error : "no provider error");
        }
        dlclose(library);
        return nullptr;
    }
    auto state = std::make_shared<ProviderState>();
    state->library = library;
    state->provider = provider;
    state->context = context;
    auto impl = std::make_unique<XdnaDSparkDraftCompute::Impl>();
    impl->state = std::move(state);
    impl->config = config;
    return std::unique_ptr<XdnaDSparkDraftCompute>(
        new XdnaDSparkDraftCompute(std::move(impl)));
#endif
}

}  // namespace dflash::common
