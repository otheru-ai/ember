#include "qwen4exp_vision_provider.h"

#include "common/model_backend.h"
#include "qwen4exp_vision.h"

#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <mutex>
#include <utility>

namespace dflash::common {
namespace {

constexpr const char * kProviderEnv = "DFLASH_QWEN_VISION_PROVIDER";
constexpr const char * kMmprojEnv = "DFLASH_QWEN_VISION_MMPROJ";
constexpr const char * kTextModelEnv = "DFLASH_QWEN_VISION_TEXT_MODEL";
constexpr const char * kEntry = "qwen4exp_vision_provider_get_v1";

bool checked_mul(size_t a, size_t b, size_t & out) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
    out = a * b;
    return true;
}

} // namespace

struct Qwen4ExpLazyVisionProvider::Impl {
    explicit Impl(int gpu_index) : gpu(gpu_index) {}
    int gpu = 0;
    void * library = nullptr;
    const qwen4exp_vision_provider_v1 * api = nullptr;
    void * context = nullptr;
    bool attempted = false;
    // The provider owns one mutable mtmd context. The server may call this
    // object from several resident-session workers before they enter the engine
    // coordinator, so first load and every encode/free pair must be serialized.
    std::mutex encode_mu;

    ~Impl() {
        if (api && api->destroy && context) api->destroy(context);
        if (library) dlclose(library);
    }

    bool load(std::string & error) {
        if (context) return true;
        if (attempted) {
            error = "Qwen4Exp vision provider initialization previously failed";
            return false;
        }
        attempted = true;
        const char * provider_path = std::getenv(kProviderEnv);
        const char * mmproj_path = std::getenv(kMmprojEnv);
        const char * text_model_path = std::getenv(kTextModelEnv);
        if (!provider_path || !provider_path[0] || !mmproj_path || !mmproj_path[0]
                || !text_model_path || !text_model_path[0]) {
            error = std::string("Qwen4Exp vision requires both ") + kProviderEnv +
                    ", " + kMmprojEnv + " and " + kTextModelEnv +
                    " (separate BF16 --mmproj and zero-tensor vocab artifacts)";
            return false;
        }
        library = dlopen(provider_path, RTLD_NOW | RTLD_LOCAL);
        if (!library) {
            const char * detail = dlerror();
            error = std::string("failed to load Qwen4Exp vision provider: ") +
                    (detail ? detail : "unknown dlopen error");
            return false;
        }
        dlerror();
        void * symbol = dlsym(library, kEntry);
        const char * symbol_error = dlerror();
        if (symbol_error || !symbol) {
            error = std::string("Qwen4Exp vision provider lacks ") + kEntry;
            return false;
        }
        qwen4exp_vision_provider_entry_v1 entry = nullptr;
        static_assert(sizeof(entry) == sizeof(symbol));
        std::memcpy(&entry, &symbol, sizeof(entry));
        api = entry();
        if (!api || api->abi_version != kQwen4ExpVisionProviderAbi ||
            !api->create || !api->destroy || !api->encode || !api->free_output) {
            error = "Qwen4Exp vision provider ABI mismatch";
            return false;
        }
        char detail[512]{};
        context = api->create(mmproj_path, text_model_path, gpu,
                              detail, sizeof(detail));
        if (!context) {
            error = detail[0] ? detail : "Qwen4Exp vision provider could not load mmproj";
            return false;
        }
        return true;
    }
};

Qwen4ExpLazyVisionProvider::Qwen4ExpLazyVisionProvider(int gpu)
    : impl_(std::make_unique<Impl>(gpu)) {}

Qwen4ExpLazyVisionProvider::~Qwen4ExpLazyVisionProvider() = default;

bool Qwen4ExpLazyVisionProvider::encode(
        const uint8_t * encoded, size_t encoded_size,
        EncodedVisionImage & out, std::string & error) {
    std::lock_guard<std::mutex> lock(impl_->encode_mu);
    out = {};
    error.clear();
    if (!encoded || encoded_size == 0) {
        error = "Qwen4Exp encoded image is empty";
        return false;
    }
    if (!impl_->load(error)) return false;
    qwen4exp_vision_provider_output_v1 raw{};
    char detail[512]{};
    if (!impl_->api->encode(impl_->context, encoded, encoded_size, &raw,
                            detail, sizeof(detail))) {
        error = detail[0] ? detail : "Qwen4Exp vision provider encode failed";
        return false;
    }
    struct OutputGuard {
        Impl * impl;
        qwen4exp_vision_provider_output_v1 * output;
        ~OutputGuard() { impl->api->free_output(impl->context, output); }
    } guard{impl_.get(), &raw};

    Qwen4ExpVisionGrid grid{raw.grid_t, raw.grid_h, raw.grid_w};
    const size_t expected_rows = qwen4exp_vision_merged_tokens(grid);
    size_t value_count = 0;
    if (grid.t != 1 || expected_rows == 0 || raw.row_count != expected_rows ||
        raw.embedding_width != kQwen4ExpVisionEmbeddingWidth ||
        !checked_mul(raw.row_count, raw.embedding_width, value_count) ||
        (value_count != 0 && !raw.rows)) {
        error = "Qwen4Exp vision provider returned an invalid image embedding contract";
        return false;
    }
    out.grid_t = static_cast<int>(grid.t);
    out.grid_h = static_cast<int>(grid.h);
    out.grid_w = static_cast<int>(grid.w);
    out.embedding_width = static_cast<int>(raw.embedding_width);
    out.embeddings.assign(raw.rows, raw.rows + value_count);
    return true;
}

} // namespace dflash::common
