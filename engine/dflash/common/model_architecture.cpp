#include "model_architecture.h"

#include "ggml.h"
#include "gguf.h"

namespace dflash::common {

ModelArchitecture inspect_gguf_architecture(const char * model_path,
                                            std::string & error) {
    error.clear();
    if (!model_path || !model_path[0]) {
        error = "no model path";
        return ModelArchitecture::UNKNOWN;
    }
    gguf_init_params params = {/*.no_alloc=*/true, /*.ctx=*/nullptr};
    gguf_context * gctx = gguf_init_from_file(model_path, params);
    if (!gctx) {
        error = std::string("failed to open GGUF: ") + model_path;
        return ModelArchitecture::UNKNOWN;
    }
    const int64_t id = gguf_find_key(gctx, "general.architecture");
    if (id < 0 || gguf_get_kv_type(gctx, id) != GGUF_TYPE_STRING) {
        error = "missing or non-string general.architecture";
        gguf_free(gctx);
        return ModelArchitecture::UNKNOWN;
    }
    const char * name = gguf_get_val_str(gctx, id);
    const ModelArchitecture architecture = model_architecture_from_name(name);
    if (architecture == ModelArchitecture::UNKNOWN) {
        error = std::string("unsupported general.architecture: ") +
                (name ? name : "(null)");
    }
    gguf_free(gctx);
    return architecture;
}

}  // namespace dflash::common
