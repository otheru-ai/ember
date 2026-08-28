// llama.cpp mtmd adapter for Ember's lazy Qwen3.8-Flash-Next vision ABI.
//
// Build this as a shared object against the exact llama.cpp revision used to
// convert the BF16 `--mmproj` artifact. Only a vocab-only view of the text GGUF
// is loaded; tensor weights are not duplicated. The stock public mtmd API then
// owns image decoding, Qwen3-VL smart-resize/patchify, and the ViT graph.
//
// This file intentionally is not part of Ember's build: it must be compiled
// with llama.cpp's public llama + mtmd headers/libraries so one ggml lineage
// owns the entire projector context.

#include "qwen4exp_vision_provider.h"

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

using dflash::common::kQwen4ExpVisionProviderAbi;
using dflash::common::kQwen4ExpVisionEmbeddingWidth;
using dflash::common::qwen4exp_vision_provider_output_v1;
using dflash::common::qwen4exp_vision_provider_v1;

namespace {

struct Context {
    llama_model * vocab_model = nullptr;
    mtmd_context * mtmd = nullptr;
    uint32_t embedding_width = 0;
    uint64_t capture_index = 0;
};

void set_error(char * out, size_t cap, const char * message) {
    if (out && cap) std::snprintf(out, cap, "%s", message);
}

bool write_all(int fd, const void * data, size_t size) {
    const auto * cursor = static_cast<const uint8_t *>(data);
    while (size != 0) {
        const ssize_t written = ::write(fd, cursor, size);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) return false;
        cursor += static_cast<size_t>(written);
        size -= static_cast<size_t>(written);
    }
    return true;
}

void little_u64(uint64_t value, std::array<uint8_t, 8> & out) {
    for (unsigned i = 0; i < 8; ++i)
        out[i] = static_cast<uint8_t>(value >> (i * 8));
}

// Hardware certification sets this prefix inside a new, private evidence
// directory. When enabled, capture failure is a request failure: silently
// accepting a vision result without the bytes compared to the pinned mtmd
// reference would turn a missing differential into release evidence.
bool capture_output(Context * ctx,
                    const qwen4exp_vision_provider_output_v1 & out,
                    char * error, size_t error_cap) {
    const char * prefix = std::getenv("DFLASH_QWEN_VISION_CAPTURE_PREFIX");
    if (!prefix || !prefix[0]) return true;
    if (prefix[0] != '/' || std::strchr(prefix, '\n') ||
        std::strchr(prefix, '\r')) {
        set_error(error, error_cap, "vision capture prefix must be an absolute safe path");
        return false;
    }
    char suffix[48]{};
    std::snprintf(suffix, sizeof(suffix), ".%06llu.f32",
                  static_cast<unsigned long long>(ctx->capture_index));
    const std::string path = std::string(prefix) + suffix;
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL |
                          O_CLOEXEC | O_NOFOLLOW, S_IRUSR | S_IWUSR);
    if (fd < 0) {
        set_error(error, error_cap, "cannot create exclusive vision capture");
        return false;
    }
    const std::array<uint8_t, 8> magic{{'E','V','I','S','F','3','2','\0'}};
    const uint64_t fields[] = {1, out.grid_t, out.grid_h, out.grid_w,
                               out.embedding_width, out.row_count};
    bool ok = write_all(fd, magic.data(), magic.size());
    std::array<uint8_t, 8> encoded{};
    for (uint64_t field : fields) {
        little_u64(field, encoded);
        ok = ok && write_all(fd, encoded.data(), encoded.size());
    }
    const size_t values = out.row_count * out.embedding_width;
    ok = ok && write_all(fd, out.rows, values * sizeof(float));
    ok = ok && ::fsync(fd) == 0;
    const int close_result = ::close(fd);
    if (!ok || close_result != 0) {
        ::unlink(path.c_str());
        set_error(error, error_cap, "could not durably write vision capture");
        return false;
    }
    ++ctx->capture_index;
    return true;
}

void * create(const char * mmproj_path, const char * text_model_path, int gpu,
              char * error, size_t error_cap) {
    if (!mmproj_path || !mmproj_path[0] || !text_model_path ||
        !text_model_path[0] || gpu != 0) {
        set_error(error, error_cap,
                  "llama.cpp Qwen vision adapter requires mmproj, text GGUF, and GPU 0");
        return nullptr;
    }
    llama_backend_init();
    Context * ctx = new Context;
    llama_model_params model_params = llama_model_default_params();
    model_params.vocab_only = true;
    ctx->vocab_model = llama_model_load_from_file(text_model_path, model_params);
    if (!ctx->vocab_model) {
        set_error(error, error_cap, "failed to load vocab-only Qwen text GGUF");
        delete ctx;
        llama_backend_free();
        return nullptr;
    }
    // Pinned mtmd.h defines the output span as
    // llama_model_n_embd_inp(model) * image-token-count floats, and mtmd_init
    // rejects an mmproj whose projection width differs from this text-model
    // value. Query that contract instead of assuming how many floats the mtmd
    // output buffer contains.
    const int32_t embedding_width = llama_model_n_embd_inp(ctx->vocab_model);
    if (embedding_width <= 0 ||
        static_cast<uint32_t>(embedding_width) !=
            kQwen4ExpVisionEmbeddingWidth) {
        char message[160]{};
        std::snprintf(message, sizeof(message),
                      "Qwen text embedding width mismatch: expected %u, got %d",
                      kQwen4ExpVisionEmbeddingWidth, embedding_width);
        set_error(error, error_cap, message);
        llama_model_free(ctx->vocab_model);
        delete ctx;
        llama_backend_free();
        return nullptr;
    }
    ctx->embedding_width = static_cast<uint32_t>(embedding_width);
    mtmd_context_params params = mtmd_context_params_default();
    params.use_gpu = true;
    params.warmup = false; // keep first-image residency lazy and bounded
    ctx->mtmd = mtmd_init_from_file(mmproj_path, ctx->vocab_model, params);
    if (!ctx->mtmd || !mtmd_support_vision(ctx->mtmd) ||
        !mtmd_decode_use_mrope(ctx->mtmd)) {
        set_error(error, error_cap,
                  "mmproj is not a Qwen M-RoPE vision projector");
        if (ctx->mtmd) mtmd_free(ctx->mtmd);
        llama_model_free(ctx->vocab_model);
        delete ctx;
        llama_backend_free();
        return nullptr;
    }
    return ctx;
}

void destroy(void * opaque) {
    Context * ctx = static_cast<Context *>(opaque);
    if (!ctx) return;
    if (ctx->mtmd) mtmd_free(ctx->mtmd);
    if (ctx->vocab_model) llama_model_free(ctx->vocab_model);
    delete ctx;
    llama_backend_free();
}

bool encode(void * opaque, const uint8_t * encoded, size_t encoded_size,
            qwen4exp_vision_provider_output_v1 * out,
            char * error, size_t error_cap) {
    if (out) *out = {};
    Context * ctx = static_cast<Context *>(opaque);
    if (!ctx || !out || !encoded || encoded_size == 0) {
        set_error(error, error_cap, "invalid llama.cpp vision encode request");
        return false;
    }
    mtmd_helper_bitmap_wrapper wrapper = mtmd_helper_bitmap_init_from_buf(
        ctx->mtmd, encoded, encoded_size, false);
    if (!wrapper.bitmap || wrapper.video_ctx) {
        if (wrapper.bitmap) mtmd_bitmap_free(wrapper.bitmap);
        if (wrapper.video_ctx) mtmd_helper_video_free(wrapper.video_ctx);
        set_error(error, error_cap, "image decode failed or input was video");
        return false;
    }
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const char * marker = mtmd_get_marker(ctx->mtmd);
    mtmd_input_text text{marker, std::strlen(marker), false, true};
    const mtmd_bitmap * bitmap = wrapper.bitmap;
    const int32_t tokenized = mtmd_tokenize(ctx->mtmd, chunks, &text,
                                            &bitmap, 1);
    mtmd_bitmap_free(wrapper.bitmap);
    if (tokenized != 0) {
        mtmd_input_chunks_free(chunks);
        set_error(error, error_cap, "Qwen image preprocessing failed");
        return false;
    }
    const mtmd_input_chunk * image_chunk = nullptr;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            if (image_chunk) {
                mtmd_input_chunks_free(chunks);
                set_error(error, error_cap, "still image produced multiple vision chunks");
                return false;
            }
            image_chunk = chunk;
        }
    }
    if (!image_chunk || mtmd_encode_chunk(ctx->mtmd, image_chunk) != 0) {
        mtmd_input_chunks_free(chunks);
        set_error(error, error_cap, "Qwen vision graph execution failed");
        return false;
    }
    const size_t rows = mtmd_input_chunk_get_n_tokens(image_chunk);
    const mtmd_image_tokens * tokens =
        mtmd_input_chunk_get_tokens_image(image_chunk);
    uint32_t max_t = 0, max_h = 0, max_w = 0;
    for (size_t i = 0; i < rows; ++i) {
        const mtmd_decoder_pos p =
            mtmd_image_tokens_get_decoder_pos(tokens, 0, i);
        max_t = std::max(max_t, p.t);
        max_h = std::max(max_h, p.x);
        max_w = std::max(max_w, p.y);
    }
    const size_t width = ctx->embedding_width;
    if (width != kQwen4ExpVisionEmbeddingWidth || rows == 0 ||
        rows > std::numeric_limits<size_t>::max() / width / sizeof(float) ||
        max_t != 0 || max_h >= std::numeric_limits<uint32_t>::max() / 2 ||
        max_w >= std::numeric_limits<uint32_t>::max() / 2 ||
        static_cast<size_t>(max_h + 1) >
            std::numeric_limits<size_t>::max() /
                static_cast<size_t>(max_w + 1) ||
        static_cast<size_t>(max_h + 1) * static_cast<size_t>(max_w + 1) !=
            rows) {
        mtmd_input_chunks_free(chunks);
        set_error(error, error_cap, "invalid Qwen still-image output grid");
        return false;
    }
    const size_t values = rows * width;
    const size_t result_bytes = values * sizeof(float);
    float * result = static_cast<float *>(std::malloc(result_bytes));
    const float * source = mtmd_get_output_embd(ctx->mtmd);
    if (!result || !source) {
        std::free(result);
        mtmd_input_chunks_free(chunks);
        set_error(error, error_cap, "Qwen vision output allocation failed");
        return false;
    }
    std::memcpy(result, source, result_bytes);
    out->grid_t = 1;
    // mtmd decoder positions are post-merger. Ember's provider contract keeps
    // the official pre-merger HF grid, so restore the 2x2 spatial factor.
    out->grid_h = (max_h + 1) * 2;
    out->grid_w = (max_w + 1) * 2;
    out->embedding_width = static_cast<uint32_t>(width);
    out->row_count = rows;
    out->rows = result;
    mtmd_input_chunks_free(chunks);
    if (!capture_output(ctx, *out, error, error_cap)) {
        std::free(out->rows);
        *out = {};
        return false;
    }
    return true;
}

void free_output(void *, qwen4exp_vision_provider_output_v1 * out) {
    if (!out) return;
    std::free(out->rows);
    *out = {};
}

const qwen4exp_vision_provider_v1 kApi = {
    kQwen4ExpVisionProviderAbi, create, destroy, encode, free_output};

} // namespace

extern "C" __attribute__((visibility("default")))
const qwen4exp_vision_provider_v1 *
qwen4exp_vision_provider_get_v1() { return &kApi; }
