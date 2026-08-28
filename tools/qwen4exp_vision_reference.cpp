// Independent real-weight oracle for Ember's Qwen3.8 vision certification.
//
// This executable is compiled inside the exact runtime-dev image against the
// public mtmd API installed from llama.cpp abdc7a0bf815d3b83e26dd523c6960e4dd597e82.
// It deliberately does not link Ember's provider adapter: the hardware gate
// compares this direct Qwen3-VL mtmd result with the bytes observed by Ember.

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr uint64_t kEmbeddingWidth = 2560;

[[noreturn]] void fail(const char * message) {
    std::fprintf(stderr, "qwen4exp-vision-reference: %s\n", message);
    std::exit(2);
}

std::vector<uint8_t> read_file(const char * path) {
    FILE * file = std::fopen(path, "rb");
    if (!file) fail("cannot open image");
    if (std::fseek(file, 0, SEEK_END) != 0) fail("cannot seek image");
    const long end = std::ftell(file);
    if (end <= 0 || std::fseek(file, 0, SEEK_SET) != 0)
        fail("image is empty or cannot be rewound");
    std::vector<uint8_t> bytes(static_cast<size_t>(end));
    if (std::fread(bytes.data(), 1, bytes.size(), file) != bytes.size() ||
        std::fclose(file) != 0)
        fail("cannot read image exactly");
    return bytes;
}

bool write_all(int fd, const void * data, size_t size) {
    const auto * cursor = static_cast<const uint8_t *>(data);
    while (size != 0) {
        const ssize_t count = ::write(fd, cursor, size);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return false;
        cursor += static_cast<size_t>(count);
        size -= static_cast<size_t>(count);
    }
    return true;
}

void little_u64(uint64_t value, std::array<uint8_t, 8> & out) {
    for (unsigned i = 0; i < 8; ++i)
        out[i] = static_cast<uint8_t>(value >> (i * 8));
}

void write_capture(const char * path, uint64_t grid_h, uint64_t grid_w,
                   size_t rows, const float * values) {
    const int fd = ::open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC |
                          O_NOFOLLOW, S_IRUSR | S_IWUSR);
    if (fd < 0) fail("refusing non-new reference output");
    const std::array<uint8_t, 8> magic{{'E','V','I','S','F','3','2','\0'}};
    const uint64_t fields[] = {1, 1, grid_h, grid_w, kEmbeddingWidth, rows};
    bool ok = write_all(fd, magic.data(), magic.size());
    std::array<uint8_t, 8> encoded{};
    for (uint64_t field : fields) {
        little_u64(field, encoded);
        ok = ok && write_all(fd, encoded.data(), encoded.size());
    }
    if (rows > std::numeric_limits<size_t>::max() /
                   static_cast<size_t>(kEmbeddingWidth) / sizeof(float))
        fail("reference output byte count overflow");
    ok = ok && write_all(fd, values,
                         rows * static_cast<size_t>(kEmbeddingWidth) * sizeof(float));
    ok = ok && ::fsync(fd) == 0;
    const int close_result = ::close(fd);
    if (!ok || close_result != 0) {
        ::unlink(path);
        fail("could not durably write reference output");
    }
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 5)
        fail("usage: MODEL MMPROJ IMAGE NEW_OUTPUT");
    const std::vector<uint8_t> encoded = read_file(argv[3]);
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    if (!model || llama_model_n_embd_inp(model) !=
                      static_cast<int32_t>(kEmbeddingWidth))
        fail("text GGUF is not the pinned Qwen embedding contract");
    mtmd_context_params params = mtmd_context_params_default();
    params.use_gpu = true;
    params.warmup = false;
    mtmd_context * context = mtmd_init_from_file(argv[2], model, params);
    if (!context || !mtmd_support_vision(context) ||
        !mtmd_decode_use_mrope(context))
        fail("mmproj is not a Qwen M-RoPE vision projector");
    mtmd_helper_bitmap_wrapper bitmap = mtmd_helper_bitmap_init_from_buf(
        context, encoded.data(), encoded.size(), false);
    if (!bitmap.bitmap || bitmap.video_ctx)
        fail("reference image decode failed or produced video");
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const char * marker = mtmd_get_marker(context);
    mtmd_input_text text{marker, std::strlen(marker), false, true};
    const mtmd_bitmap * image = bitmap.bitmap;
    if (mtmd_tokenize(context, chunks, &text, &image, 1) != 0)
        fail("reference image preprocessing failed");
    mtmd_bitmap_free(bitmap.bitmap);
    const mtmd_input_chunk * image_chunk = nullptr;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            if (image_chunk) fail("reference produced multiple image chunks");
            image_chunk = chunk;
        }
    }
    if (!image_chunk || mtmd_encode_chunk(context, image_chunk) != 0)
        fail("reference vision graph failed");
    const size_t rows = mtmd_input_chunk_get_n_tokens(image_chunk);
    const mtmd_image_tokens * tokens =
        mtmd_input_chunk_get_tokens_image(image_chunk);
    uint32_t max_t = 0, max_h = 0, max_w = 0;
    for (size_t i = 0; i < rows; ++i) {
        const mtmd_decoder_pos position =
            mtmd_image_tokens_get_decoder_pos(tokens, 0, i);
        max_t = std::max(max_t, position.t);
        max_h = std::max(max_h, position.x);
        max_w = std::max(max_w, position.y);
    }
    const float * output = mtmd_get_output_embd(context);
    if (!output || rows == 0 || max_t != 0 ||
        static_cast<size_t>(max_h + 1) * static_cast<size_t>(max_w + 1) != rows)
        fail("reference output grid is invalid");
    write_capture(argv[4], static_cast<uint64_t>(max_h + 1) * 2,
                  static_cast<uint64_t>(max_w + 1) * 2, rows, output);
    mtmd_input_chunks_free(chunks);
    mtmd_free(context);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
