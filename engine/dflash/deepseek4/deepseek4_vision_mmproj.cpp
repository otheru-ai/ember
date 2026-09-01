#include "deepseek4_vision_mmproj.h"

#include "common/gguf_bounds.h"

#include "ggml.h"
#include "gguf.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sys/stat.h>
#include <unordered_map>
#include <unordered_set>
#include <unistd.h>

namespace dflash {
namespace {

bool require_string(const gguf_context * gguf, const char * key,
                    const char * expected, std::string & error) {
    const int64_t id = gguf_find_key(gguf, key);
    const char * value = id >= 0 && gguf_get_kv_type(gguf, id) == GGUF_TYPE_STRING
        ? gguf_get_val_str(gguf, id) : nullptr;
    if (!value || std::strcmp(value, expected) != 0) {
        error = "DeepSeek4 mmproj metadata differs: " + std::string(key);
        return false;
    }
    return true;
}

bool require_u32(const gguf_context * gguf, const char * key,
                 uint32_t expected, std::string & error) {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_UINT32 ||
        gguf_get_val_u32(gguf, id) != expected) {
        error = "DeepSeek4 mmproj metadata differs: " + std::string(key);
        return false;
    }
    return true;
}

bool require_f32(const gguf_context * gguf, const char * key,
                 float expected, std::string & error) {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_FLOAT32 ||
        gguf_get_val_f32(gguf, id) != expected) {
        error = "DeepSeek4 mmproj metadata differs: " + std::string(key);
        return false;
    }
    return true;
}

bool require_bool(const gguf_context * gguf, const char * key,
                  bool expected, std::string & error) {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_BOOL ||
        gguf_get_val_bool(gguf, id) != expected) {
        error = "DeepSeek4 mmproj metadata differs: " + std::string(key);
        return false;
    }
    return true;
}

bool validate_config(const gguf_context * gguf, int32_t model_n_embd,
                     std::string & error) {
    const auto & config = deepseek4_vision_native_config();
    if (model_n_embd != config.output_embedding_length) {
        error = "DeepSeek4 mmproj output width differs from the language model";
        return false;
    }
    return
        require_string(gguf, "clip.projector_type", "deepseekv4vision", error) &&
        require_f32(gguf, "clip.vision.attention.layer_norm_epsilon",
                    config.norm_epsilon, error) &&
        require_bool(gguf, "clip.use_silu", true, error) &&
        require_u32(gguf, "clip.vision.projector.scale_factor",
                    static_cast<uint32_t>(config.scale_factor), error) &&
        require_u32(gguf, "clip.vision.image_min_pixels",
                    static_cast<uint32_t>(config.image_min_pixels), error) &&
        require_f32(gguf, "clip.vision.rope_theta", config.rope_theta, error) &&
        require_u32(gguf, "clip.vision.max_n_token",
                    static_cast<uint32_t>(config.max_n_token), error) &&
        require_f32(gguf, "clip.vision.max_wh_ratio",
                    config.max_wh_ratio, error) &&
        require_u32(gguf, "clip.vision.image_size",
                    static_cast<uint32_t>(config.nominal_image_size), error) &&
        require_u32(gguf, "clip.vision.patch_size",
                    static_cast<uint32_t>(config.patch_size), error) &&
        require_u32(gguf, "clip.vision.embedding_length",
                    static_cast<uint32_t>(config.embedding_length), error) &&
        require_u32(gguf, "clip.vision.feed_forward_length",
                    static_cast<uint32_t>(config.feed_forward_length), error) &&
        require_u32(gguf, "clip.vision.attention.head_count",
                    static_cast<uint32_t>(config.head_count), error) &&
        require_u32(gguf, "clip.vision.block_count",
                    static_cast<uint32_t>(config.block_count), error);
}

bool tensor_shape_matches(const ggml_tensor * tensor,
                          const std::vector<int64_t> & expected) {
    if (!tensor || expected.empty() || expected.size() > GGML_MAX_DIMS) {
        return false;
    }
    for (size_t dim = 0; dim < expected.size(); ++dim) {
        if (tensor->ne[dim] != expected[dim]) return false;
    }
    for (size_t dim = expected.size(); dim < GGML_MAX_DIMS; ++dim) {
        if (tensor->ne[dim] != 1) return false;
    }
    return true;
}

}  // namespace

bool deepseek4_validate_vision_mmproj_metadata(
        const gguf_context * gguf, ggml_context * meta,
        int32_t model_n_embd, size_t file_size,
        Deepseek4VisionMmprojMetadata & out, std::string & error) {
    out = {};
    error.clear();
    if (!gguf || !meta || !validate_config(gguf, model_n_embd, error)) {
        if (error.empty()) error = "invalid DeepSeek4 mmproj metadata context";
        return false;
    }
    const auto specs = deepseek4_vision_tensor_specs();
    if (gguf_get_n_tensors(gguf) != static_cast<int64_t>(specs.size())) {
        error = "DeepSeek4 mmproj tensor count differs from 299";
        return false;
    }
    std::unordered_map<std::string, const Deepseek4VisionTensorSpec *> expected;
    expected.reserve(specs.size());
    for (const auto & spec : specs) expected.emplace(spec.name, &spec);

    const size_t data_offset = gguf_get_data_offset(gguf);
    std::unordered_set<std::string> seen;
    seen.reserve(specs.size());
    out.config = deepseek4_vision_native_config();
    out.file_size = file_size;
    out.data_offset = data_offset;
    out.tensors.reserve(specs.size());
    for (int64_t id = 0; id < gguf_get_n_tensors(gguf); ++id) {
        const char * raw_name = gguf_get_tensor_name(gguf, id);
        if (!raw_name) {
            error = "DeepSeek4 mmproj contains an unnamed tensor";
            out = {};
            return false;
        }
        const std::string name(raw_name);
        const auto found = expected.find(name);
        if (found == expected.end() || !seen.insert(name).second) {
            error = "DeepSeek4 mmproj contains an unknown or duplicate tensor: " + name;
            out = {};
            return false;
        }
        const Deepseek4VisionTensorSpec & spec = *found->second;
        ggml_tensor * tensor = ggml_get_tensor(meta, raw_name);
        const ggml_type expected_type =
            spec.storage == Deepseek4VisionStorage::F16
                ? GGML_TYPE_F16 : GGML_TYPE_F32;
        if (!tensor || tensor->type != expected_type ||
            !tensor_shape_matches(tensor, spec.shape) ||
            !ggml_is_contiguous(tensor)) {
            error = "DeepSeek4 mmproj tensor shape/type differs: " + name;
            out = {};
            return false;
        }
        const size_t tensor_offset = gguf_get_tensor_offset(gguf, id);
        const size_t tensor_size = gguf_get_tensor_size(gguf, id);
        if (tensor_size != ggml_nbytes(tensor) ||
            !common::gguf_tensor_in_file(data_offset, tensor_offset,
                                         tensor_size, file_size)) {
            error = common::gguf_bounds_error(
                "DeepSeek4 mmproj GGUF", raw_name,
                ggml_type_name(tensor->type), data_offset, tensor_offset,
                tensor_size, file_size);
            out = {};
            return false;
        }
        out.tensors.push_back({name, spec.shape, spec.storage,
                               data_offset + tensor_offset, tensor_size});
    }
    if (seen.size() != specs.size()) {
        error = "DeepSeek4 mmproj is missing a required tensor";
        out = {};
        return false;
    }
    return true;
}

bool deepseek4_load_vision_mmproj_metadata(
        const std::string & path, int32_t model_n_embd,
        Deepseek4VisionMmprojMetadata & out, std::string & error) {
    out = {};
    error.clear();
    if (path.empty()) {
        error = "invalid DeepSeek4 mmproj path";
        return false;
    }
    const int fd = open(path.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) {
        error = "cannot open DeepSeek4 mmproj GGUF";
        return false;
    }
    struct stat st {};
    if (fstat(fd, &st) != 0) {
        close(fd);
        error = "cannot stat DeepSeek4 mmproj GGUF";
        return false;
    }
    if (!S_ISREG(st.st_mode) || st.st_size < 0) {
        close(fd);
        error = "DeepSeek4 mmproj GGUF is not a regular file";
        return false;
    }
    const uintmax_t wide_size = static_cast<uintmax_t>(st.st_size);
    if (wide_size > static_cast<uintmax_t>(SIZE_MAX)) {
        close(fd);
        error = "DeepSeek4 mmproj GGUF is too large to address";
        return false;
    }
    const size_t file_size = static_cast<size_t>(wide_size);
    const int parse_fd = fcntl(fd, F_DUPFD_CLOEXEC, 0);
    if (parse_fd < 0) {
        close(fd);
        error = "cannot duplicate DeepSeek4 mmproj descriptor";
        return false;
    }
    std::FILE * stream = fdopen(parse_fd, "rb");
    if (!stream) {
        close(parse_fd);
        close(fd);
        error = "cannot open DeepSeek4 mmproj metadata stream";
        return false;
    }
    ggml_context * meta = nullptr;
    gguf_init_params params {};
    params.no_alloc = true;
    params.ctx = &meta;
    gguf_context * gguf = gguf_init_from_file_ptr(stream, params);
    const int stream_close = std::fclose(stream);
    if (!gguf || !meta || stream_close != 0) {
        if (gguf) gguf_free(gguf);
        if (meta) ggml_free(meta);
        close(fd);
        error = !gguf || !meta
            ? "cannot parse DeepSeek4 mmproj GGUF metadata"
            : "cannot close DeepSeek4 mmproj metadata stream";
        return false;
    }
    const bool valid = deepseek4_validate_vision_mmproj_metadata(
        gguf, meta, model_n_embd, file_size, out, error);
    gguf_free(gguf);
    ggml_free(meta);
    if (close(fd) != 0 && valid) {
        out = {};
        error = "cannot close DeepSeek4 mmproj GGUF";
        return false;
    }
    return valid;
}

}  // namespace dflash
