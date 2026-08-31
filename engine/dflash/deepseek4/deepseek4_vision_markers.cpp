#include "deepseek4_vision_markers.h"

#include "common/gguf_bounds.h"

#include "ggml.h"
#include "gguf.h"

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace dflash {
namespace {

struct MarkerDescriptor {
    const char * name;
    std::vector<float> Deepseek4ImageMarkers::* destination;
};

constexpr MarkerDescriptor kMarkers[] = {
    {"mm.image_begin.weight", &Deepseek4ImageMarkers::start},
    {"mm.image_pad.weight", &Deepseek4ImageMarkers::pad},
    {"v.image_newline.weight", &Deepseek4ImageMarkers::newline},
    {"mm.image_end.weight", &Deepseek4ImageMarkers::end},
};

bool read_exact_at(int fd, size_t offset, void * destination, size_t bytes) {
    auto * out = static_cast<uint8_t *>(destination);
    size_t done = 0;
    while (done < bytes) {
        const ssize_t n = pread(fd, out + done, bytes - done,
                                static_cast<off_t>(offset + done));
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            return false;
        }
        done += static_cast<size_t>(n);
    }
    return true;
}

}  // namespace

bool deepseek4_load_image_markers(const std::string & mmproj_path,
                                  int32_t model_n_embd,
                                  Deepseek4ImageMarkers & out,
                                  std::string & error) {
    out = {};
    error.clear();
    if (mmproj_path.empty() || model_n_embd <= 0) {
        error = "invalid DeepSeek4 mmproj path or model embedding width";
        return false;
    }

    const int fd = open(mmproj_path.c_str(),
                        O_RDONLY | O_NONBLOCK | O_CLOEXEC);
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
    const uintmax_t file_size_wide = static_cast<uintmax_t>(st.st_size);
    if (file_size_wide > static_cast<uintmax_t>(SIZE_MAX)) {
        close(fd);
        error = "DeepSeek4 mmproj GGUF is too large to address";
        return false;
    }
    const size_t file_size = static_cast<size_t>(file_size_wide);

    const int parse_fd = fcntl(fd, F_DUPFD_CLOEXEC, 0);
    if (parse_fd < 0) {
        close(fd);
        error = "cannot duplicate DeepSeek4 mmproj descriptor";
        return false;
    }
    std::FILE * parse_file = fdopen(parse_fd, "rb");
    if (!parse_file) {
        close(parse_fd);
        close(fd);
        error = "cannot open DeepSeek4 mmproj metadata stream";
        return false;
    }

    ggml_context * meta = nullptr;
    gguf_init_params params {};
    params.no_alloc = true;
    params.ctx = &meta;
    gguf_context * gguf = gguf_init_from_file_ptr(parse_file, params);
    const int stream_close = std::fclose(parse_file);
    if (!gguf || !meta || stream_close != 0) {
        if (gguf) gguf_free(gguf);
        if (meta) ggml_free(meta);
        close(fd);
        error = !gguf || !meta
            ? "cannot parse DeepSeek4 mmproj GGUF metadata"
            : "cannot close DeepSeek4 mmproj metadata stream";
        return false;
    }

    bool valid = true;
    const size_t data_offset = gguf_get_data_offset(gguf);
    const size_t expected_bytes = static_cast<size_t>(model_n_embd) *
                                  sizeof(float);
    for (const MarkerDescriptor & marker : kMarkers) {
        const int64_t id = gguf_find_tensor(gguf, marker.name);
        if (id < 0) {
            error = "DeepSeek4 mmproj is missing required marker tensor: " +
                    std::string(marker.name);
            valid = false;
            break;
        }
        ggml_tensor * tensor = ggml_get_tensor(meta, marker.name);
        if (!tensor || tensor->type != GGML_TYPE_F32 ||
            tensor->ne[0] != model_n_embd ||
            ggml_nelements(tensor) != model_n_embd ||
            !ggml_is_contiguous(tensor) ||
            gguf_get_tensor_size(gguf, id) != expected_bytes) {
            error = "DeepSeek4 mmproj marker must be one contiguous F32 "
                    "embedding row: " + std::string(marker.name);
            valid = false;
            break;
        }
        const size_t tensor_offset = gguf_get_tensor_offset(gguf, id);
        const size_t tensor_size = gguf_get_tensor_size(gguf, id);
        if (!common::gguf_tensor_in_file(data_offset, tensor_offset,
                                         tensor_size, file_size)) {
            error = common::gguf_bounds_error(
                "DeepSeek4 mmproj GGUF", marker.name,
                ggml_type_name(tensor->type), data_offset, tensor_offset,
                tensor_size, file_size);
            valid = false;
            break;
        }
        std::vector<float> & values = out.*(marker.destination);
        values.resize(static_cast<size_t>(model_n_embd));
        if (!read_exact_at(fd, data_offset + tensor_offset, values.data(),
                           tensor_size)) {
            error = "cannot read complete DeepSeek4 mmproj marker tensor: " +
                    std::string(marker.name);
            valid = false;
            break;
        }
    }

    gguf_free(gguf);
    ggml_free(meta);
    if (close(fd) != 0 && valid) {
        error = "cannot close DeepSeek4 mmproj GGUF";
        valid = false;
    }
    if (!valid) {
        out = {};
    }
    return valid;
}

}  // namespace dflash
