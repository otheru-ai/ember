#include "deepseek4_vision_biases.h"

#include "deepseek4_vision_contract.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace dflash {
namespace {

bool parse_bias_name(const char * name, int & layer) {
    constexpr char prefix[] = "blk.";
    if (!name || std::strncmp(name, prefix, sizeof(prefix) - 1) != 0) {
        return false;
    }
    const char * cursor = name + sizeof(prefix) - 1;
    if (*cursor < '0' || *cursor > '9') return false;
    if (*cursor == '0' && cursor[1] >= '0' && cursor[1] <= '9') return false;
    int value = 0;
    while (*cursor >= '0' && *cursor <= '9') {
        const int digit = *cursor - '0';
        if (value > (std::numeric_limits<int>::max() - digit) / 10) {
            return false;
        }
        value = value * 10 + digit;
        ++cursor;
    }
    if (*cursor != '.') return false;
    if (!deepseek4_is_vision_router_bias_suffix(cursor + 1)) return false;
    layer = value;
    return true;
}

bool has_vision_bias_leaf(const char * name) {
    constexpr char leaf[] = ".exp_probs_b_vl.bias";
    if (!name) return false;
    const size_t name_len = std::strlen(name);
    const size_t leaf_len = sizeof(leaf) - 1;
    return name_len >= leaf_len &&
           std::strcmp(name + name_len - leaf_len, leaf) == 0;
}

}  // namespace

bool deepseek4_inspect_vision_router_biases(
        const std::string & model_path, int expected_layers,
        int expected_experts, Deepseek4VisionBiasInventory & out,
        std::string & error) {
    out = {};
    error.clear();
    if (model_path.empty() || expected_layers <= 0 || expected_experts <= 0) {
        error = "invalid DeepSeek4 vision-bias inventory request";
        return false;
    }

    const int fd = open(model_path.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) {
        error = "cannot open DeepSeek4 language GGUF";
        return false;
    }
    struct stat st {};
    if (fstat(fd, &st) != 0) {
        close(fd);
        error = "cannot stat DeepSeek4 language GGUF";
        return false;
    }
    if (!S_ISREG(st.st_mode)) {
        close(fd);
        error = "DeepSeek4 language GGUF is not a regular file";
        return false;
    }
    const int parse_fd = fcntl(fd, F_DUPFD_CLOEXEC, 0);
    close(fd);
    if (parse_fd < 0) {
        error = "cannot duplicate DeepSeek4 language GGUF descriptor";
        return false;
    }
    std::FILE * stream = fdopen(parse_fd, "rb");
    if (!stream) {
        close(parse_fd);
        error = "cannot open DeepSeek4 language GGUF metadata stream";
        return false;
    }

    ggml_context * meta = nullptr;
    gguf_init_params params {};
    params.no_alloc = true;
    params.ctx = &meta;
    gguf_context * gguf = gguf_init_from_file_ptr(stream, params);
    const int close_result = std::fclose(stream);
    if (!gguf || !meta || close_result != 0) {
        if (gguf) gguf_free(gguf);
        if (meta) ggml_free(meta);
        error = !gguf || !meta
            ? "cannot parse DeepSeek4 language GGUF metadata"
            : "cannot close DeepSeek4 language GGUF metadata stream";
        return false;
    }

    bool valid = true;
    const int64_t arch_id = gguf_find_key(gguf, "general.architecture");
    const char * architecture = arch_id >= 0 &&
        gguf_get_kv_type(gguf, arch_id) == GGUF_TYPE_STRING
            ? gguf_get_val_str(gguf, arch_id) : nullptr;
    if (arch_id < 0 || gguf_get_kv_type(gguf, arch_id) != GGUF_TYPE_STRING ||
        !architecture || std::strcmp(architecture, "deepseek4") != 0) {
        error = "language GGUF architecture is not deepseek4";
        valid = false;
    }

    std::vector<unsigned char> seen(static_cast<size_t>(expected_layers), 0);
    const int64_t tensor_count = gguf_get_n_tensors(gguf);
    for (int64_t id = 0; valid && id < tensor_count; ++id) {
        const char * name = gguf_get_tensor_name(gguf, id);
        int layer = -1;
        if (!parse_bias_name(name, layer)) {
            if (has_vision_bias_leaf(name)) {
                error = "vision router bias has a noncanonical block name";
                valid = false;
            }
            continue;
        }
        if (layer < 0 || layer >= expected_layers ||
            seen[static_cast<size_t>(layer)] != 0) {
            error = "vision router bias has an out-of-range layer";
            valid = false;
            break;
        }
        ggml_tensor * tensor = ggml_get_tensor(meta, name);
        if (!tensor || tensor->type != GGML_TYPE_F32 ||
            ggml_n_dims(tensor) != 1 || tensor->ne[0] != expected_experts) {
            error = "vision router bias must be one F32 expert row: " +
                    std::string(name ? name : "(null)");
            valid = false;
            break;
        }
        seen[static_cast<size_t>(layer)] = 1;
        out.layer_ids.push_back(layer);
    }
    if (valid) {
        for (int layer = 0; layer < expected_layers; ++layer) {
            if (seen[static_cast<size_t>(layer)] == 0) {
                error = "language GGUF is missing vision router bias for layer " +
                        std::to_string(layer);
                valid = false;
                break;
            }
        }
    }
    if (valid) std::sort(out.layer_ids.begin(), out.layer_ids.end());

    gguf_free(gguf);
    ggml_free(meta);
    if (!valid) out = {};
    return valid;
}

}  // namespace dflash
