#include "q8_model_weights.h"

#include "q8_0_pack.h"

extern "C" {
#include "../../src/model/gguf.h"
}

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ember::xdna2 {
namespace {

using GgufOwner = std::unique_ptr<gguf_file, decltype(&gguf_free)>;

bool fail(std::string * error, const std::string & message) {
    if (error) *error = message;
    return false;
}

const gguf_tensor * find_tensor(const gguf_file * file,
                                const std::string & name) {
    for (uint64_t i = 0; i < file->n_tensors; ++i) {
        if (std::strcmp(file->tensors[i].name, name.c_str()) == 0)
            return &file->tensors[i];
    }
    return nullptr;
}

bool checked_add(uint64_t a, uint64_t b, uint64_t * result) {
    if (a > std::numeric_limits<uint64_t>::max() - b) return false;
    *result = a + b;
    return true;
}

bool validate_projection(const gguf_file * file,
                         uint64_t file_size,
                         const std::string & name,
                         uint64_t dim0,
                         uint64_t dim1,
                         size_t projection_bytes,
                         const gguf_tensor ** result,
                         std::string * error) {
    const gguf_tensor * tensor = find_tensor(file, name);
    if (!tensor) return fail(error, "missing tensor " + name);
    if (tensor->type != kQ8GgmlType)
        return fail(error, name + " must use Q8_0 (type 8)");
    if (tensor->n_dims != 2 || tensor->dims[0] != dim0 ||
        tensor->dims[1] != dim1) {
        return fail(error, name + " has an incompatible shared-expert shape");
    }
    uint64_t absolute = 0;
    uint64_t end = 0;
    if (!checked_add(file->data_offset, tensor->offset, &absolute) ||
        !checked_add(absolute, projection_bytes, &end) || end > file_size) {
        return fail(error, name + " extends beyond the GGUF file");
    }
    *result = tensor;
    return true;
}

bool pread_all(int fd, uint64_t offset, std::vector<uint8_t> & destination,
               const std::string & name, std::string * error) {
    if (offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max()))
        return fail(error, name + " offset exceeds off_t");
    size_t complete = 0;
    while (complete < destination.size()) {
        const uint64_t position = offset + complete;
        if (position > static_cast<uint64_t>(
                           std::numeric_limits<off_t>::max())) {
            return fail(error, name + " read offset exceeds off_t");
        }
        const ssize_t count = ::pread(
            fd, destination.data() + complete, destination.size() - complete,
            static_cast<off_t>(position));
        if (count < 0) {
            if (errno == EINTR) continue;
            return fail(error, "read failed for " + name + ": " +
                         std::strerror(errno));
        }
        if (count == 0) return fail(error, "short read for " + name);
        complete += static_cast<size_t>(count);
    }
    return true;
}

}  // namespace

bool load_q8_model_projection(const char * path,
                              const char * tensor_name,
                              int k,
                              int n,
                              Q8ModelProjection & projection,
                              std::string * error) {
    projection = {};
    if (!path || !path[0]) return fail(error, "model path is empty");
    if (!tensor_name || !tensor_name[0])
        return fail(error, "tensor name is empty");
    const size_t raw_bytes = q8_projection_bytes(k, n);
    if (!raw_bytes || !q8_supported_shape(k, n))
        return fail(error, "unsupported Q8 AIE projection shape");

    GgufOwner file(gguf_open(path), &gguf_free);
    if (!file) return fail(error, std::string("cannot parse GGUF: ") + path);
    if (std::strcmp(gguf_get_str(file.get(), "general.architecture", ""),
                    "deepseek4-dflash-draft") != 0) {
        return fail(error, "GGUF is not a deepseek4-dflash-draft model");
    }
    const int fd = ::open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return fail(error, std::string("cannot open model: ") +
                     std::strerror(errno));
    struct FdOwner {
        int fd;
        ~FdOwner() { if (fd >= 0) ::close(fd); }
    } fd_owner{fd};
    struct stat status {};
    if (::fstat(fd, &status) != 0 || status.st_size < 0)
        return fail(error, "cannot determine model file size");

    const gguf_tensor * tensor = nullptr;
    if (!validate_projection(file.get(), static_cast<uint64_t>(status.st_size),
                             tensor_name, static_cast<uint64_t>(k),
                             static_cast<uint64_t>(n), raw_bytes, &tensor,
                             error)) return false;
    projection.name = tensor_name;
    projection.k = k;
    projection.n = n;
    projection.raw.resize(raw_bytes);
    if (!pread_all(fd, file->data_offset + tensor->offset, projection.raw,
                   tensor->name, error)) {
        projection = {};
        return false;
    }
    std::string pack_error;
    if (!pack_q8_projection_corrected_bf16(
            projection.raw.data(), projection.raw.size(), k, n,
            projection.packed, &pack_error)) {
        projection = {};
        return fail(error, "cannot pack projection: " + pack_error);
    }
    return true;
}

bool load_q8_model_shared_expert(const char * path,
                                 int layer,
                                 Q8ModelSharedExpert & expert,
                                 std::string * error) {
    expert = {};
    if (!path || !path[0]) return fail(error, "model path is empty");
    if (layer < 0 || layer >= kQ8ModelLayers)
        return fail(error, "DSpark layer must be in [0,2]");

    GgufOwner file(gguf_open(path), &gguf_free);
    if (!file) return fail(error, std::string("cannot parse GGUF: ") + path);
    if (std::strcmp(gguf_get_str(file.get(), "general.architecture", ""),
                    "deepseek4-dflash-draft") != 0) {
        return fail(error, "GGUF is not a deepseek4-dflash-draft model");
    }
    if (gguf_get_int(file.get(),
                     "deepseek4-dflash-draft.block_count", -1) !=
            kQ8ModelLayers) {
        return fail(error, "GGUF DSpark layer metadata is incompatible");
    }

    const int fd = ::open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return fail(error, std::string("cannot open model: ") +
                     std::strerror(errno));
    struct FdOwner {
        int fd;
        ~FdOwner() { if (fd >= 0) ::close(fd); }
    } fd_owner{fd};
    struct stat status {};
    if (::fstat(fd, &status) != 0 || status.st_size < 0)
        return fail(error, "cannot determine model file size");
    const uint64_t file_size = static_cast<uint64_t>(status.st_size);

    const size_t gate_bytes = q8_projection_bytes(4096, 2048);
    const size_t down_bytes = q8_projection_bytes(2048, 4096);
    if (!gate_bytes || !down_bytes)
        return fail(error, "internal Q8 projection shape failure");
    char tensor_name[96];
    const gguf_tensor * gate_tensor = nullptr;
    const gguf_tensor * up_tensor = nullptr;
    const gguf_tensor * down_tensor = nullptr;
    std::snprintf(tensor_name, sizeof(tensor_name),
                  "blk.%d.ffn_gate_shexp.weight", layer);
    if (!validate_projection(file.get(), file_size, tensor_name, 4096, 2048,
                             gate_bytes, &gate_tensor, error)) return false;
    std::snprintf(tensor_name, sizeof(tensor_name),
                  "blk.%d.ffn_up_shexp.weight", layer);
    if (!validate_projection(file.get(), file_size, tensor_name, 4096, 2048,
                             gate_bytes, &up_tensor, error)) return false;
    std::snprintf(tensor_name, sizeof(tensor_name),
                  "blk.%d.ffn_down_shexp.weight", layer);
    if (!validate_projection(file.get(), file_size, tensor_name, 2048, 4096,
                             down_bytes, &down_tensor, error)) return false;

    expert.layer = layer;
    expert.gate.resize(gate_bytes);
    expert.up.resize(gate_bytes);
    expert.down.resize(down_bytes);
    const auto absolute = [&](const gguf_tensor * tensor) {
        return file->data_offset + tensor->offset;
    };
    if (!pread_all(fd, absolute(gate_tensor), expert.gate,
                   gate_tensor->name, error) ||
        !pread_all(fd, absolute(up_tensor), expert.up,
                   up_tensor->name, error) ||
        !pread_all(fd, absolute(down_tensor), expert.down,
                   down_tensor->name, error)) {
        expert = {};
        return false;
    }
    std::string pack_error;
    if (!pack_q8_expert_v2(
            expert.gate.data(), expert.gate.size(), expert.up.data(),
            expert.up.size(), expert.down.data(), expert.down.size(),
            expert.packed, &pack_error)) {
        expert = {};
        return fail(error, "cannot pack shared expert: " + pack_error);
    }
    return true;
}

}  // namespace ember::xdna2
