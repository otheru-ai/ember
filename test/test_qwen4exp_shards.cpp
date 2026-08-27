#include "qwen4exp_model.h"

#include "ggml.h"
#include "gguf.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using dflash::common::qwen4exp_discover_gguf_shards;
using dflash::common::validate_qwen4exp_gguf;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(condition, message) do {                                      \
    if (condition) { ++g_pass; } else {                                     \
        ++g_fail; std::fprintf(stderr, "FAIL: %s\n", message);             \
    }                                                                        \
} while (0)

namespace {

struct TempDir {
    std::filesystem::path path;

    TempDir() {
        char pattern[] = "/tmp/ember-qwen-shards-XXXXXX";
        char * created = ::mkdtemp(pattern);
        if (!created) {
            std::fprintf(stderr, "mkdtemp failed: %s\n", std::strerror(errno));
            std::abort();
        }
        path = created;
    }

    ~TempDir() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
};

void write_byte(const std::filesystem::path & path) {
    FILE * file = std::fopen(path.c_str(), "wb");
    if (!file || std::fputc(0, file) == EOF || std::fclose(file) != 0) {
        std::fprintf(stderr, "failed to create %s\n", path.c_str());
        std::abort();
    }
}

void write_tiny_gguf(const std::filesystem::path & path, uint16_t split_no,
                     uint16_t split_count, int32_t tensor_count,
                     const char * tensor_name, const char * invariant,
                     bool include_tensor_data = true) {
    ggml_init_params model_params{
        /*.mem_size=*/1024 * 1024,
        /*.mem_buffer=*/nullptr,
        /*.no_alloc=*/true,
    };
    ggml_context * model = ggml_init(model_params);
    gguf_context * gguf = gguf_init_empty();
    if (!model || !gguf) std::abort();
    gguf_set_val_str(gguf, "general.architecture", "qwen4exp");
    gguf_set_val_str(gguf, "test.invariant", invariant);
    gguf_set_val_u16(gguf, "split.no", split_no);
    gguf_set_val_u16(gguf, "split.count", split_count);
    gguf_set_val_i32(gguf, "split.tensors.count", tensor_count);
    ggml_tensor * tensor = ggml_new_tensor_1d(model, GGML_TYPE_F32, 1);
    ggml_set_name(tensor, tensor_name);
    gguf_add_tensor(gguf, tensor);
    if (!gguf_write_to_file(gguf, path.c_str(), true)) std::abort();

    if (include_tensor_data) {
        // A writer context has no file-derived data_offset yet. Its serialized
        // metadata size is the actual aligned tensor-data origin.
        const size_t end = gguf_get_meta_size(gguf) +
                           gguf_get_tensor_offset(gguf, 0) +
                           gguf_get_tensor_size(gguf, 0);
        const int fd = ::open(path.c_str(), O_WRONLY | O_CLOEXEC);
        if (fd < 0 || ::ftruncate(fd, static_cast<off_t>(end)) != 0) std::abort();
        ::close(fd);
    }
    gguf_free(gguf);
    ggml_free(model);
}

std::filesystem::path shard(const TempDir & dir, int number, int count) {
    char name[80];
    std::snprintf(name, sizeof(name), "model-%05d-of-%05d.gguf", number, count);
    return dir.path / name;
}

} // namespace

int main() {
    {
        TempDir dir;
        const auto first = shard(dir, 1, 3);
        const auto second = shard(dir, 2, 3);
        const auto third = shard(dir, 3, 3);
        write_byte(first); write_byte(second); write_byte(third);
        std::vector<std::string> paths;
        std::string error;
        CHECK(qwen4exp_discover_gguf_shards(second.c_str(), paths, error),
              "any canonical shard discovers its complete set");
        CHECK(paths.size() == 3 && paths[0] == first.string() &&
                  paths[1] == second.string() && paths[2] == third.string(),
              "canonical shards are returned in exact numeric order");
    }
    {
        TempDir dir;
        write_byte(shard(dir, 1, 2));
        std::vector<std::string> paths;
        std::string error;
        CHECK(!qwen4exp_discover_gguf_shards(shard(dir, 1, 2).c_str(), paths,
                                             error) &&
                  error.find("missing") != std::string::npos,
              "a missing canonical shard fails closed");
    }
    {
        TempDir dir;
        write_byte(shard(dir, 1, 2)); write_byte(shard(dir, 2, 2));
        write_byte(shard(dir, 1, 3));
        std::vector<std::string> paths;
        std::string error;
        CHECK(!qwen4exp_discover_gguf_shards(shard(dir, 2, 2).c_str(), paths,
                                             error) &&
                  error.find("mismatched filename shard count") !=
                      std::string::npos,
              "one stem cannot advertise conflicting shard counts");
    }
    {
        TempDir dir;
        const auto first = shard(dir, 1, 2);
        const auto second = shard(dir, 2, 2);
        write_byte(first);
        if (::link(first.c_str(), second.c_str()) != 0) std::abort();
        std::vector<std::string> paths;
        std::string error;
        CHECK(!qwen4exp_discover_gguf_shards(first.c_str(), paths, error) &&
                  error.find("duplicate inode") != std::string::npos,
              "hard-linked shard aliases fail closed");
    }
    {
        TempDir dir;
        const auto first = shard(dir, 1, 2);
        const auto second = shard(dir, 2, 2);
        write_tiny_gguf(first, 0, 2, 2, "one.weight", "same");
        write_tiny_gguf(second, 0, 2, 2, "two.weight", "same");
        std::string error;
        CHECK(!validate_qwen4exp_gguf(second.c_str(), error) &&
                  error.find("split.no/count") != std::string::npos,
              "split.no is checked against filename order");
    }
    {
        TempDir dir;
        const auto first = shard(dir, 1, 2);
        const auto second = shard(dir, 2, 2);
        write_tiny_gguf(first, 0, 2, 3, "one.weight", "same");
        write_tiny_gguf(second, 1, 2, 3, "two.weight", "same");
        std::string error;
        CHECK(!validate_qwen4exp_gguf(first.c_str(), error) &&
                  error.find("global tensor inventory") != std::string::npos,
              "declared aggregate tensor inventory is exact");
    }
    {
        TempDir dir;
        const auto first = shard(dir, 1, 2);
        const auto second = shard(dir, 2, 2);
        write_tiny_gguf(first, 0, 2, 2, "same.weight", "same");
        write_tiny_gguf(second, 1, 2, 2, "same.weight", "same");
        std::string error;
        CHECK(!validate_qwen4exp_gguf(first.c_str(), error) &&
                  error.find("duplicate or invalid tensor") != std::string::npos,
              "duplicate tensor names across shards fail closed");
    }
    {
        TempDir dir;
        const auto first = shard(dir, 1, 2);
        const auto second = shard(dir, 2, 2);
        write_tiny_gguf(first, 0, 2, 2, "one.weight", "left");
        write_tiny_gguf(second, 1, 2, 2, "two.weight", "right");
        std::string error;
        CHECK(!validate_qwen4exp_gguf(first.c_str(), error) &&
                  error.find("test.invariant") != std::string::npos,
              "invariant metadata must match exactly across shards");
    }
    {
        TempDir dir;
        const auto first = shard(dir, 1, 2);
        const auto second = shard(dir, 2, 2);
        write_tiny_gguf(first, 0, 2, 2, "one.weight", "same", false);
        write_tiny_gguf(second, 1, 2, 2, "two.weight", "same");
        std::string error;
        CHECK(!validate_qwen4exp_gguf(first.c_str(), error) &&
                  error.find("extends past end") != std::string::npos,
              "every shard tensor payload is checked against its own file");
    }

    std::printf("qwen4exp shards: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
