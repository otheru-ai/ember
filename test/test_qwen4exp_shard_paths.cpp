#include "qwen4exp_shards.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

using dflash::common::qwen4exp_discover_gguf_shards;

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
        char pattern[] = "/tmp/ember-qwen-shard-paths-XXXXXX";
        char * created = ::mkdtemp(pattern);
        if (!created) std::abort();
        path = created;
    }
    ~TempDir() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
};

std::filesystem::path shard(const TempDir & dir, int number, int count) {
    char name[80];
    std::snprintf(name, sizeof(name), "model-%05d-of-%05d.gguf", number, count);
    return dir.path / name;
}

void write_byte(const std::filesystem::path & path) {
    FILE * file = std::fopen(path.c_str(), "wb");
    if (!file || std::fputc(0, file) == EOF || std::fclose(file) != 0)
        std::abort();
}

} // namespace

int main() {
    {
        TempDir dir;
        const auto plain = dir.path / "model.gguf";
        write_byte(plain);
        std::vector<std::string> paths;
        std::string error;
        CHECK(qwen4exp_discover_gguf_shards(plain.c_str(), paths, error) &&
                  paths.size() == 1 && paths.front() == plain.string(),
              "an unsharded GGUF preserves one-file behavior");
    }
    {
        TempDir dir;
        const auto first = shard(dir, 1, 3);
        const auto second = shard(dir, 2, 3);
        const auto third = shard(dir, 3, 3);
        write_byte(first); write_byte(second); write_byte(third);
        std::vector<std::string> paths;
        std::string error;
        CHECK(qwen4exp_discover_gguf_shards(second.c_str(), paths, error) &&
                  paths.size() == 3 && paths[0] == first.string() &&
                  paths[1] == second.string() && paths[2] == third.string(),
              "any canonical shard resolves the exact ordered set");
    }
    {
        TempDir dir;
        const auto first = shard(dir, 1, 2);
        write_byte(first);
        std::vector<std::string> paths;
        std::string error;
        CHECK(!qwen4exp_discover_gguf_shards(first.c_str(), paths, error) &&
                  error.find("missing") != std::string::npos,
              "missing shards fail closed");
    }
    {
        TempDir dir;
        const auto first = shard(dir, 1, 2);
        const auto second = shard(dir, 2, 2);
        write_byte(first); write_byte(second); write_byte(shard(dir, 1, 3));
        std::vector<std::string> paths;
        std::string error;
        CHECK(!qwen4exp_discover_gguf_shards(second.c_str(), paths, error) &&
                  error.find("mismatched filename shard count") !=
                      std::string::npos,
              "conflicting filename shard counts fail closed");
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
              "inode-aliased shards fail closed");
    }

    std::printf("qwen4exp shard paths: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
