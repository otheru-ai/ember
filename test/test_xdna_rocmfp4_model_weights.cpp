#include "rocmfp4_model_weights.h"
#include "rocmfp4_pack.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

static int g_pass = 0;
static int g_fail = 0;
#define CHECK(condition, message) do { \
    if (condition) ++g_pass; \
    else { ++g_fail; std::printf("  FAIL: %s\n", message); } \
} while (0)

static void write_bytes(FILE * file, const void * data, size_t bytes) {
    if (std::fwrite(data, 1, bytes, file) != bytes) {
        std::perror("fwrite");
        std::abort();
    }
}

template <typename T>
static void write_value(FILE * file, T value) {
    write_bytes(file, &value, sizeof(value));
}

static void write_string(FILE * file, const char * value) {
    const uint64_t length = std::strlen(value);
    write_value(file, length);
    write_bytes(file, value, static_cast<size_t>(length));
}

static void pwrite_all(int fd, uint64_t offset, const std::vector<uint8_t> & data) {
    size_t complete = 0;
    while (complete < data.size()) {
        const ssize_t count = ::pwrite(
            fd, data.data() + complete, data.size() - complete,
            static_cast<off_t>(offset + complete));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            std::perror("pwrite");
            std::abort();
        }
        complete += static_cast<size_t>(count);
    }
}

static std::vector<uint8_t> projection_fixture(size_t bytes, uint8_t salt) {
    std::vector<uint8_t> result(bytes, 0);
    for (size_t block = 0;
         block < bytes / ember::xdna2::kRocmfp4BlockBytes; ++block) {
        uint8_t * q = result.data() +
            block * ember::xdna2::kRocmfp4BlockBytes;
        q[(block + salt) & 15u] = static_cast<uint8_t>(
            1u + ((block * 3u + salt) & 7u));
        q[16] = static_cast<uint8_t>(0x38u + (salt & 7u));
    }
    return result;
}

struct Fixture {
    std::string path;
    uint64_t data_offset = 0;
    uint64_t file_bytes = 0;
    int expert_id = 7;
    std::vector<uint8_t> gate;
    std::vector<uint8_t> up;
    std::vector<uint8_t> down;
};

static Fixture make_fixture() {
    char path[] = "/tmp/ember-xdna-model-XXXXXX";
    const int fd = ::mkstemp(path);
    if (fd < 0) {
        std::perror("mkstemp");
        std::abort();
    }
    FILE * file = ::fdopen(fd, "wb");
    if (!file) {
        std::perror("fdopen");
        std::abort();
    }
    write_bytes(file, "GGUF", 4);
    write_value<uint32_t>(file, 3);
    write_value<uint64_t>(file, 3);
    write_value<uint64_t>(file, 3);

    write_string(file, "general.architecture");
    write_value<uint32_t>(file, 8);  // GGUF_STRING
    write_string(file, "deepseek4-dflash-draft");
    write_string(file, "deepseek4-dflash-draft.block_count");
    write_value<uint32_t>(file, 4);  // GGUF_U32
    write_value<uint32_t>(file, 3);
    write_string(file, "deepseek4-dflash-draft.expert_count");
    write_value<uint32_t>(file, 4);
    write_value<uint32_t>(file, 256);

    const uint64_t projection =
        ember::xdna2::rocmfp4_projection_bytes(4096, 2048);
    const char * names[] = {
        "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_up_exps.weight",
        "blk.0.ffn_down_exps.weight",
    };
    for (int tensor = 0; tensor < 3; ++tensor) {
        write_string(file, names[tensor]);
        write_value<uint32_t>(file, 3);
        write_value<uint64_t>(file, tensor == 2 ? 2048 : 4096);
        write_value<uint64_t>(file, tensor == 2 ? 4096 : 2048);
        write_value<uint64_t>(file, 256);
        write_value<uint32_t>(file, ember::xdna2::kRocmfp4FastGgmlType);
        write_value<uint64_t>(file,
                              static_cast<uint64_t>(tensor) * projection * 256);
    }
    const long metadata_end = std::ftell(file);
    if (metadata_end < 0) std::abort();
    const uint64_t data_offset =
        (static_cast<uint64_t>(metadata_end) + 31u) / 32u * 32u;
    std::vector<uint8_t> padding(
        static_cast<size_t>(data_offset - static_cast<uint64_t>(metadata_end)));
    write_bytes(file, padding.data(), padding.size());
    if (std::fflush(file) != 0) std::abort();
    const uint64_t file_bytes = data_offset + 3u * projection * 256u;
    if (::ftruncate(fd, static_cast<off_t>(file_bytes)) != 0) {
        std::perror("ftruncate");
        std::abort();
    }

    Fixture fixture;
    fixture.path = path;
    fixture.data_offset = data_offset;
    fixture.file_bytes = file_bytes;
    fixture.gate = projection_fixture(static_cast<size_t>(projection), 1);
    fixture.up = projection_fixture(static_cast<size_t>(projection), 2);
    fixture.down = projection_fixture(static_cast<size_t>(projection), 3);
    const uint64_t expert_offset =
        static_cast<uint64_t>(fixture.expert_id) * projection;
    pwrite_all(fd, data_offset + expert_offset, fixture.gate);
    pwrite_all(fd, data_offset + projection * 256u + expert_offset, fixture.up);
    pwrite_all(fd, data_offset + projection * 512u + expert_offset, fixture.down);
    if (std::fclose(file) != 0) std::abort();
    return fixture;
}

int main() {
    std::printf("ember XDNA ROCMFP4 model-weight tests\n");
    Fixture fixture = make_fixture();
    std::vector<ember::xdna2::Rocmfp4ModelExpert> loaded;
    std::string error;
    CHECK(ember::xdna2::load_rocmfp4_model_experts(
              fixture.path.c_str(), 0, {fixture.expert_id}, loaded, &error),
          "valid sparse GGUF loads one trained expert slice");
    CHECK(loaded.size() == 1 && loaded[0].expert_id == fixture.expert_id,
          "loaded expert identity retained");
    CHECK(loaded.size() == 1 && loaded[0].gate == fixture.gate &&
              loaded[0].up == fixture.up && loaded[0].down == fixture.down,
          "raw projection slices are byte-exact");
    std::vector<uint8_t> expected_packed;
    CHECK(ember::xdna2::pack_rocmfp4_expert_v7(
              fixture.gate.data(), fixture.gate.size(), fixture.up.data(),
              fixture.up.size(), fixture.down.data(), fixture.down.size(),
              expected_packed, &error) && loaded.size() == 1 &&
              loaded[0].packed == expected_packed,
          "loader packing matches independent pack call");

    CHECK(!ember::xdna2::load_rocmfp4_model_experts(
              fixture.path.c_str(), 0, {256}, loaded, &error) &&
              error.find("[0,255]") != std::string::npos,
          "out-of-range expert rejected");
    CHECK(!ember::xdna2::load_rocmfp4_model_experts(
              fixture.path.c_str(), 0, {7, 7}, loaded, &error) &&
              error.find("unique") != std::string::npos,
          "duplicate expert rejected");
    CHECK(!ember::xdna2::load_rocmfp4_model_experts(
              fixture.path.c_str(), 1, {7}, loaded, &error) &&
              error.find("missing tensor") != std::string::npos,
          "missing layer tensor rejected");

    const int fd = ::open(fixture.path.c_str(), O_WRONLY);
    CHECK(fd >= 0, "fixture reopened for truncation");
    if (fd >= 0) {
        CHECK(::ftruncate(fd, static_cast<off_t>(fixture.file_bytes - 1)) == 0,
              "fixture truncation succeeds");
        ::close(fd);
        CHECK(!ember::xdna2::load_rocmfp4_model_experts(
                  fixture.path.c_str(), 0, {7}, loaded, &error) &&
                  error.find("beyond") != std::string::npos,
              "full tensor extent must fit inside GGUF");
    }
    ::unlink(fixture.path.c_str());
    std::printf("──────────────────────────────\n  %d passed, %d failed\n",
                g_pass, g_fail);
    return g_fail ? 1 : 0;
}
