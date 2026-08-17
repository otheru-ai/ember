#include "q8_model_weights.h"
#include "q8_0_pack.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
    if (bytes && std::fwrite(data, 1, bytes, file) != bytes) std::abort();
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

static void pwrite_all(int fd, uint64_t offset,
                       const std::vector<uint8_t> & data) {
    size_t complete = 0;
    while (complete < data.size()) {
        const ssize_t count = ::pwrite(
            fd, data.data() + complete, data.size() - complete,
            static_cast<off_t>(offset + complete));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) std::abort();
        complete += static_cast<size_t>(count);
    }
}

static std::vector<uint8_t> projection_fixture(size_t bytes, uint8_t salt) {
    std::vector<uint8_t> result(bytes, 0);
    for (size_t block = 0; block < bytes / ember::xdna2::kQ8BlockBytes;
         ++block) {
        uint8_t * q = result.data() + block * ember::xdna2::kQ8BlockBytes;
        const uint16_t fp16_one = 0x3c00u;
        std::memcpy(q, &fp16_one, sizeof(fp16_one));
        q[2 + ((block + salt) % ember::xdna2::kQ8BlockWeights)] =
            static_cast<uint8_t>(static_cast<int8_t>(
                static_cast<int>(salt) - 4));
    }
    return result;
}

struct Fixture {
    std::string path;
    uint64_t data_offset = 0;
    uint64_t file_bytes = 0;
    std::vector<uint8_t> gate;
    std::vector<uint8_t> up;
    std::vector<uint8_t> down;
};

static Fixture make_fixture() {
    char path[] = "/tmp/ember-xdna-q8-model-XXXXXX";
    const int fd = ::mkstemp(path);
    if (fd < 0) std::abort();
    FILE * file = ::fdopen(fd, "wb");
    if (!file) std::abort();
    write_bytes(file, "GGUF", 4);
    write_value<uint32_t>(file, 3);
    write_value<uint64_t>(file, 3);
    write_value<uint64_t>(file, 2);

    write_string(file, "general.architecture");
    write_value<uint32_t>(file, 8);  // GGUF_STRING
    write_string(file, "deepseek4-dflash-draft");
    write_string(file, "deepseek4-dflash-draft.block_count");
    write_value<uint32_t>(file, 4);  // GGUF_U32
    write_value<uint32_t>(file, 3);

    const uint64_t projection =
        ember::xdna2::q8_projection_bytes(4096, 2048);
    const char * names[] = {
        "blk.0.ffn_gate_shexp.weight",
        "blk.0.ffn_up_shexp.weight",
        "blk.0.ffn_down_shexp.weight",
    };
    for (int tensor = 0; tensor < 3; ++tensor) {
        write_string(file, names[tensor]);
        write_value<uint32_t>(file, 2);
        write_value<uint64_t>(file, tensor == 2 ? 2048 : 4096);
        write_value<uint64_t>(file, tensor == 2 ? 4096 : 2048);
        write_value<uint32_t>(file, ember::xdna2::kQ8GgmlType);
        write_value<uint64_t>(file,
                              static_cast<uint64_t>(tensor) * projection);
    }
    const long metadata_end = std::ftell(file);
    if (metadata_end < 0) std::abort();
    const uint64_t data_offset =
        (static_cast<uint64_t>(metadata_end) + 31u) / 32u * 32u;
    std::vector<uint8_t> padding(
        static_cast<size_t>(data_offset - static_cast<uint64_t>(metadata_end)));
    write_bytes(file, padding.data(), padding.size());
    if (std::fflush(file) != 0) std::abort();
    const uint64_t file_bytes = data_offset + 3u * projection;
    if (::ftruncate(fd, static_cast<off_t>(file_bytes)) != 0) std::abort();

    Fixture fixture;
    fixture.path = path;
    fixture.data_offset = data_offset;
    fixture.file_bytes = file_bytes;
    fixture.gate = projection_fixture(static_cast<size_t>(projection), 1);
    fixture.up = projection_fixture(static_cast<size_t>(projection), 2);
    fixture.down = projection_fixture(static_cast<size_t>(projection), 3);
    pwrite_all(fd, data_offset, fixture.gate);
    pwrite_all(fd, data_offset + projection, fixture.up);
    pwrite_all(fd, data_offset + 2u * projection, fixture.down);
    if (std::fclose(file) != 0) std::abort();
    return fixture;
}

int main() {
    std::printf("ember XDNA Q8 model-weight tests\n");
    Fixture fixture = make_fixture();
    ember::xdna2::Q8ModelSharedExpert loaded;
    std::string error;
    CHECK(ember::xdna2::load_q8_model_shared_expert(
              fixture.path.c_str(), 0, loaded, &error),
          "valid sparse GGUF loads trained shared expert");
    CHECK(loaded.layer == 0, "loaded layer identity retained");
    CHECK(loaded.gate == fixture.gate && loaded.up == fixture.up &&
              loaded.down == fixture.down,
          "raw projection tensors are byte-exact");
    std::vector<uint8_t> expected;
    CHECK(ember::xdna2::pack_q8_expert_v2(
              fixture.gate.data(), fixture.gate.size(), fixture.up.data(),
              fixture.up.size(), fixture.down.data(), fixture.down.size(),
              expected, &error) && loaded.packed == expected,
          "loader packing matches independent pack call");

    CHECK(!ember::xdna2::load_q8_model_shared_expert(
              fixture.path.c_str(), 3, loaded, &error) &&
              error.find("[0,2]") != std::string::npos,
          "out-of-range layer rejected");
    CHECK(!ember::xdna2::load_q8_model_shared_expert(
              fixture.path.c_str(), 1, loaded, &error) &&
              error.find("missing tensor") != std::string::npos,
          "missing layer tensor rejected");

    const int fd = ::open(fixture.path.c_str(), O_WRONLY);
    CHECK(fd >= 0, "fixture reopened for truncation");
    if (fd >= 0) {
        CHECK(::ftruncate(fd, static_cast<off_t>(fixture.file_bytes - 1)) == 0,
              "fixture truncation succeeds");
        ::close(fd);
        CHECK(!ember::xdna2::load_q8_model_shared_expert(
                  fixture.path.c_str(), 0, loaded, &error) &&
                  error.find("beyond") != std::string::npos,
              "full tensor extent must fit inside GGUF");
    }
    ::unlink(fixture.path.c_str());
    std::printf("──────────────────────────────\n  %d passed, %d failed\n",
                g_pass, g_fail);
    return g_fail ? 1 : 0;
}
