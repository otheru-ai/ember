#include "activation_dump.h"

// A second architecture's record geometry, so the writer is exercised with two
// different shapes and cannot silently accept a mismatched one. The writer
// itself is architecture-neutral; only the geometry differs.
static constexpr size_t kOtherActivationLayers = 48;
static constexpr size_t kOtherActivationWidth = 2560;
static constexpr size_t kOtherActivationFloats =
    kOtherActivationLayers * kOtherActivationWidth;

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace dflash::common;

static int g_pass = 0;
static int g_fail = 0;
#define CHECK(cond, msg) do { if (cond) ++g_pass; else { ++g_fail; std::fprintf(stderr, "FAIL: %s\n", msg); } } while (0)

static std::string temporary_directory() {
    std::string pattern = "/tmp/ember-act-XXXXXX";
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char * result = ::mkdtemp(writable.data());
    return result ? std::string(result) : std::string();
}

static uint64_t file_size(const std::string & path) {
    struct stat status {};
    return ::stat(path.c_str(), &status) == 0
        ? static_cast<uint64_t>(status.st_size) : UINT64_MAX;
}

int main() {
    const std::string root = temporary_directory();
    CHECK(!root.empty(), "temporary directory created");
    const std::string path = root + "/harmful.f32";
    const uint64_t record_bytes =
        static_cast<uint64_t>(kOtherActivationFloats * sizeof(float));
    std::vector<float> first(kOtherActivationFloats);
    std::vector<float> second(kOtherActivationFloats);
    for (size_t i = 0; i < first.size(); ++i) {
        first[i] = static_cast<float>(i % 997U) / 997.0f;
        second[i] = -first[i];
    }
    ActivationDumpResult result;
    std::string error;
    CHECK(append_activation_dump(path, first, kOtherActivationFloats, "OtherArch", result, error),
          "first complete record appended");
    CHECK(result.ordinal == 0 && result.byte_offset == 0,
          "first record position reported");
    CHECK(file_size(path) == record_bytes, "first record has exact size");
    CHECK(append_activation_dump(path, second, kOtherActivationFloats, "OtherArch", result, error),
          "second complete record appended");
    CHECK(result.ordinal == 1 && result.byte_offset == record_bytes,
          "second record position reported");
    CHECK(file_size(path) == 2 * record_bytes, "two records have exact size");

    std::ifstream input(path, std::ios::binary);
    std::vector<float> roundtrip(2 * kOtherActivationFloats);
    input.read(reinterpret_cast<char *>(roundtrip.data()),
               static_cast<std::streamsize>(roundtrip.size() * sizeof(float)));
    CHECK(input.good(), "raw F32 records are readable");
    CHECK(roundtrip.front() == first.front() &&
          roundtrip[kOtherActivationFloats + 1] == second[1],
          "raw record order and values preserved");

    std::vector<float> wrong(7, 0.0f);
    CHECK(!append_activation_dump(path, wrong, kOtherActivationFloats, "OtherArch", result, error) &&
          error.find(std::to_string(kOtherActivationFloats)) !=
              std::string::npos &&
          error.find("got 7") != std::string::npos,
          "wrong activation shape rejected");
    CHECK(file_size(path) == 2 * record_bytes,
          "shape rejection does not alter output");
    first[17] = std::numeric_limits<float>::quiet_NaN();
    CHECK(!append_activation_dump(path, first, kOtherActivationFloats, "OtherArch", result, error) &&
          error.find("non-finite") != std::string::npos,
          "non-finite activation rejected");

    const std::string partial = root + "/partial.f32";
    { std::ofstream out(partial, std::ios::binary); out.put('x'); }
    first[17] = 0.0f;
    CHECK(!append_activation_dump(partial, first, kOtherActivationFloats, "OtherArch", result, error) &&
          error.find("whole number of") != std::string::npos,
          "pre-existing partial capture rejected");
    CHECK(file_size(partial) == 1, "partial capture remains untouched");
    CHECK(!append_activation_dump("relative.f32", first, kOtherActivationFloats, "OtherArch", result, error),
          "relative dump path rejected");

    // The DeepSeek-V4 geometry, which is why the record size is a parameter:
    // 43 captured layers x 4096, not the other arch's 48 x 2560.
    const size_t ds4_floats = 43u * 4096u;
    const uint64_t ds4_bytes = static_cast<uint64_t>(ds4_floats * sizeof(float));
    const std::string ds4_path = root + "/ds4.f32";
    std::vector<float> ds4(ds4_floats, 0.25f);
    CHECK(append_activation_dump(ds4_path, ds4, ds4_floats, "DeepSeek4",
                                 result, error),
          "DeepSeek4-shaped record appends");
    CHECK(file_size(ds4_path) == ds4_bytes, "DeepSeek4 record has exact size");

    // A file holds records of ONE size. The format is headerless, so appending
    // a differently shaped record would silently corrupt every later slice.
    CHECK(!append_activation_dump(ds4_path, first, kOtherActivationFloats,
                                  "OtherArch", result, error) &&
          !error.empty(),
          "mixing record geometries in one file is refused");
    CHECK(file_size(ds4_path) == ds4_bytes,
          "refused geometry change leaves the dump byte-identical");

    // A zero-size record geometry is a caller bug, not an empty write.
    CHECK(!append_activation_dump(ds4_path, ds4, 0, "DeepSeek4", result, error),
          "zero-length record geometry is refused");

    (void)::unlink((ds4_path + ".lock").c_str());
    (void)::unlink(ds4_path.c_str());

    (void)::unlink((path + ".lock").c_str());
    (void)::unlink(path.c_str());
    (void)::unlink((partial + ".lock").c_str());
    (void)::unlink(partial.c_str());
    (void)::rmdir(root.c_str());
    std::printf("activation_dump: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
