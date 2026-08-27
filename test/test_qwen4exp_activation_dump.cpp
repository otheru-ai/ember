#include "qwen4exp_activation_dump.h"

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
    std::string pattern = "/tmp/ember-qwen-act-XXXXXX";
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
        static_cast<uint64_t>(kQwen4ExpActivationFloats * sizeof(float));
    std::vector<float> first(kQwen4ExpActivationFloats);
    std::vector<float> second(kQwen4ExpActivationFloats);
    for (size_t i = 0; i < first.size(); ++i) {
        first[i] = static_cast<float>(i % 997U) / 997.0f;
        second[i] = -first[i];
    }
    Qwen4ExpActivationDumpResult result;
    std::string error;
    CHECK(qwen4exp_append_activation_dump(path, first, result, error),
          "first complete record appended");
    CHECK(result.ordinal == 0 && result.byte_offset == 0,
          "first record position reported");
    CHECK(file_size(path) == record_bytes, "first record has exact size");
    CHECK(qwen4exp_append_activation_dump(path, second, result, error),
          "second complete record appended");
    CHECK(result.ordinal == 1 && result.byte_offset == record_bytes,
          "second record position reported");
    CHECK(file_size(path) == 2 * record_bytes, "two records have exact size");

    std::ifstream input(path, std::ios::binary);
    std::vector<float> roundtrip(2 * kQwen4ExpActivationFloats);
    input.read(reinterpret_cast<char *>(roundtrip.data()),
               static_cast<std::streamsize>(roundtrip.size() * sizeof(float)));
    CHECK(input.good(), "raw F32 records are readable");
    CHECK(roundtrip.front() == first.front() &&
          roundtrip[kQwen4ExpActivationFloats + 1] == second[1],
          "raw record order and values preserved");

    std::vector<float> wrong(7, 0.0f);
    CHECK(!qwen4exp_append_activation_dump(path, wrong, result, error) &&
          error.find("48x2560") != std::string::npos,
          "wrong activation shape rejected");
    CHECK(file_size(path) == 2 * record_bytes,
          "shape rejection does not alter output");
    first[17] = std::numeric_limits<float>::quiet_NaN();
    CHECK(!qwen4exp_append_activation_dump(path, first, result, error) &&
          error.find("non-finite") != std::string::npos,
          "non-finite activation rejected");

    const std::string partial = root + "/partial.f32";
    { std::ofstream out(partial, std::ios::binary); out.put('x'); }
    first[17] = 0.0f;
    CHECK(!qwen4exp_append_activation_dump(partial, first, result, error) &&
          error.find("partial record") != std::string::npos,
          "pre-existing partial capture rejected");
    CHECK(file_size(partial) == 1, "partial capture remains untouched");
    CHECK(!qwen4exp_append_activation_dump("relative.f32", first, result, error),
          "relative dump path rejected");

    (void)::unlink((path + ".lock").c_str());
    (void)::unlink(path.c_str());
    (void)::unlink((partial + ".lock").c_str());
    (void)::unlink(partial.c_str());
    (void)::rmdir(root.c_str());
    std::printf("qwen4exp_activation_dump: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
