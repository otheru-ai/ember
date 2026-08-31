#include "logits_probe_format.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { ++g_pass; } else { ++g_fail; std::fprintf(stderr, "FAIL: %s\n", msg); } \
} while (0)

static std::string read_file(const std::string &path) {
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return {};
    std::string result;
    char buffer[4096];
    for (;;) {
        const ssize_t count = read(fd, buffer, sizeof(buffer));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) break;
        result.append(buffer, static_cast<size_t>(count));
    }
    (void)close(fd);
    return result;
}

int main() {
    char root_template[] = "/tmp/ember-logits-format-XXXXXX";
    char *root = mkdtemp(root_template);
    CHECK(root != nullptr, "temporary directory");
    if (!root) return 1;

    const std::string abc_path = std::string(root) + "/abc";
    {
        const int fd = open(abc_path.c_str(), O_WRONLY | O_CREAT | O_EXCL |
                                              O_CLOEXEC, S_IRUSR | S_IWUSR);
        CHECK(fd >= 0, "create SHA fixture");
        if (fd >= 0) {
            CHECK(write(fd, "abc", 3U) == 3, "write SHA fixture");
            CHECK(close(fd) == 0, "close SHA fixture");
        }
    }
    std::string digest;
    std::string error;
    CHECK(ember_sha256_regular_file(abc_path, digest, error), "hash regular file");
    CHECK(digest == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "SHA-256 known vector");

    int32_t token = -1;
    CHECK(ember_parse_logits_probe_token("0", token, error) && token == 0,
          "parse zero token");
    CHECK(ember_parse_logits_probe_token("2147483647", token, error) &&
              token == std::numeric_limits<int32_t>::max(),
          "parse maximum token");
    CHECK(!ember_parse_logits_probe_token("-1", token, error),
          "reject negative token");
    CHECK(!ember_parse_logits_probe_token("12x", token, error),
          "reject token suffix");

    EmberLogitsProbeBundle bundle;
    bundle.model_path = "/models/control.gguf";
    bundle.model_sha256 = std::string(64U, '1');
    bundle.binary_sha256 = std::string(64U, '2');
    bundle.ember_revision = std::string(40U, '3');
    bundle.prefill_mode = "exact-q1";
    bundle.comparison_role = "authority";
    bundle.prefill_chunk = 1;
    bundle.token_ids = {17, 23};
    bundle.logits = {1.0F, -2.5F, 0.25F};
    const std::string output_dir = std::string(root) + "/bundle";
    CHECK(ember_write_logits_probe_bundle(output_dir, bundle, error),
          "write evidence bundle");
    struct stat status {};
    CHECK(stat((output_dir + "/logits.f32").c_str(), &status) == 0 &&
              status.st_size == 12,
          "payload has exact F32 width");
    const std::string payload = read_file(output_dir + "/logits.f32");
    const std::string expected_payload(
        "\x00\x00\x80\x3f\x00\x00\x20\xc0\x00\x00\x80\x3e", 12U);
    CHECK(payload == expected_payload, "payload is exact little-endian F32");
    const std::string manifest = read_file(output_dir + "/manifest.json");
    CHECK(manifest.find("\"token_ids\":[17,23]") != std::string::npos,
          "manifest binds tokens");
    CHECK(manifest.find("\"row_position\":1") != std::string::npos,
          "manifest binds final input position");
    CHECK(manifest.find("\"vocab_width\":3") != std::string::npos,
          "manifest binds vocabulary width");
    CHECK(manifest.find("\"placement\":\"monolithic-gpu\"") !=
              std::string::npos,
          "manifest binds monolithic GPU placement");
    CHECK(manifest.find("\"prefill\":\"exact-q1\"") != std::string::npos &&
              manifest.find("\"prefill_chunk\":1") != std::string::npos,
          "manifest binds prefill topology");
    CHECK(manifest.find("\"comparison_role\":\"authority\"") !=
              std::string::npos,
          "manifest labels comparison authority");
    CHECK(!ember_write_logits_probe_bundle(output_dir, bundle, error),
          "existing evidence directory fails closed");

    bundle.logits[0] = std::numeric_limits<float>::quiet_NaN();
    const std::string nan_dir = std::string(root) + "/nan";
    CHECK(!ember_write_logits_probe_bundle(nan_dir, bundle, error),
          "non-finite logits rejected");
    CHECK(stat(nan_dir.c_str(), &status) != 0,
          "invalid payload creates no directory");

    (void)unlink((output_dir + "/manifest.json").c_str());
    (void)unlink((output_dir + "/logits.f32").c_str());
    (void)rmdir(output_dir.c_str());
    (void)unlink(abc_path.c_str());
    (void)rmdir(root);
    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
