#include "logits_probe_format.h"
#include "deepseek4_model_contract.h"

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
    CHECK(dflash::common::deepseek4_layer_contract_supported(43U, 3U, false),
          "production layer contract accepted");
    CHECK(!dflash::common::deepseek4_layer_contract_supported(1U, 1U, false),
          "single-layer control rejected by default");
    CHECK(dflash::common::deepseek4_layer_contract_supported(1U, 1U, true),
          "single-layer control accepted when explicit");
    CHECK(!dflash::common::deepseek4_layer_contract_supported(1U, 3U, true) &&
              !dflash::common::deepseek4_layer_contract_supported(43U, 1U, true) &&
              !dflash::common::deepseek4_layer_contract_supported(2U, 1U, true),
          "control flag accepts no other layer contract");

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

    std::string retained_digest;
    CHECK(ember_logits_payload_sha256(
              bundle.logits, retained_digest, error),
          "hash in-memory logit payload");
    CHECK(retained_digest ==
              "6be7a0225e3fc2f6ad1f1f1827362d7a7d849414c3f116b6d7667f4ff671083e",
          "in-memory payload digest is stable");

    EmberLayerCaptureBundle capture;
    capture.logits = bundle;
    capture.logits.comparison_role = "first-boundary-diagnostic";
    capture.checkpoint_name = "post_layer_0_mean_hc";
    capture.checkpoint_layer = 0;
    capture.checkpoint_width = 3;
    capture.checkpoint = {0.5F, -0.25F, 2.0F};
    capture.retained_logits_sha256 = retained_digest;
    capture.capture_logits_identical = true;
    const std::string capture_dir = std::string(root) + "/capture";
    CHECK(ember_write_layer_capture_bundle(capture_dir, capture, error),
          "write paired layer capture bundle");
    CHECK(stat((capture_dir + "/logits.f32").c_str(), &status) == 0 &&
              status.st_size == 12,
          "capture retains exact logit payload width");
    CHECK(stat((capture_dir + "/layer0-mean-hc.f32").c_str(), &status) == 0 &&
              status.st_size == 12,
          "capture checkpoint has exact F32 width");
    const std::string capture_manifest =
        read_file(capture_dir + "/manifest.json");
    CHECK(capture_manifest.find(
              "\"schema\":\"ember-ds4-layer-capture-v1\"") !=
              std::string::npos,
          "capture manifest names schema");
    CHECK(capture_manifest.find(
              "\"retained_authority_payload_sha256\":\"" +
              retained_digest + "\"") != std::string::npos,
          "capture manifest binds retained authority");
    CHECK(capture_manifest.find(
              "\"capture_logits_identical\":true") !=
              std::string::npos,
          "capture manifest records non-perturbation proof");
    CHECK(capture_manifest.find("\"checkpoint_layer\":0") !=
              std::string::npos &&
              capture_manifest.find("\"checkpoint_width\":3") !=
              std::string::npos,
          "capture manifest binds checkpoint shape");
    CHECK(!ember_write_layer_capture_bundle(
              capture_dir, capture, error),
          "existing capture directory fails closed");

    capture.capture_logits_identical = false;
    const std::string perturbed_dir = std::string(root) + "/perturbed";
    CHECK(!ember_write_layer_capture_bundle(
              perturbed_dir, capture, error),
          "capture without non-perturbation proof rejected");
    CHECK(stat(perturbed_dir.c_str(), &status) != 0,
          "rejected capture creates no directory");
    capture.capture_logits_identical = true;
    capture.retained_logits_sha256 = std::string(64U, 'f');
    const std::string wrong_baseline_dir =
        std::string(root) + "/wrong-baseline";
    CHECK(!ember_write_layer_capture_bundle(
              wrong_baseline_dir, capture, error),
          "capture differing from retained logits rejected");
    CHECK(stat(wrong_baseline_dir.c_str(), &status) != 0,
          "wrong-baseline capture creates no directory");

    bundle.logits[0] = std::numeric_limits<float>::quiet_NaN();
    const std::string nan_dir = std::string(root) + "/nan";
    CHECK(!ember_write_logits_probe_bundle(nan_dir, bundle, error),
          "non-finite logits rejected");
    CHECK(stat(nan_dir.c_str(), &status) != 0,
          "invalid payload creates no directory");

    (void)unlink((output_dir + "/manifest.json").c_str());
    (void)unlink((output_dir + "/logits.f32").c_str());
    (void)rmdir(output_dir.c_str());
    (void)unlink((capture_dir + "/manifest.json").c_str());
    (void)unlink((capture_dir + "/layer0-mean-hc.f32").c_str());
    (void)unlink((capture_dir + "/logits.f32").c_str());
    (void)rmdir(capture_dir.c_str());
    (void)unlink(abc_path.c_str());
    (void)rmdir(root);
    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
