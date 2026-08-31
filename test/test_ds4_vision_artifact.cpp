#include "dflash/deepseek4/deepseek4_vision_artifact.h"
#include "dflash/deepseek4/deepseek4_vision_contract.h"

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>

using namespace dflash;

static int g_pass;
static int g_fail;

#define CHECK(cond, msg) do {                                              \
    if (cond) { ++g_pass; }                                                \
    else { ++g_fail; std::fprintf(stderr, "FAIL: %s\n", msg); }           \
} while (0)

static void append_u32(std::vector<uint8_t> & bytes, uint32_t value) {
    bytes.push_back(static_cast<uint8_t>(value));
    bytes.push_back(static_cast<uint8_t>(value >> 8));
    bytes.push_back(static_cast<uint8_t>(value >> 16));
    bytes.push_back(static_cast<uint8_t>(value >> 24));
}

static std::vector<uint8_t> artifact_bytes(int32_t h = 2, int32_t w = 3) {
    std::vector<uint8_t> bytes = {'D', 'S', '4', 'V', 'I', 'M', 'G', 0};
    append_u32(bytes, 2);
    append_u32(bytes, 2);
    append_u32(bytes, static_cast<uint32_t>(h));
    append_u32(bytes, static_cast<uint32_t>(w));
    append_u32(bytes, 0);
    append_u32(bytes, 0);
    const int values = h > 0 && w > 0 ? h * w * 2 : 0;
    for (int i = 0; i < values; ++i) {
        const float value = static_cast<float>(i) + 0.25f;
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        append_u32(bytes, bits);
    }
    return bytes;
}

static void test_valid_artifact() {
    Deepseek4VisionArtifact artifact;
    std::string error;
    const std::vector<uint8_t> bytes = artifact_bytes();
    CHECK(deepseek4_parse_vision_artifact(bytes, 2, artifact, error),
          "version-2 row-major aligner artifact parses");
    CHECK(artifact.n_embd == 2 && artifact.n_llm_h == 2 &&
          artifact.n_llm_w == 3 && artifact.image_embeddings.size() == 12,
          "required grid and embedding dimensions survive parsing");
    CHECK(artifact.n_llm_h >= 2 && artifact.n_llm_w >= 2,
          "ordering fixture exposes raster-vs-N-layout permutation bugs");
    CHECK(artifact.image_embeddings.front() == 0.25f &&
          artifact.image_embeddings.back() == 11.25f,
          "little-endian float payload is decoded exactly");
    CHECK(artifact.digest != 0, "active artifact receives a nonzero digest");

    Deepseek4ImageMarkers markers;
    markers.start = {-1.0f, -2.0f};
    markers.pad = {-3.0f, -4.0f};
    markers.newline = {-5.0f, -6.0f};
    markers.end = {-7.0f, -8.0f};
    Deepseek4PreparedImage prepared;
    CHECK(deepseek4_prepare_image(1000, artifact.n_llm_h,
                                  artifact.n_llm_w, 5, artifact.n_embd,
                                  artifact.image_embeddings, markers,
                                  prepared, &error),
          "parsed offline rows feed the shared learned-marker language seam");
    CHECK(prepared.block.image_perm ==
              std::vector<int32_t>({0, 3, 1, 4, 2, 5}),
          "artifact grid drives the exact official aligner-row permutation");

    Deepseek4VisionArtifact repeated;
    CHECK(deepseek4_parse_vision_artifact(bytes, 2, repeated, error) &&
          repeated.digest == artifact.digest,
          "semantic digest is stable for identical content");
    std::vector<uint8_t> changed = bytes;
    changed.back() ^= 1;
    CHECK(deepseek4_parse_vision_artifact(changed, 2, repeated, error) &&
          repeated.digest != artifact.digest,
          "semantic digest changes with aligner output");
}

static void test_version_and_grid_fail_closed() {
    Deepseek4VisionArtifact artifact;
    std::string error;
    std::vector<uint8_t> bytes = artifact_bytes();
    bytes[8] = 3;
    CHECK(!deepseek4_parse_vision_artifact(bytes, 2, artifact, error),
          "unknown artifact version is rejected");

    bytes = artifact_bytes();
    const uint8_t old_magic[8] = {'D', 'S', '4', 'I', 'M', 'G', 'E', '1'};
    std::memcpy(bytes.data(), old_magic, sizeof(old_magic));
    CHECK(!deepseek4_parse_vision_artifact(bytes, 2, artifact, error),
          "legacy DS4IMGE1 artifact is rejected rather than misread");
    CHECK(!deepseek4_parse_vision_artifact(artifact_bytes(0, 3), 2,
                                           artifact, error),
          "missing language-grid height is rejected rather than inferred");
    CHECK(!deepseek4_parse_vision_artifact(artifact_bytes(2, 0), 2,
                                           artifact, error),
          "missing language-grid width is rejected rather than inferred");
    CHECK(!deepseek4_parse_vision_artifact(artifact_bytes(), 4,
                                           artifact, error),
          "model embedding-width mismatch is rejected");
}

static void test_exact_length_and_header_contract() {
    Deepseek4VisionArtifact artifact;
    std::string error;
    std::vector<uint8_t> bytes = artifact_bytes();
    bytes.pop_back();
    CHECK(!deepseek4_parse_vision_artifact(bytes, 2, artifact, error),
          "truncated embedding payload is rejected");
    bytes = artifact_bytes();
    bytes.push_back(0);
    CHECK(!deepseek4_parse_vision_artifact(bytes, 2, artifact, error),
          "trailing payload bytes are rejected");
    bytes = artifact_bytes();
    bytes[24] = 1;
    CHECK(!deepseek4_parse_vision_artifact(bytes, 2, artifact, error),
          "unknown flags are rejected");
    bytes = artifact_bytes();
    bytes[28] = 1;
    CHECK(!deepseek4_parse_vision_artifact(bytes, 2, artifact, error),
          "nonzero reserved fields are rejected");
}

static void test_bounded_file_loader() {
    std::vector<uint8_t> bytes = artifact_bytes();
    char path[] = "/tmp/ember-ds4-vision-artifact-XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0, "artifact fixture temporary file opens");
    bool wrote = fd >= 0;
    size_t offset = 0;
    while (wrote && offset < bytes.size()) {
        const ssize_t n = write(fd, bytes.data() + offset,
                                bytes.size() - offset);
        if (n <= 0) {
            wrote = false;
        } else {
            offset += static_cast<size_t>(n);
        }
    }
    if (fd >= 0) close(fd);
    CHECK(wrote && offset == bytes.size(),
          "artifact fixture bytes reach the filesystem exactly");

    Deepseek4VisionArtifact artifact;
    std::string error;
    CHECK(wrote && deepseek4_load_vision_artifact(
                       path, 2, artifact, error),
          "bounded file loader delegates to the versioned parser");
    CHECK(artifact.n_llm_h == 2 && artifact.n_llm_w == 3,
          "file-loaded artifact retains its required grid");
    unlink(path);
    CHECK(!deepseek4_load_vision_artifact(path, 2, artifact, error),
          "missing artifact path fails closed");
}

int main() {
    test_valid_artifact();
    test_version_and_grid_fail_closed();
    test_exact_length_and_header_contract();
    test_bounded_file_loader();
    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail != 0;
}
