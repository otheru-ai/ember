// Evidence bundle contract for the default-off full-logit probe.
//
// The comparison is only meaningful when the raw vocabulary row is bound to
// the exact model, token sequence, executable and source revision that made
// it.  Keep that bookkeeping independent of the GPU runtime so it remains
// unit-testable on the ordinary host build.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct EmberLogitsProbeBundle {
    std::string model_path;
    std::string model_sha256;
    std::string binary_sha256;
    std::string ember_revision;
    std::string prefill_mode;
    std::string comparison_role;
    int prefill_chunk = 0;
    std::vector<int32_t> token_ids;
    std::vector<float> logits;
};

struct EmberLayerCaptureBundle {
    EmberLogitsProbeBundle logits;
    std::string checkpoint_name;
    int checkpoint_layer = -1;
    int checkpoint_width = 0;
    std::vector<float> checkpoint;
    std::string retained_logits_sha256;
    bool capture_logits_identical = false;
};

bool ember_parse_logits_probe_token(const char *text, int32_t &token,
                                    std::string &error);

bool ember_sha256_regular_file(const std::string &path, std::string &digest,
                               std::string &error);

bool ember_logits_payload_sha256(const std::vector<float> &logits,
                                 std::string &digest,
                                 std::string &error);

bool ember_write_logits_probe_bundle(const std::string &directory,
                                     const EmberLogitsProbeBundle &bundle,
                                     std::string &error);

bool ember_write_layer_capture_bundle(const std::string &directory,
                                      const EmberLayerCaptureBundle &bundle,
                                      std::string &error);
