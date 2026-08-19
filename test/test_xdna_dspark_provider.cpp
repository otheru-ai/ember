#include "dspark_draft_compute_xdna.h"

#include <cmath>
#include <cstdio>
#include <string>

using namespace dflash::common;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; } \
    else { std::fprintf(stderr, "FAIL: %s\n", msg); g_fail++; } \
} while (0)

int main(int argc, char ** argv) {
    CHECK(argc == 2, "mock provider path supplied");
    if (argc != 2) return 1;

    XdnaDSparkDraftConfig config;
    config.plugin_path = argv[1];
    config.draft_model_path = "mock-draft.gguf";
    config.n_embd = 4;
    config.n_target_layers = 3;
    config.block_size = 5;
    config.n_swa = 8;
    config.head_dim = 2;
    const float mock_weight[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    ember_xdna_dspark_tensor_view_v1 weight_view{};
    weight_view.abi_version = EMBER_XDNA_DSPARK_PROVIDER_ABI_VERSION;
    weight_view.struct_size = sizeof(weight_view);
    weight_view.name = "mock.weight";
    weight_view.data = mock_weight;
    weight_view.bytes = sizeof(mock_weight);
    weight_view.type = 0;
    weight_view.n_dims = 1;
    weight_view.dims[0] = 4;
    weight_view.strides[0] = sizeof(float);
    config.weights_cpu_accessible = true;
    config.weight_views = &weight_view;
    config.weight_view_count = 1;

    std::string error;
    auto compute = make_xdna_dspark_draft_compute(config, &error);
    CHECK(compute != nullptr, "DSpark provider loads");
    CHECK(error.empty(), "provider load has no error");
    if (!compute) return 1;
    CHECK(compute->healthy(), "provider reports healthy");
    CHECK(std::string(compute->name()) == "mock-xdna-dspark",
          "provider name is exposed");
    CHECK(!compute->failure_is_fatal(), "provider fallback is optional");

    float noise_a[20];
    float noise_b[20];
    float features_b[12];
    float context_kv_a[12];
    for (int i = 0; i < 20; ++i) {
        noise_a[i] = (float)i;
        noise_b[i] = (float)(100 + i);
    }
    for (int i = 0; i < 12; ++i) features_b[i] = (float)(300 + i);
    for (int i = 0; i < 12; ++i) context_kv_a[i] = (float)(500 + i);

    XdnaDSparkDraftRequest request_a;
    request_a.committed = 7;
    request_a.ctx_len = 2;
    request_a.noise_embed = noise_a;
    request_a.context_kv = context_kv_a;
    XdnaDSparkDraftRequest request_b;
    request_b.committed = 11;
    request_b.ctx_len = 1;
    request_b.noise_embed = noise_b;
    request_b.ctx_features = features_b;
    auto job_a = compute->submit(request_a, &error);
    auto job_b = compute->submit(request_b, &error);
    CHECK(job_a != nullptr && job_b != nullptr,
          "two resident-session drafts can be in flight");

    // submit() owns its inputs: changing the caller buffers cannot affect an
    // outstanding proposal. Wait out of order to exercise session isolation.
    noise_a[0] = -999.0f;
    context_kv_a[11] = -999.0f;
    XdnaDSparkDraftOutput output_b;
    CHECK(job_b && job_b->wait(output_b, &error),
          "second session completes first");
    CHECK(output_b.hidden.size() == 20 &&
          std::fabs(output_b.hidden[0] - 422.0f) < 1e-6f,
          "second session result matches retained inputs");
    CHECK(output_b.confidence_hidden.size() == 20 &&
          std::fabs(output_b.confidence_hidden[0] - 422.5f) < 1e-6f,
          "confidence state is returned separately");

    XdnaDSparkDraftOutput output_a;
    CHECK(job_a && job_a->wait(output_a, &error),
          "first session completes after second");
    CHECK(output_a.hidden.size() == 20 &&
          std::fabs(output_a.hidden[0] - 518.0f) < 1e-6f,
          "preprojected context-KV lifetime is independent");
    CHECK(job_a && !job_a->wait(output_a, &error),
          "a completed proposal cannot be consumed twice");

    XdnaDSparkDraftRequest invalid = request_a;
    invalid.ctx_len = 9;
    CHECK(!compute->submit(invalid, &error),
          "feature windows beyond n_swa fail closed");

    invalid = request_a;
    invalid.ctx_features = nullptr;
    invalid.main_context = nullptr;
    invalid.context_kv = nullptr;
    CHECK(!compute->submit(invalid, &error),
          "context requests require raw or preprojected input");

    XdnaDSparkDraftConfig bad = config;
    bad.plugin_path += ".missing";
    error.clear();
    CHECK(!make_xdna_dspark_draft_compute(bad, &error),
          "missing provider is rejected");
    CHECK(!error.empty(), "missing provider reports an error");

    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
