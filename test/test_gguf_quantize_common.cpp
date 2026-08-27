#include "gguf_quantize_common.h"

#include <cstdio>
#include <string>
#include <vector>

namespace quant = ember::gguf_quantize;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(condition, message) do {                                      \
    if (condition) { ++g_pass; } else {                                     \
        ++g_fail; std::fprintf(stderr, "FAIL: %s\n", message);             \
    }                                                                        \
} while (0)

static bool parse(const std::vector<const char *> & arguments,
                  quant::Options & options, std::string & error) {
    return quant::parse_options(static_cast<int>(arguments.size()),
                                arguments.data(), options, error);
}

static void test_cli() {
    quant::Options options;
    std::string error;
    CHECK(parse({"tool", "--build-info-json"}, options, error) &&
              options.build_info_json,
          "build-info mode is a standalone invocation");
    CHECK(parse({"tool", "--tensor-type", "^per_layer_token_embd\\.weight$=Q4_0_ROCMI4",
                 "--keep-split", "--device-budget-bytes", "137438953472",
                 "--runtime-reserve-bytes", "34359738368",
                 "--intervention-manifest", "heretic.json",
                 "in.gguf", "out.gguf", "Q4_0_ROCMI4", "12"},
                options, error) &&
              options.keep_split && options.threads == 12 &&
              options.tensor_type_overrides.size() == 1 &&
              options.tensor_type_overrides[0].format == quant::TensorFormat::rocmi4 &&
              options.budget_configured &&
              options.device_budget_bytes == 137438953472ULL &&
              options.runtime_reserve_bytes == 34359738368ULL &&
              options.intervention_manifest_path == "heretic.json",
          "release-agent option order and positional contract parse exactly");
    CHECK(!parse({"tool", "--tensor-type", "x=Q4_0_ROCMI4",
                  "--intervention-manifest", "a.json",
                  "--intervention-manifest", "b.json",
                  "in.gguf", "out.gguf", "Q4_0_ROCMI4", "4"},
                 options, error) && error.find("only once") != std::string::npos,
          "an intervention manifest cannot be ambiguously repeated");
    CHECK(!parse({"tool", "--tensor-type", "x=Q4_0_ROCMI4", "in.gguf",
                  "--keep-split", "out.gguf", "Q4_0_ROCMI4", "4"},
                 options, error) &&
              error.find("precede") != std::string::npos,
          "options after positionals are rejected rather than ignored");
    CHECK(parse({"tool", "--tensor-type", "head=Q6_K", "--tensor-type",
                 "matrix=Q4_0_ROCMFP4_FAST", "in.gguf", "out.gguf",
                 "Q4_0_ROCMI4", "4"}, options, error) &&
              options.tensor_type_overrides.size() == 2 &&
              options.tensor_type_overrides[0].format == quant::TensorFormat::q6_k &&
              options.tensor_type_overrides[1].format == quant::TensorFormat::rocmfp4_fast,
          "safe mixed-quant per-tensor formats parse explicitly");
    CHECK(!parse({"tool", "--tensor-type", "x=Q5_K", "in.gguf", "out.gguf",
                  "Q4_0_ROCMI4", "4"}, options, error),
          "tensor overrides reject formats outside the kernel-backed bakeoff set");
    CHECK(!parse({"tool", "in.gguf", "out.gguf", "Q4_0_ROCMI4", "0"},
                 options, error),
          "a release quantization requires an explicit PLE override and threads");
    CHECK(!parse({"tool", "--tensor-type", "x=Q4_0_ROCMI4", "--dry-size-json",
                  "in.gguf", "out.gguf", "Q4_0_ROCMI4", "4"}, options, error) &&
              error.find("requires explicit") != std::string::npos,
          "dry size accounting has no implicit device budget or reserve");
}

static void test_policy() {
    CHECK(quant::tensor_allows_quantization("blk.0.attn_q.weight", 2),
          "ordinary rank-two projection weights are quantized");
    CHECK(quant::tensor_allows_quantization("blk.0.ffn_up_exps.weight", 3),
          "rank-three expert weights are quantized");
    CHECK(!quant::tensor_allows_quantization("blk.0.attn_norm.weight", 2),
          "norm weights remain unquantized");
    CHECK(!quant::tensor_allows_quantization("output.weight", 2),
          "the output projection remains unquantized");
    CHECK(!quant::tensor_allows_quantization("blk.0.ffn_gate_inp.weight", 2),
          "router weights remain unquantized");
    CHECK(!quant::tensor_allows_quantization("blk.0.ssm_conv1d.weight", 2),
          "convolution weights remain unquantized");
    CHECK(!quant::tensor_allows_quantization("v.position_embd.weight", 2) &&
              !quant::tensor_allows_quantization("v.patch_embd.weight", 2),
          "position and patch weights remain unquantized");
    CHECK(!quant::tensor_allows_quantization("blk.0.ssm_a", 2) &&
              !quant::tensor_allows_quantization("scale.weight", 1),
          "non-weights and rank-one weights remain unquantized");
}

static void test_build_info() {
    const std::string revision(40, 'a');
    CHECK(quant::build_info_json(revision) ==
              "{\"tool\":\"ember-gguf-quantize\","
              "\"ember_revision\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\","
              "\"rocmfpx_revision\":\"c49ebdbd5c9f01ec242369f9e7f7967855f80cba\","
              "\"format\":\"Q4_0_ROCMI4\",\"ggml_tensor_type\":108,"
              "\"per_tensor_formats\":[\"Q4_0_ROCMI4\",\"Q6_K\",\"Q4_0_ROCMFP4_FAST\"],"
              "\"intervention_manifest_schema\":1}",
          "build-info JSON is one exact provenance object");
    quant::SizeReport fits;
    fits.artifact_bytes = 100ULL * 1073741824ULL;
    fits.shard_bytes = {40ULL * 1073741824ULL, 60ULL * 1073741824ULL};
    fits.runtime_reserve_bytes = 32ULL * 1073741824ULL;
    fits.budget_bytes = 128ULL * 1073741824ULL;
    fits.fits = false;
    CHECK(quant::size_report_json(fits) ==
              "{\"artifact_bytes\":107374182400,\"artifact_gib\":100.000000,"
              "\"shard_count\":2,\"shard_bytes\":[42949672960,64424509440],"
              "\"runtime_reserve_bytes\":34359738368,\"runtime_reserve_gib\":32.000000,"
              "\"budget_bytes\":137438953472,\"budget_gib\":128.000000,"
              "\"total_bytes\":141733920768,\"total_gib\":132.000000,"
              "\"headroom_bytes\":-4294967296,\"headroom_gib\":-4.000000,\"fits\":false}",
          "size JSON reports exact bytes, GiB, negative headroom, and fit decision");
}

int main() {
    test_cli();
    test_policy();
    test_build_info();
    std::printf("gguf quantize common: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
