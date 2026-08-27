#include "qwen4exp_model.h"

#include <cstdio>
#include <cstring>

using dflash::common::ModelArchitecture;
using dflash::common::Qwen4ExpContract;
using dflash::common::model_architecture_from_name;
using dflash::common::model_architecture_name;
using dflash::common::qwen4exp_layer_has_ple;
using dflash::common::qwen4exp_layer_is_qsa;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(condition, message) do {                                      \
    if (condition) { ++g_pass; } else {                                     \
        ++g_fail; std::fprintf(stderr, "FAIL: %s\n", message);             \
    }                                                                        \
} while (0)

int main() {
    CHECK(model_architecture_from_name("deepseek4") ==
              ModelArchitecture::DEEPSEEK4,
          "DeepSeek architecture identification remains intact");
    CHECK(model_architecture_from_name("qwen4exp") ==
              ModelArchitecture::QWEN4_EXP,
          "canonical Qwen4Exp GGUF architecture is recognized");
    CHECK(model_architecture_from_name("qwen4_exp") ==
              ModelArchitecture::QWEN4_EXP,
          "upstream model_type spelling is recognized at the seam");
    CHECK(model_architecture_from_name("qwen3_next") ==
              ModelArchitecture::UNKNOWN,
          "Qwen4Exp cannot silently use a Qwen3-Next graph");
    CHECK(std::strcmp(model_architecture_name(ModelArchitecture::QWEN4_EXP),
                      "qwen4exp") == 0,
          "architecture diagnostics use the GGUF spelling");

    CHECK(Qwen4ExpContract::block_count == 48 &&
              Qwen4ExpContract::embedding_length == 2560 &&
              Qwen4ExpContract::vocab_size == 248320,
          "released model dimensions are pinned");
    CHECK(Qwen4ExpContract::expert_count == 512 &&
              Qwen4ExpContract::expert_used_count == 10,
          "released MoE routing dimensions are pinned");
    CHECK(Qwen4ExpContract::qsa_token_budget == 2048 &&
              Qwen4ExpContract::qsa_block_top_k == 512,
          "QSA selects 512 complete four-token blocks");
    for (uint32_t layer = 0; layer < Qwen4ExpContract::block_count; ++layer) {
        const bool expected = layer == 3 || layer == 7 || layer == 11 ||
                              layer == 15 || layer == 19 || layer == 23 ||
                              layer == 27 || layer == 31 || layer == 35 ||
                              layer == 39 || layer == 43 || layer == 47;
        CHECK(qwen4exp_layer_is_qsa(layer) == expected,
              "QSA layer schedule is exactly 3 GDN plus 1 QSA");
        CHECK(qwen4exp_layer_has_ple(layer) == (layer == 1),
              "PLE exists only on zero-based layer one");
    }
    CHECK(!qwen4exp_layer_is_qsa(Qwen4ExpContract::block_count),
          "out-of-range layers are never classified as QSA");

    std::printf("qwen4exp contract: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
