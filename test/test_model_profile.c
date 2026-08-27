#include "../src/model/model_profile.h"

#include <stdio.h>

static int g_pass;
static int g_fail;

#define CHECK(cond, msg) do { \
    if (cond) ++g_pass; \
    else { ++g_fail; fprintf(stderr, "FAIL: %s\n", msg); } \
} while (0)

int main(void) {
    ember_prompt_profile profile = EMBER_PROMPT_DEEPSEEK_DSML;
    CHECK(ember_prompt_profile_from_arch("deepseek4", &profile) &&
              profile == EMBER_PROMPT_DEEPSEEK_DSML,
          "deepseek4 selects DSML");
    CHECK(ember_prompt_profile_from_arch("qwen4exp", &profile) &&
              profile == EMBER_PROMPT_QWEN4_CHATML,
          "canonical qwen4exp selects ChatML");
    CHECK(ember_prompt_profile_from_arch("qwen4_exp", &profile) &&
              profile == EMBER_PROMPT_QWEN4_CHATML,
          "Transformers qwen4_exp spelling selects ChatML");
    CHECK(!ember_prompt_profile_from_arch("qwen35", &profile),
          "unknown architectures fail closed");
    CHECK(!ember_prompt_profile_from_arch(NULL, &profile),
          "missing architecture fails closed");
    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
