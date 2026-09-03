#include "../src/model/model_profile.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    CHECK(!ember_prompt_profile_from_arch("llama", &profile),
          "unknown architectures fail closed");
    CHECK(!ember_prompt_profile_from_arch(NULL, &profile),
          "missing architecture fails closed");

    // ember_prompt_profile_detect owns the GGUF read. Its failure paths matter
    // more than its success path: a server that cannot read the header must
    // refuse to start rather than assume DeepSeek and silently render the wrong
    // special tokens. Each of these must also produce a diagnostic naming the
    // cause, because the operator sees only that string.
    char *err = (char *)0x1;
    CHECK(!ember_prompt_profile_detect(NULL, &profile, &err) && err != NULL &&
              strstr(err, "invalid") != NULL,
          "a NULL path is rejected with a diagnostic");
    free(err);

    err = NULL;
    CHECK(!ember_prompt_profile_detect("/nonexistent/ember-test.gguf",
                                       &profile, &err) &&
              err != NULL && strstr(err, "cannot read GGUF metadata") != NULL &&
              strstr(err, "/nonexistent/ember-test.gguf") != NULL,
          "an unreadable file reports the path it could not read");
    free(err);

    err = NULL;
    CHECK(!ember_prompt_profile_detect("/nonexistent/ember-test.gguf",
                                       NULL, &err) && err != NULL,
          "a NULL out-parameter is rejected before opening anything");
    free(err);

    // The error pointer is optional; passing none must not crash.
    CHECK(!ember_prompt_profile_detect("/nonexistent/ember-test.gguf",
                                       &profile, NULL),
          "detection without an error sink still fails closed");
    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
