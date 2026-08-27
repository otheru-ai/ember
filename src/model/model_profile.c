#include "model_profile.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gguf.h"

bool ember_prompt_profile_from_arch(const char *architecture,
                                    ember_prompt_profile *out) {
    if (!architecture || !out) return false;
    if (strcmp(architecture, "deepseek4") == 0) {
        *out = EMBER_PROMPT_DEEPSEEK_DSML;
        return true;
    }
    // #27742 defines the GGUF architecture spelling as qwen4exp. Accept the
    // Transformers spelling too so pre-release conversion tools can fail at
    // the engine's stricter tensor gate rather than selecting DeepSeek DSML.
    if (strcmp(architecture, "qwen4exp") == 0 ||
        strcmp(architecture, "qwen4_exp") == 0) {
        *out = EMBER_PROMPT_QWEN4_CHATML;
        return true;
    }
    return false;
}

static void set_error(char **err, const char *prefix, const char *detail) {
    if (!err) return;
    const int needed = snprintf(NULL, 0, "%s%s%s", prefix,
                                detail ? ": " : "", detail ? detail : "");
    if (needed < 0) return;
    *err = (char *)malloc((size_t)needed + 1);
    if (!*err) return;
    (void)snprintf(*err, (size_t)needed + 1, "%s%s%s", prefix,
                   detail ? ": " : "", detail ? detail : "");
}

bool ember_prompt_profile_detect(const char *model_path,
                                 ember_prompt_profile *out,
                                 char **err) {
    if (err) *err = NULL;
    if (!model_path || !out) {
        set_error(err, "invalid model-profile arguments", NULL);
        return false;
    }
    gguf_file *gguf = ember_gguf_open(model_path);
    if (!gguf) {
        set_error(err, "cannot read GGUF metadata", model_path);
        return false;
    }
    const char *architecture =
        ember_gguf_get_str(gguf, "general.architecture", NULL);
    const bool ok = ember_prompt_profile_from_arch(architecture, out);
    if (!ok) {
        set_error(err, architecture ? "unsupported GGUF architecture"
                                    : "missing GGUF general.architecture",
                  architecture);
    }
    ember_gguf_free(gguf);
    return ok;
}
