// Model-family dispatch above the backend ABI.
//
// Prompt syntax is part of a model's token ABI.  The client-facing `model`
// field is only an advertised alias, so routing from it would let a request
// select markers that do not belong to the loaded weights.  Ember instead
// reads `general.architecture` from the GGUF header once at startup.
#ifndef EMBER_MODEL_PROFILE_H
#define EMBER_MODEL_PROFILE_H

#include <stdbool.h>

#include "../server/chat_api.h"

bool ember_prompt_profile_from_arch(const char *architecture,
                                    ember_prompt_profile *out);

// Read the GGUF header and resolve its prompt ABI. Returns false for an
// unreadable file, missing/unknown architecture, or invalid arguments.
bool ember_prompt_profile_detect(const char *model_path,
                                 ember_prompt_profile *out,
                                 char **err);

#endif  // EMBER_MODEL_PROFILE_H
