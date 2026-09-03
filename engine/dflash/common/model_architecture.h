// Architecture identification for Ember's standalone engine.
//
// The backend factory reads `general.architecture` out of the GGUF and refuses
// anything it does not implement, rather than loading a file whose tensor
// layout it is only assuming. DeepSeek-V4-Flash declares "deepseek4"; there is
// no second implemented architecture, so UNKNOWN is the whole of the rest.

#pragma once

#include <cstring>
#include <string>

namespace dflash::common {

enum class ModelArchitecture {
    DEEPSEEK4,
    UNKNOWN,
};

inline ModelArchitecture model_architecture_from_name(const char * name) {
    if (!name) return ModelArchitecture::UNKNOWN;
    if (std::strcmp(name, "deepseek4") == 0) {
        return ModelArchitecture::DEEPSEEK4;
    }
    return ModelArchitecture::UNKNOWN;
}

inline const char * model_architecture_name(ModelArchitecture architecture) {
    switch (architecture) {
        case ModelArchitecture::DEEPSEEK4: return "deepseek4";
        case ModelArchitecture::UNKNOWN: return "unknown";
    }
    return "unknown";
}

// Opens the GGUF metadata only. `error` carries the reason on UNKNOWN, which
// is what the caller reports; an unreadable file and an unimplemented
// architecture are different messages because they need different fixes.
ModelArchitecture inspect_gguf_architecture(const char * model_path,
                                            std::string & error);

}  // namespace dflash::common
