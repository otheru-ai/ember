// Thread-safe diagnostic channel shared by the DeepSeek4 loader and backend.
// The storage API is C; temporary C++ strings are accepted by a thin inline
// adapter until the loader itself is migrated.
#ifndef DFLASH_COMMON_ERRORS_H
#define DFLASH_COMMON_ERRORS_H

#ifdef __cplusplus
extern "C" {
#endif

void dflash_set_last_error(const char *message);
const char *dflash_last_error(void);

#ifdef __cplusplus
}

#include <string>

namespace dflash::common {

inline void set_last_error(std::string message) {
    dflash_set_last_error(message.c_str());
}

inline const char *last_error() {
    return dflash_last_error();
}

}  // namespace dflash::common
#endif

#endif
