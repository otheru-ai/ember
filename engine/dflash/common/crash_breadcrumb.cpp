// Definition of the crash breadcrumb global. See crash_breadcrumb.h.
//
// Default visibility so tools/segvtrace.c can find it with
// dlsym(RTLD_DEFAULT, "ds4_crash_breadcrumb") from its LD_PRELOAD context.
#include "crash_breadcrumb.h"

extern "C" {
__attribute__((visibility("default")))
struct ds4_crash_bc ds4_crash_breadcrumb = {};
__attribute__((visibility("default")))
struct ds4_kv_view_bc ds4_kv_view_breadcrumb = {};
}
