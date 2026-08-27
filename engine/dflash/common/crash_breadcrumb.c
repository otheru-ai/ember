// Definition of the crash breadcrumb globals. See crash_breadcrumb.h.
//
// Default visibility lets tools/segvtrace.c resolve these symbols with
// dlsym(RTLD_DEFAULT, ...) from its LD_PRELOAD context.
#include "crash_breadcrumb.h"

__attribute__((visibility("default")))
struct ds4_crash_bc ds4_crash_breadcrumb = {0};

__attribute__((visibility("default")))
struct ds4_kv_view_bc ds4_kv_view_breadcrumb = {0};
