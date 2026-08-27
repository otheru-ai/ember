#include "common/errors.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

static pthread_mutex_t g_error_lock = PTHREAD_MUTEX_INITIALIZER;
static char *g_last_error;
static _Thread_local char *g_error_snapshot;

static char *duplicate_message(const char *message) {
    const char *source = message ? message : "";
    const size_t length = strlen(source);
    char *copy = (char *)malloc(length + 1);
    if (!copy) return NULL;
    memcpy(copy, source, length + 1);
    return copy;
}

void dflash_set_last_error(const char *message) {
    char *replacement = duplicate_message(message);
    if (!replacement) return;
    pthread_mutex_lock(&g_error_lock);
    char *previous = g_last_error;
    g_last_error = replacement;
    pthread_mutex_unlock(&g_error_lock);
    free(previous);
}

const char *dflash_last_error(void) {
    pthread_mutex_lock(&g_error_lock);
    char *snapshot = duplicate_message(g_last_error);
    pthread_mutex_unlock(&g_error_lock);
    if (!snapshot) return g_error_snapshot ? g_error_snapshot : "";
    free(g_error_snapshot);
    g_error_snapshot = snapshot;
    return g_error_snapshot;
}
