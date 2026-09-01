// B3 — exact-DSML tool-call replay memory. See tool_memory.h.
// Small count-bounded LRU; single-threaded (worker), linear scan (N is small
// and lookups are ~one per replayed assistant tool turn).
#include "tool_memory.h"

#include <stdlib.h>
#include <string.h>

void ember_tool_memory_init(ember_tool_memory *tm, int max_entries) {
    tm->cap = max_entries > 0 ? max_entries : 512;
    tm->e = (ember_tool_mem_entry *)calloc((size_t)tm->cap, sizeof(*tm->e));
    tm->n = 0;
    tm->clock = 0;
}

static void entry_clear(ember_tool_mem_entry *e) {
    free(e->id); free(e->bytes); free(e->ids);
    e->id = e->bytes = NULL; e->ids = NULL; e->len = 0; e->n_ids = 0;
}

void ember_tool_memory_free(ember_tool_memory *tm) {
    if (!tm->e) return;
    for (int i = 0; i < tm->n; i++) entry_clear(&tm->e[i]);
    free(tm->e);
    tm->e = NULL; tm->n = tm->cap = 0;
}

static int find_idx(ember_tool_memory *tm, const char *id) {
    for (int i = 0; i < tm->n; i++)
        if (tm->e[i].id && strcmp(tm->e[i].id, id) == 0) return i;
    return -1;
}

void ember_tool_memory_put(ember_tool_memory *tm, const char *id,
                           const char *bytes, size_t len,
                           const int32_t *ids, int n_ids) {
    if (!tm->e || !id || !id[0] || !bytes) return;
    // copy bytes
    char *b = (char *)malloc(len + 1);
    if (!b) return;
    memcpy(b, bytes, len); b[len] = '\0';
    // copy token ids (the exact splice payload; may legitimately be empty)
    int32_t *ic = NULL;
    if (ids && n_ids > 0) {
        ic = (int32_t *)malloc((size_t)n_ids * sizeof(int32_t));
        if (!ic) { free(b); return; }
        memcpy(ic, ids, (size_t)n_ids * sizeof(int32_t));
    } else {
        n_ids = 0;
    }

    int i = find_idx(tm, id);
    if (i < 0) {
        if (tm->n < tm->cap) {
            i = tm->n++;
            tm->e[i].id = strdup(id);
            if (!tm->e[i].id) { free(b); free(ic); tm->n--; return; }
        } else {
            // evict least-recently-used
            int lru = 0;
            for (int k = 1; k < tm->n; k++)
                if (tm->e[k].stamp < tm->e[lru].stamp) lru = k;
            char *keep = strdup(id);
            if (!keep) { free(b); free(ic); return; }
            entry_clear(&tm->e[lru]);
            tm->e[lru].id = keep;
            i = lru;
        }
    } else {
        free(tm->e[i].bytes);
        free(tm->e[i].ids);
    }
    tm->e[i].bytes = b;
    tm->e[i].len = len;
    tm->e[i].ids = ic;
    tm->e[i].n_ids = n_ids;
    tm->e[i].stamp = ++tm->clock;
}

const char *ember_tool_memory_get(ember_tool_memory *tm, const char *id) {
    if (!tm->e || !id || !id[0]) return NULL;
    int i = find_idx(tm, id);
    if (i < 0) return NULL;
    tm->e[i].stamp = ++tm->clock;
    return tm->e[i].bytes;
}

const int32_t *ember_tool_memory_get_tokens(ember_tool_memory *tm, const char *id, int *n) {
    if (n) *n = 0;
    if (!tm->e || !id || !id[0]) return NULL;
    int i = find_idx(tm, id);
    if (i < 0 || tm->e[i].n_ids <= 0) return NULL;
    tm->e[i].stamp = ++tm->clock;
    if (n) *n = tm->e[i].n_ids;
    return tm->e[i].ids;
}
