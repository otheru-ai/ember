// Protocol-bound continuation state.
//
// A tool-call id set names the exact sampled token frontier that produced it.
// Keeping the complete frontier (not only the generated tool-call tokens) lets
// an ID-only tool-result request reconstruct the authoritative prompt, restore
// its disk KV snapshot when available, and append a separately-tokenized suffix.
#ifndef EMBER_CONTINUATION_H
#define EMBER_CONTINUATION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef enum {
    EMBER_API_CHAT = 1,
    EMBER_API_RESPONSES = 2,
    EMBER_API_ANTHROPIC = 3,
    EMBER_API_COMPLETIONS = 4,
} ember_api_kind;

typedef struct {
    ember_api_kind api;
    char         **call_ids;
    int            n_call_ids;
    int32_t       *frontier_ids;
    int            n_frontier;
    char          *visible_text;
    char          *tools_json;
    unsigned long  stamp;
} ember_continuation_entry;

typedef struct {
    ember_continuation_entry *entries;
    int            n;
    int            cap;
    size_t         bytes;
    size_t         max_bytes;
    unsigned long  clock;
    char          *persist_dir;
    uint8_t        persist_identity[16];
    unsigned long  persist_failures;
} ember_continuation_store;

void ember_continuation_init(ember_continuation_store *s, int max_entries,
                             size_t max_bytes);
void ember_continuation_free(ember_continuation_store *s);

// Enable model-scoped persistence and restore valid records. Returns the number
// restored, or -1 if persistence could not be initialized.
int ember_continuation_enable_persistence(ember_continuation_store *s,
                                          const char *root_dir,
                                          const uint8_t identity[16]);

// Remember one authoritative frontier. call_ids are treated as an unordered
// exact set. Existing records sharing any id are replaced.
bool ember_continuation_put(ember_continuation_store *s, ember_api_kind api,
                            const char *const *call_ids, int n_call_ids,
                            const int32_t *frontier_ids, int n_frontier,
                            const char *visible_text,
                            const char *tools_json);

// Find an entry only when api and the complete call-id set match. The returned
// entry is borrowed until the next store mutation.
const ember_continuation_entry *
ember_continuation_get(ember_continuation_store *s, ember_api_kind api,
                       const char *const *call_ids, int n_call_ids);

#endif
