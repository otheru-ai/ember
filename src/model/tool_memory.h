// B3 — exact-DSML tool-call replay memory.
//
// Maps a minted tool-call id -> the EXACT bytes the model sampled for that
// assistant turn AND the exact token ids it decoded. On replay the renderer
// emits a splice sentinel and the encoder splices the stored token ids verbatim
// (rather than re-tokenizing the text — DSML markers are special tokens that do
// not round-trip through detokenize->retokenize), so the re-encoded history is
// TOKEN-identical to what was sampled. That's what lets the post-tool-call
// snapshot continue instead of re-prefilling (see docs/b3-tool-call-replay-scope.md).
//
// Accessed ONLY on the generation worker thread, so it needs no locking.
#ifndef EMBER_TOOL_MEMORY_H
#define EMBER_TOOL_MEMORY_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    char         *id;      // owned
    char         *bytes;   // owned: exact sampled text (name verification / debug)
    size_t        len;
    int32_t      *ids;     // owned: exact sampled token ids (for splicing)
    int           n_ids;
    unsigned long stamp;   // LRU: higher = more recently used
} ember_tool_mem_entry;

typedef struct {
    ember_tool_mem_entry *e;
    int           n, cap;
    unsigned long clock;
} ember_tool_memory;

void ember_tool_memory_init(ember_tool_memory *tm, int max_entries);
void ember_tool_memory_free(ember_tool_memory *tm);

// Store id -> (bytes[0:len], ids[0:n_ids]) (copied). Re-put replaces; LRU-evicts.
void ember_tool_memory_put(ember_tool_memory *tm, const char *id,
                           const char *bytes, size_t len,
                           const int32_t *ids, int n_ids);

// Borrowed NUL-terminated exact bytes for id, or NULL. Marks most-recent.
const char *ember_tool_memory_get(ember_tool_memory *tm, const char *id);

// Borrowed exact token ids for id (*n set to count), or NULL. Marks most-recent.
const int32_t *ember_tool_memory_get_tokens(ember_tool_memory *tm, const char *id, int *n);

#endif  // EMBER_TOOL_MEMORY_H
