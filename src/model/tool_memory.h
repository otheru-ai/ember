// B3 — exact-DSML tool-call replay memory.
//
// Maps a minted tool-call id -> the EXACT sampled DSML block and, when its byte
// boundaries align with decoded-token boundaries, the exact token ids for that
// block. The surrounding reasoning/content is always rendered canonically.
// This matches Dwarfstar's replay scope and prevents hidden assistant text from
// being resurrected through a tool id. DSML markers do not reliably survive a
// detokenize->retokenize round trip, so exact block token ids are spliced when
// available. The exact-token rationale and invariants are documented below.
//
// Accessed ONLY on the generation worker thread, so it needs no locking.
#ifndef EMBER_TOOL_MEMORY_H
#define EMBER_TOOL_MEMORY_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

typedef struct {
    char         *id;      // owned
    char         *bytes;   // owned: exact sampled DSML block
    size_t        len;
    int32_t      *ids;     // owned: exact sampled token ids (for splicing)
    int           n_ids;
    unsigned long stamp;   // LRU: higher = more recently used
} ember_tool_mem_entry;

typedef struct {
    ember_tool_mem_entry *e;
    int           n, cap;
    size_t        bytes, max_bytes;
    unsigned long clock;
    char         *persist_dir;       // model-scoped directory, or NULL
    uint8_t       persist_identity[16];
    unsigned long persist_failures;  // diagnostic; RAM fallback remains valid
} ember_tool_memory;

void ember_tool_memory_init(ember_tool_memory *tm, int max_entries, size_t max_bytes);
void ember_tool_memory_free(ember_tool_memory *tm);

// Enable model-scoped, crash-safe persistence under root_dir and restore the
// newest valid entries. Returns the number loaded, or -1 if persistence could
// not be initialized. RAM replay remains usable after a failure.
int ember_tool_memory_enable_persistence(ember_tool_memory *tm,
                                         const char *root_dir,
                                         const uint8_t identity[16]);

bool ember_tool_memory_persistence_enabled(const ember_tool_memory *tm);

// Store id -> (bytes[0:len], ids[0:n_ids]) (copied). Re-put replaces; LRU-evicts.
void ember_tool_memory_put(ember_tool_memory *tm, const char *id,
                           const char *bytes, size_t len,
                           const int32_t *ids, int n_ids);

// Borrowed NUL-terminated exact bytes for id, or NULL. Marks most-recent.
// Borrowed pointer into the store, valid ONLY while the caller holds the lock
// that also serializes ember_tool_memory_put() -- srv->state_lock in the
// server. _put() evicts under pressure and free()s exactly this buffer, so an
// unlocked read is a use-after-free rather than a stale read. Not a pure read
// either: it bumps the LRU stamp and touches the persisted file.
const char *ember_tool_memory_get(ember_tool_memory *tm, const char *id);

// Borrowed exact token ids for id (*n set to count), or NULL. Marks most-recent.
// Same borrowing and locking contract as ember_tool_memory_get().
const int32_t *ember_tool_memory_get_tokens(ember_tool_memory *tm, const char *id, int *n);

#endif  // EMBER_TOOL_MEMORY_H
