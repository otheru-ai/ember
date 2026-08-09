// Prefix KV cache — the token-level bookkeeping that lets the backend restore a
// prior turn's KV and prefill only the new suffix. This is the logic that gives
// the 29×/56× prefill reuse; it holds NO GPU state itself — it maps token
// prefixes to backend snapshot slots and decides where to snapshot.
//
// Two reuse points, ported from the ds4/lucebox prefix cache:
//   - lookup(): longest stored prefix that is a prefix of the new prompt →
//     (slot, prefix_len). The backend restores that slot and prefills only
//     prompt[prefix_len:].
//   - snap_cut(): where to snapshot the NEW prompt for future reuse — the last
//     completed turn boundary, or (cold prompt, no boundary yet) the ANCHOR:
//     the last user marker before the first assistant marker, so the shared
//     system-prompt prefix is cached from turn 1.
//
// SUPERSEDES the vendored engine policy layer. engine/dflash/server/
// prefix_cache.cpp (the inline+full-slot PrefixCache class) is NOT instantiated
// in ember, and DiskPrefixCache's policy methods (continued_interval /
// cold_prefix_boundary / boundary_align/trim, maybe_store_continued) are linked
// but UNWIRED — ember consumes only its init/save/lookup/longest_prefix_len blob
// interface. Checkpoint-cut and eviction policy live here + in main.c instead.
// Do not assume ember shares ds4/lucebox checkpoint TIMING: it reimplements it.
// (This is why disk_prefix_cache.h's boundary_align_tokens default is inert.)
#ifndef EMBER_KV_CACHE_H
#define EMBER_KV_CACHE_H

#include <stdbool.h>
#include <stdint.h>

#define EMBER_KV_MAX_SLOTS 64

typedef struct {
    int32_t *ids;      // owned copy of the prefix tokens [0,len)
    int      len;
    int      slot;     // backend snapshot slot holding this prefix's KV
    int64_t  lru;      // higher = more recently used
} ember_kv_entry;

typedef struct {
    ember_kv_entry entries[EMBER_KV_MAX_SLOTS];
    int            n_entries;
    int            cap;          // usable slots (<= MAX_SLOTS)
    int            next_slot;    // round-robin cursor
    int64_t        lru_counter;
    bool           reserved[EMBER_KV_MAX_SLOTS];
    unsigned       pins[EMBER_KV_MAX_SLOTS];
    int            anchor_min;   // min tokens for an anchor checkpoint
    // chat markers (single special-token ids) for boundary/anchor detection
    int32_t        user_tok;
    int32_t        assistant_tok;
    int32_t        eos_tok;      // turn separator
} ember_kv_cache;

void ember_kv_init(ember_kv_cache *c, int cap_slots, int32_t user_tok,
                   int32_t assistant_tok, int32_t eos_tok);
void ember_kv_free(ember_kv_cache *c);

// Longest stored prefix of `prompt` → *slot,*len (or -1,0 if no hit). Marks the
// hit entry most-recently-used.
void ember_kv_lookup(ember_kv_cache *c, const int32_t *prompt, int n,
                     int *slot_out, int *len_out);

// Prevent a selected snapshot slot from being evicted/overwritten while a
// concurrent backend request restores it.
bool ember_kv_pin(ember_kv_cache *c, int slot);
void ember_kv_unpin(ember_kv_cache *c, int slot);

// Where to snapshot `prompt` for future reuse: the last completed turn boundary
// (eos followed by a role marker), else the anchor (last user before first
// assistant), else -1 if the prompt is too short. Returns the cut length.
// When `is_cold_anchor` is non-NULL it is set to 1 iff the returned cut is the
// cold-prompt anchor (no completed boundary yet) — the caller tags such a save
// as a cold system-prefix anchor for eviction protection — and 0 otherwise.
int ember_kv_snap_cut(ember_kv_cache *c, const int32_t *prompt, int n,
                      int *is_cold_anchor);

// Reserve a slot for a new snapshot at `cut` (LRU-evicting at capacity, but
// preferring to keep prefixes that are ancestors of others). Returns the slot.
int ember_kv_reserve(ember_kv_cache *c);

// Abandon an uncommitted reservation after generation/snapshot failure.
void ember_kv_release(ember_kv_cache *c, int slot);

// Commit: record that `slot` now holds prompt[0,cut). Drops any stale entry on
// the same slot.
// Record the logical token prefix for a backend snapshot. Returns false for an
// invalid entry or allocation failure; the slot remains uncached logically.
bool ember_kv_commit(ember_kv_cache *c, int slot,
                     const int32_t *prompt, int cut);

#endif  // EMBER_KV_CACHE_H
