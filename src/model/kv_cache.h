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

// Where to snapshot `prompt` for future reuse: the last completed turn boundary
// (eos followed by a role marker), else the anchor (last user before first
// assistant), else -1 if the prompt is too short. Returns the cut length.
int ember_kv_snap_cut(ember_kv_cache *c, const int32_t *prompt, int n);

// Reserve a slot for a new snapshot at `cut` (LRU-evicting at capacity, but
// preferring to keep prefixes that are ancestors of others). Returns the slot.
int ember_kv_reserve(ember_kv_cache *c);

// Commit: record that `slot` now holds prompt[0,cut). Drops any stale entry on
// the same slot.
void ember_kv_commit(ember_kv_cache *c, int slot, const int32_t *prompt, int cut);

#endif  // EMBER_KV_CACHE_H
