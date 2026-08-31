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
// SUPERSEDES the vendored engine policy layer. The engine bridge supplies only
// snapshot blobs and disk init/save/lookup/longest-prefix primitives;
// checkpoint-cut and slot policy live here and in main.c. Do not assume Ember
// shares ds4/lucebox checkpoint timing: it deliberately reimplements it above
// the backend ABI.
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
    // Content digest of the image whose embeddings are spliced INSIDE this
    // stored prefix, or 0 when the prefix stops before any image token.
    //
    // Token IDs alone cannot distinguish two images: the vision graft emits a
    // fixed 64-entry palette cycle at image positions, so every picture yields
    // the SAME ids. Without this field, a request carrying image B matches the
    // prefix cached for image A, restores A's KV, prefills nothing, and answers
    // about A -- fluently, and with nothing in the logs to say so.
    uint64_t img_digest;
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
//
// `img_digest` is the content digest of the image in THIS prompt (0 = none). An
// entry that stored image content is only a legal hit when its digest matches:
// the slot holds KV for exactly entry->len tokens, so a mismatched entry cannot
// be salvaged by shortening the match -- restoring it would hand back more KV
// than the returned length claims. Reject, do not truncate.
//
// Entries that stop before any image (digest 0) are unaffected, which is the
// common and valuable case: the long shared system prefix is still reused.
void ember_kv_lookup(ember_kv_cache *c, const int32_t *prompt, int n,
                     uint64_t img_digest, int *slot_out, int *len_out);

// Token-only disk keys cannot identify image content. Clamp any candidate
// lookup/save length to the first image row; -1 means no image is present.
// Kept as a pure policy helper so both doors are pinned by GPU-free tests.
int ember_kv_clamp_before_image(int candidate, int img_span_start);

// A post-tool snapshot physically contains the full current frontier and
// cannot be relabeled as a shorter pre-image prefix. Until the disk format is
// media-aware, skip this snapshot entirely when either request metadata or the
// legacy token span says that frontier contains image state.
bool ember_kv_post_tool_snapshot_safe(bool request_has_images,
                                      int img_span_start);

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
// `img_digest` is the image in this prompt and `img_span_start` the index of
// its first token (-1 when there is no image). The entry records the digest
// only when the cut actually reaches into the image span -- a prefix that stops
// short of it contains no image content and must stay reusable across images.
bool ember_kv_commit(ember_kv_cache *c, int slot,
                     const int32_t *prompt, int cut,
                     uint64_t img_digest, int img_span_start);

#endif  // EMBER_KV_CACHE_H
