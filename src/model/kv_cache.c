#include "kv_cache.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

void ember_kv_init(ember_kv_cache *c, int cap_slots, int32_t user_tok,
                   int32_t assistant_tok, int32_t eos_tok) {
    memset(c, 0, sizeof(*c));
    c->cap = cap_slots < EMBER_KV_MAX_SLOTS ? cap_slots : EMBER_KV_MAX_SLOTS;
    if (c->cap < 0) c->cap = 0;
    c->anchor_min = 512;
    c->user_tok = user_tok;
    c->assistant_tok = assistant_tok;
    c->eos_tok = eos_tok;
}

void ember_kv_free(ember_kv_cache *c) {
    for (int i = 0; i < c->n_entries; i++) free(c->entries[i].ids);
    memset(c, 0, sizeof(*c));
}

static bool is_prefix(const int32_t *a, int alen, const int32_t *b, int blen) {
    if (alen < 0 || blen < 0 || (alen > 0 && (!a || !b))) return false;
    if (alen > blen) return false;
    return memcmp(a, b, (size_t)alen * sizeof(int32_t)) == 0;
}

void ember_kv_lookup(ember_kv_cache *c, const int32_t *prompt, int n,
                     uint64_t img_digest, int *slot_out, int *len_out) {
    if (!slot_out || !len_out) return;
    if (!c || n < 0 || (n > 0 && !prompt)) {
        *slot_out = -1;
        *len_out = 0;
        return;
    }
    int best = -1, best_len = 0;
    for (int i = 0; i < c->n_entries; i++) {
        ember_kv_entry *e = &c->entries[i];
        if (e->len == 0 || e->len > n || e->len <= best_len) continue;
        // An entry holding image content is usable only for the SAME image.
        // Shortening the match is not an option: the slot's KV covers exactly
        // e->len tokens, so a shorter returned length would leave the backend
        // holding image KV it believes it has not prefilled.
        if (e->img_digest != 0 && e->img_digest != img_digest) continue;
        if (is_prefix(e->ids, e->len, prompt, n)) { best = i; best_len = e->len; }
    }
    if (best >= 0) {
        c->entries[best].lru = ++c->lru_counter;
        *slot_out = c->entries[best].slot;
        *len_out = best_len;
    } else {
        *slot_out = -1;
        *len_out = 0;
    }
}

bool ember_kv_pin(ember_kv_cache *c, int slot) {
    if (!c || slot < 0 || slot >= c->cap ||
        c->pins[slot] == UINT_MAX) return false;
    for (int i = 0; i < c->n_entries; ++i) {
        if (c->entries[i].slot != slot) continue;
        c->pins[slot]++;
        return true;
    }
    return false;
}

void ember_kv_unpin(ember_kv_cache *c, int slot) {
    if (!c || slot < 0 || slot >= c->cap || c->pins[slot] == 0) return;
    c->pins[slot]--;
}

// Turn boundaries: an eos token followed (within a couple tokens) by a role
// marker. Retains the most recent `max` cut positions. Long-running agent
// conversations routinely contain more boundaries than the small stack buffer
// in ember_kv_snap_cut(); keeping the first `max` would permanently freeze the
// snapshot frontier once that buffer filled.
static int find_boundaries(const ember_kv_cache *c, const int32_t *p, int n,
                           int *cuts, int max) {
    if (!c || !p || !cuts || n <= 1 || max <= 0) return 0;
    int k = 0;
    for (int i = 0; i < n - 1; i++) {
        if (p[i] != c->eos_tok) continue;
        for (int j = i + 1; j <= i + 3 && j < n; j++) {
            if (p[j] == c->user_tok || p[j] == c->assistant_tok) {
                if (k < max) {
                    cuts[k++] = j + 1;
                } else {
                    memmove(cuts, cuts + 1,
                            (size_t)(max - 1) * sizeof(*cuts));
                    cuts[max - 1] = j + 1;
                }
                break;
            }
        }
    }
    return k;
}

// Anchor: the last user marker before the first assistant marker.
static int anchor_cut(const ember_kv_cache *c, const int32_t *p, int n) {
    int last_user = -1;
    for (int i = 0; i < n; i++) {
        if (p[i] == c->assistant_tok) break;
        if (p[i] == c->user_tok) last_user = i;
    }
    return last_user >= c->anchor_min ? last_user : -1;
}

int ember_kv_snap_cut(ember_kv_cache *c, const int32_t *prompt, int n,
                      int *is_cold_anchor) {
    if (is_cold_anchor) *is_cold_anchor = 0;
    int cuts[16];
    int nb = find_boundaries(c, prompt, n, cuts, 16);
    if (nb > 0) {
        // Cut at the FULL prompt, not at the last turn boundary.
        //
        // cuts[nb-1] is the boundary opening the turn being answered now, so it
        // sits *before* that turn's tool result and before the reply about to be
        // generated. Committing there leaves the cache permanently one exchange
        // behind: measured against a growing agent conversation, every turn
        // restored 5,709 tokens less than the previous turn had reached and
        // re-prefilled 11,468 tokens while adding only 5,759 — a constant lag
        // that never converges. It is not "no extra prefill"; it is one whole
        // exchange of extra prefill, forever.
        //
        // n is still a legal cut: the prompt ends with the generation prompt
        // (<|Assistant|><think>), and the next turn re-renders that same
        // assistant turn starting with those exact bytes, so [0,n) remains a
        // prefix of the next prompt. main.c bounds it with snap_cut <= n_prompt,
        // and the snapshot is taken at the end of prefill — before any sampled
        // token — so it does not reach into the generation frontier. The
        // post-tool snapshot still commits the longer prompt+generated frontier
        // when the reply is clean; this one is the floor that survives a
        // watchdog trip or budget force-close, which suppress that path.
        return n;
    }
    int anchor = anchor_cut(c, prompt, n);  // cold prompt → anchor (or -1)
    if (is_cold_anchor && anchor > 0) *is_cold_anchor = 1;
    return anchor;
}

// Prefer evicting a "leaf": an entry whose ids are NOT a strict prefix of any
// other entry, so shared ancestor prefixes stay resident. Among leaves, evict
// the least-recently-used.
static int evict_victim(ember_kv_cache *c) {
    int victim = -1;
    int64_t best_lru = 0;
    for (int i = 0; i < c->n_entries; i++) {
        if (c->reserved[c->entries[i].slot] ||
            c->pins[c->entries[i].slot] > 0) continue;
        bool is_ancestor = false;
        for (int j = 0; j < c->n_entries; j++) {
            if (j == i) continue;
            if (is_prefix(c->entries[i].ids, c->entries[i].len,
                          c->entries[j].ids, c->entries[j].len)) {
                is_ancestor = true;
                break;
            }
        }
        if (is_ancestor) continue;
        if (victim < 0 || c->entries[i].lru < best_lru) {
            victim = i;
            best_lru = c->entries[i].lru;
        }
    }
    if (victim >= 0) return victim;
    // All unreserved entries were ancestors. Keep correctness over policy and
    // choose the oldest unreserved entry.
    for (int i = 0; i < c->n_entries; ++i) {
        if (c->reserved[c->entries[i].slot] ||
            c->pins[c->entries[i].slot] > 0) continue;
        if (victim < 0 || c->entries[i].lru < best_lru) {
            victim = i;
            best_lru = c->entries[i].lru;
        }
    }
    return victim;
}

int ember_kv_reserve(ember_kv_cache *c) {
    if (!c || c->cap <= 0) return -1;
    int cap = c->cap;
    if (c->n_entries < cap) {
        // Return a slot id NOT owned by any live entry. The old round-robin
        // cursor could, once holes appeared, hand back a slot a committed entry
        // (possibly a protected ancestor) still owned — bypassing the eviction
        // policy and, on a non-committing generation, corrupting its KV.
        for (int s = 0; s < cap; s++) {
            if (c->reserved[s] || c->pins[s] > 0) continue;
            bool used = false;
            for (int i = 0; i < c->n_entries; i++)
                if (c->entries[i].slot == s) { used = true; break; }
            if (!used) {
                c->reserved[s] = true;
                return s;
            }
        }
    }
    int v = evict_victim(c);
    if (v < 0) return -1;
    int slot = c->entries[v].slot;
    free(c->entries[v].ids);
    c->entries[v] = c->entries[--c->n_entries];  // swap-remove
    c->reserved[slot] = true;
    return slot;
}

void ember_kv_release(ember_kv_cache *c, int slot) {
    if (!c || slot < 0 || slot >= c->cap) return;
    c->reserved[slot] = false;
}

bool ember_kv_commit(ember_kv_cache *c, int slot,
                     const int32_t *prompt, int cut,
                     uint64_t img_digest, int img_span_start) {
    if (!c || c->cap <= 0 || slot < 0 || slot >= c->cap ||
        !prompt || cut <= 0) return false;
    // drop any stale entry pointing at this slot
    for (int i = c->n_entries - 1; i >= 0; i--) {
        if (c->entries[i].slot == slot) {
            free(c->entries[i].ids);
            c->entries[i] = c->entries[--c->n_entries];
        }
    }
    if (c->n_entries >= EMBER_KV_MAX_SLOTS ||
        (size_t)cut > SIZE_MAX / sizeof(int32_t)) return false;
    int32_t *copy = (int32_t *)malloc((size_t)cut * sizeof(int32_t));
    if (!copy) return false;
    memcpy(copy, prompt, (size_t)cut * sizeof(int32_t));
    ember_kv_entry *e = &c->entries[c->n_entries++];
    e->ids = copy;
    e->len = cut;
    e->slot = slot;
    e->lru = ++c->lru_counter;
    // Tag the entry with the image ONLY if the stored prefix actually reaches
    // into the image span. A cut that stops short holds no image content, and
    // tagging it would needlessly forbid reuse across images -- which is
    // exactly the long shared system prefix we most want to keep.
    e->img_digest =
        (img_span_start >= 0 && cut > img_span_start) ? img_digest : 0;
    c->reserved[slot] = false;
    return true;
}
