#include "kv_cache.h"

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
    if (alen > blen) return false;
    return memcmp(a, b, (size_t)alen * sizeof(int32_t)) == 0;
}

void ember_kv_lookup(ember_kv_cache *c, const int32_t *prompt, int n,
                     int *slot_out, int *len_out) {
    int best = -1, best_len = 0;
    for (int i = 0; i < c->n_entries; i++) {
        ember_kv_entry *e = &c->entries[i];
        if (e->len == 0 || e->len > n || e->len <= best_len) continue;
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

// Turn boundaries: an eos token followed (within a couple tokens) by a role
// marker. Returns the count of boundaries and fills up to `max` cut positions.
static int find_boundaries(const ember_kv_cache *c, const int32_t *p, int n,
                           int *cuts, int max) {
    int k = 0;
    for (int i = 0; i < n - 1 && k < max; i++) {
        if (p[i] != c->eos_tok) continue;
        for (int j = i + 1; j <= i + 3 && j < n; j++) {
            if (p[j] == c->user_tok || p[j] == c->assistant_tok) {
                cuts[k++] = j + 1;  // cut just after the role marker
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

int ember_kv_snap_cut(ember_kv_cache *c, const int32_t *prompt, int n) {
    int cuts[16];
    int nb = find_boundaries(c, prompt, n, cuts, 16);
    if (nb > 0) {
        // second-to-last boundary = through the last completed exchange
        int cut = nb >= 2 ? cuts[nb - 2] : cuts[nb - 1];
        return cut;
    }
    return anchor_cut(c, prompt, n);  // cold prompt → anchor (or -1)
}

// Prefer evicting a "leaf": an entry whose ids are NOT a strict prefix of any
// other entry, so shared ancestor prefixes stay resident. Among leaves, evict
// the least-recently-used.
static int evict_victim(ember_kv_cache *c) {
    int victim = -1;
    int64_t best_lru = 0;
    for (int i = 0; i < c->n_entries; i++) {
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
    return victim >= 0 ? victim : 0;  // fall back to entry 0
}

int ember_kv_reserve(ember_kv_cache *c) {
    int cap = c->cap > 0 ? c->cap : 1;
    if (c->n_entries < cap) {
        // Return a slot id NOT owned by any live entry. The old round-robin
        // cursor could, once holes appeared, hand back a slot a committed entry
        // (possibly a protected ancestor) still owned — bypassing the eviction
        // policy and, on a non-committing generation, corrupting its KV.
        for (int s = 0; s < cap; s++) {
            bool used = false;
            for (int i = 0; i < c->n_entries; i++)
                if (c->entries[i].slot == s) { used = true; break; }
            if (!used) return s;
        }
    }
    int v = evict_victim(c);
    int slot = c->entries[v].slot;
    free(c->entries[v].ids);
    c->entries[v] = c->entries[--c->n_entries];  // swap-remove
    return slot;
}

void ember_kv_commit(ember_kv_cache *c, int slot, const int32_t *prompt, int cut) {
    // drop any stale entry pointing at this slot
    for (int i = c->n_entries - 1; i >= 0; i--) {
        if (c->entries[i].slot == slot) {
            free(c->entries[i].ids);
            c->entries[i] = c->entries[--c->n_entries];
        }
    }
    if (c->n_entries >= EMBER_KV_MAX_SLOTS) return;
    ember_kv_entry *e = &c->entries[c->n_entries++];
    e->ids = (int32_t *)malloc((size_t)cut * sizeof(int32_t));
    memcpy(e->ids, prompt, (size_t)cut * sizeof(int32_t));
    e->len = cut;
    e->slot = slot;
    e->lru = ++c->lru_counter;
}
