#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/model/kv_cache.h"

static int g_pass = 0, g_fail = 0;
#define CHECK(c, m) do { if (c) g_pass++; else { g_fail++; printf("  FAIL: %s\n", m); } } while (0)

// marker ids: user=1, assistant=2, eos=3
#define U 1
#define A 2
#define E 3

static void test_longest_prefix(void) {
    ember_kv_cache c;
    ember_kv_init(&c, 32, U, A, E);
    c.anchor_min = 2;  // small for tests
    int32_t p1[] = {U, 10, 11, 12, A, 20};       // 6 tokens
    int slot = ember_kv_reserve(&c);
    ember_kv_commit(&c, slot, p1, 4, 0, -1);            // cache prefix [U,10,11,12]

    // a longer prompt extending that prefix → hit at len 4
    int32_t p2[] = {U, 10, 11, 12, A, 20, E, U, 30};
    int s, len;
    ember_kv_lookup(&c, p2, 9, 0, &s, &len);
    CHECK(s == slot && len == 4, "longest-prefix hit reuses the cached slot");

    // a divergent prompt → no hit
    int32_t p3[] = {U, 99};
    ember_kv_lookup(&c, p3, 2, 0, &s, &len);
    CHECK(s == -1 && len == 0, "divergent prompt: no hit");
    ember_kv_free(&c);
}

static void test_anchor_cut(void) {
    ember_kv_cache c;
    ember_kv_init(&c, 32, U, A, E);
    c.anchor_min = 3;
    // system scaffolding then first user question, no assistant yet (cold turn 1)
    // [U(0) s s s U(4) q q]  → anchor = last user before first assistant = idx 4
    int32_t p[] = {U, 100, 101, 102, U, 200, 201};
    int is_anchor = -1;
    int cut = ember_kv_snap_cut(&c, p, 7, &is_anchor);
    CHECK(cut == 4, "anchor = last user marker before first assistant");
    CHECK(is_anchor == 1, "cold-prompt anchor cut is flagged (reason=COLD)");
    ember_kv_free(&c);
}

static void test_turn_boundary_cut(void) {
    ember_kv_cache c;
    ember_kv_init(&c, 32, U, A, E);
    // [U q A a E U q2 A a2 E U q3]  boundaries after each E→role
    // eos at idx 4 → role U at 5 → cut 6 ; eos at idx 10 → role U at 11 → cut 12
    int32_t p[] = {U,10,A,20,E, U,11,A,21,E, U,12};  // 12 tokens
    int is_anchor = -1;
    int cut = ember_kv_snap_cut(&c, p, 12, &is_anchor);
    // Two boundaries exist (cut 6 and cut 11), but once ANY completed exchange
    // is present the cut is the whole prompt. Stopping at cut 11 leaves the
    // turn currently being answered — its tool result and the reply about to be
    // generated — outside the snapshot, so the next turn re-prefills that whole
    // exchange. Measured against a growing agent conversation that was a
    // constant 11,468-token re-prefill per turn while only 5,759 tokens were
    // added. [0,n) stays a valid prefix of the next prompt because the trailing
    // generation prompt is re-rendered verbatim.
    CHECK(cut == 12, "snap at the full prompt, not the last turn boundary");
    CHECK(is_anchor == 0, "completed-boundary cut is not a cold anchor (reason=NORMAL)");
    ember_kv_free(&c);
}

static void test_recent_turn_boundary_cut(void) {
    ember_kv_cache c;
    ember_kv_init(&c, 32, U, A, E);

    // More boundaries than the internal 16-entry scratch buffer. Every E,U
    // pair contributes a cut; the newest cuts are 39 and 41, so the requested
    // last boundary is 41. The old first-16 scan returned 31 and stopped
    // advancing forever after a long conversation crossed 16 turns.
    int32_t p[41];
    p[0] = U;
    for (int turn = 0; turn < 20; turn++) {
        p[1 + turn * 2] = E;
        p[2 + turn * 2] = U;
    }
    int is_anchor = -1;
    int cut = ember_kv_snap_cut(&c, p, 41, &is_anchor);
    CHECK(cut == 41,
          "snapshot frontier follows the most recent turns after 16 boundaries");
    CHECK(is_anchor == 0,
          "recent completed-boundary cut is not a cold anchor");
    ember_kv_free(&c);
}

static void test_lru_eviction(void) {
    ember_kv_cache c;
    ember_kv_init(&c, 2, U, A, E);  // only 2 slots
    int32_t a[] = {U, 1};
    int32_t b[] = {U, 2};
    int32_t d[] = {U, 3};
    int s;
    s = ember_kv_reserve(&c); ember_kv_commit(&c, s, a, 2, 0, -1);
    s = ember_kv_reserve(&c); ember_kv_commit(&c, s, b, 2, 0, -1);
    // touch 'a' so 'b' is LRU
    int slot, len; ember_kv_lookup(&c, a, 2, 0, &slot, &len);
    // reserve a third → must evict (prefer LRU leaf = b)
    s = ember_kv_reserve(&c); ember_kv_commit(&c, s, d, 2, 0, -1);
    // 'a' should still be present, 'b' evicted
    ember_kv_lookup(&c, a, 2, 0, &slot, &len);
    CHECK(slot >= 0 && len == 2, "recently-used entry survives eviction");
    ember_kv_lookup(&c, b, 2, 0, &slot, &len);
    CHECK(slot == -1, "LRU entry evicted at capacity");
    CHECK(c.n_entries == 2, "cache stays within capacity");
    ember_kv_free(&c);
}

static void test_disabled_and_invalid_cache(void) {
    ember_kv_cache c;
    ember_kv_init(&c, 0, U, A, E);
    int32_t p[] = {U, 1};
    CHECK(ember_kv_reserve(&c) == -1,
          "zero-capacity cache does not manufacture a slot");
    CHECK(!ember_kv_commit(&c, 0, p, 2, 0, -1),
          "zero-capacity cache rejects commits");
    CHECK(c.n_entries == 0, "invalid commit cannot create an entry");
    ember_kv_free(&c);
}

static void test_inflight_slot_lifecycle(void) {
    ember_kv_cache c;
    ember_kv_init(&c, 2, U, A, E);
    int32_t a[] = {U, 1};
    int32_t b[] = {U, 2};
    int s0 = ember_kv_reserve(&c);
    int s1 = ember_kv_reserve(&c);
    CHECK(s0 >= 0 && s1 >= 0 && s0 != s1,
          "concurrent reservations receive distinct physical slots");
    CHECK(ember_kv_reserve(&c) == -1,
          "all in-flight slots apply backpressure");
    ember_kv_release(&c, s1);
    CHECK(ember_kv_commit(&c, s0, a, 2, 0, -1),
          "reserved slot can be committed");
    CHECK(ember_kv_commit(&c, s1, b, 2, 0, -1),
          "released slot remains usable");

    int slot = -1, len = 0;
    ember_kv_lookup(&c, a, 2, 0, &slot, &len);
    CHECK(slot == s0 && ember_kv_pin(&c, slot),
          "restore target can be pinned");
    int replacement = ember_kv_reserve(&c);
    CHECK(replacement == s1,
          "pinned restore target cannot be evicted");
    ember_kv_release(&c, replacement);
    ember_kv_unpin(&c, s0);
    replacement = ember_kv_reserve(&c);
    CHECK(replacement >= 0,
          "unpin makes a restore slot evictable again");
    ember_kv_release(&c, replacement);
    ember_kv_free(&c);
}


// The failure this field exists to prevent. The vision graft emits a FIXED
// palette cycle at image positions, so two different pictures produce identical
// prompt token IDs. Without the digest, image B's request matches image A's
// cached prefix, restores A's KV, prefills nothing, and answers about A.
static void test_image_digest_prevents_aliasing(void) {
    ember_kv_cache c;
    ember_kv_init(&c, 32, U, A, E);
    c.anchor_min = 2;

    // Identical tokens; the palette run starts at index 2. Only the pictures
    // (and therefore the digests) differ.
    int32_t prompt[] = {U, 50, 90, 91, 92, 93};
    const int span = 2;
    const uint64_t img_a = 0xAAAAAAAAAAAAAAAAULL;
    const uint64_t img_b = 0xBBBBBBBBBBBBBBBBULL;

    int slot = ember_kv_reserve(&c);
    ember_kv_commit(&c, slot, prompt, 6, img_a, span);   // cut reaches the image

    int s2, len;
    ember_kv_lookup(&c, prompt, 6, img_a, &s2, &len);
    CHECK(s2 == slot && len == 6, "same image still reuses its own cached KV");

    ember_kv_lookup(&c, prompt, 6, img_b, &s2, &len);
    CHECK(s2 == -1 && len == 0,
          "a DIFFERENT image with identical tokens must NOT hit that entry");

    ember_kv_lookup(&c, prompt, 6, 0, &s2, &len);
    CHECK(s2 == -1 && len == 0,
          "a prompt with no image must not restore image KV either");

    // A prefix that stops BEFORE the image carries no image content, so it must
    // stay reusable across images -- this is the long shared system prefix, the
    // case the cache exists for. Losing it would be a silent perf regression.
    ember_kv_cache c2;
    ember_kv_init(&c2, 32, U, A, E);
    c2.anchor_min = 2;
    int slot2 = ember_kv_reserve(&c2);
    ember_kv_commit(&c2, slot2, prompt, span, img_a, span);  // cut == span start
    ember_kv_lookup(&c2, prompt, 6, img_b, &s2, &len);
    CHECK(s2 == slot2 && len == span,
          "a prefix ending before the image is reused across different images");

    ember_kv_free(&c);
    ember_kv_free(&c2);
}

int main(void) {
    test_image_digest_prevents_aliasing();
    printf("ember kv_cache tests\n");
    test_longest_prefix();
    test_anchor_cut();
    test_turn_boundary_cut();
    test_recent_turn_boundary_cut();
    test_lru_eviction();
    test_disabled_and_invalid_cache();
    test_inflight_slot_lifecycle();
    printf("──────────────────────────────\n  %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
