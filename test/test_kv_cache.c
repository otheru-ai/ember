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
    ember_kv_commit(&c, slot, p1, 4);            // cache prefix [U,10,11,12]

    // a longer prompt extending that prefix → hit at len 4
    int32_t p2[] = {U, 10, 11, 12, A, 20, E, U, 30};
    int s, len;
    ember_kv_lookup(&c, p2, 9, &s, &len);
    CHECK(s == slot && len == 4, "longest-prefix hit reuses the cached slot");

    // a divergent prompt → no hit
    int32_t p3[] = {U, 99};
    ember_kv_lookup(&c, p3, 2, &s, &len);
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
    int cut = ember_kv_snap_cut(&c, p, 7);
    CHECK(cut == 4, "anchor = last user marker before first assistant");
    ember_kv_free(&c);
}

static void test_turn_boundary_cut(void) {
    ember_kv_cache c;
    ember_kv_init(&c, 32, U, A, E);
    // [U q A a E U q2 A a2 E U q3]  boundaries after each E→role
    // eos at idx 4 → role U at 5 → cut 6 ; eos at idx 10 → role U at 11 → cut 12
    int32_t p[] = {U,10,A,20,E, U,11,A,21,E, U,12};  // 12 tokens
    int cut = ember_kv_snap_cut(&c, p, 12);
    // two boundaries (cut 5 and 10... recompute): eos idx4→U idx5→cut6; eos idx9→U idx10→cut11
    CHECK(cut == 6, "snap at second-to-last boundary (through last completed exchange)");
    ember_kv_free(&c);
}

static void test_lru_eviction(void) {
    ember_kv_cache c;
    ember_kv_init(&c, 2, U, A, E);  // only 2 slots
    int32_t a[] = {U, 1};
    int32_t b[] = {U, 2};
    int32_t d[] = {U, 3};
    int s;
    s = ember_kv_reserve(&c); ember_kv_commit(&c, s, a, 2);
    s = ember_kv_reserve(&c); ember_kv_commit(&c, s, b, 2);
    // touch 'a' so 'b' is LRU
    int slot, len; ember_kv_lookup(&c, a, 2, &slot, &len);
    // reserve a third → must evict (prefer LRU leaf = b)
    s = ember_kv_reserve(&c); ember_kv_commit(&c, s, d, 2);
    // 'a' should still be present, 'b' evicted
    ember_kv_lookup(&c, a, 2, &slot, &len);
    CHECK(slot >= 0 && len == 2, "recently-used entry survives eviction");
    ember_kv_lookup(&c, b, 2, &slot, &len);
    CHECK(slot == -1, "LRU entry evicted at capacity");
    CHECK(c.n_entries == 2, "cache stays within capacity");
    ember_kv_free(&c);
}

int main(void) {
    printf("ember kv_cache tests\n");
    test_longest_prefix();
    test_anchor_cut();
    test_turn_boundary_cut();
    test_lru_eviction();
    printf("──────────────────────────────\n  %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
