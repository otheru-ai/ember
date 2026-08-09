#include "../src/model/continuation.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int pass, fail;
#define CHECK(x) do { if (x) pass++; else { fail++; fprintf(stderr, \
    "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); } } while (0)

static void test_sets_and_replacement(void) {
    ember_continuation_store s;
    ember_continuation_init(&s, 4, 4096);
    const char *ids[] = {"call_a", "call_b"};
    int32_t tok[] = {1, 2, 3};
    CHECK(ember_continuation_put(&s, EMBER_API_CHAT, ids, 2, tok, 3,
                                 "visible", "tools"));
    const char *reverse[] = {"call_b", "call_a"};
    const ember_continuation_entry *e =
        ember_continuation_get(&s, EMBER_API_CHAT, reverse, 2);
    CHECK(e && e->n_frontier == 3 && e->frontier_ids[2] == 3);
    CHECK(!ember_continuation_get(&s, EMBER_API_RESPONSES, reverse, 2));
    const char *partial[] = {"call_a"};
    CHECK(!ember_continuation_get(&s, EMBER_API_CHAT, partial, 1));

    int32_t newer[] = {9, 8};
    CHECK(ember_continuation_put(&s, EMBER_API_CHAT, partial, 1,
                                 newer, 2, NULL, NULL));
    CHECK(!ember_continuation_get(&s, EMBER_API_CHAT, reverse, 2));
    e = ember_continuation_get(&s, EMBER_API_CHAT, partial, 1);
    CHECK(e && e->n_frontier == 2 && e->frontier_ids[0] == 9);
    ember_continuation_free(&s);
}

static void test_persistence(void) {
    char root[] = "/tmp/ember-continuation-XXXXXX";
    CHECK(mkdtemp(root) != NULL);
    uint8_t identity[16] = {1, 2, 3, 4};
    const char *ids[] = {"call_persist_1", "call_persist_2"};
    int32_t tok[] = {100, -2, 300, 400};

    ember_continuation_store a;
    ember_continuation_init(&a, 8, 65536);
    CHECK(ember_continuation_enable_persistence(&a, root, identity) == 0);
    CHECK(ember_continuation_put(&a, EMBER_API_RESPONSES, ids, 2,
                                 tok, 4, "shown", "[{\"type\":\"function\"}]"));
    char dir[PATH_MAX];
    snprintf(dir, sizeof(dir), "%s", a.persist_dir);
    ember_continuation_free(&a);

    ember_continuation_store b;
    ember_continuation_init(&b, 8, 65536);
    CHECK(ember_continuation_enable_persistence(&b, root, identity) == 1);
    const ember_continuation_entry *e =
        ember_continuation_get(&b, EMBER_API_RESPONSES, ids, 2);
    CHECK(e && e->n_frontier == 4 && e->frontier_ids[1] == -2);
    CHECK(e && !strcmp(e->visible_text, "shown"));
    CHECK(e && strstr(e->tools_json, "function"));

    size_t path_len = strlen(dir) + strlen(ids[0]) + 6;
    char *path = (char *)malloc(path_len);
    CHECK(path != NULL);
    if (!path) {
        ember_continuation_free(&b);
        return;
    }
    snprintf(path, path_len, "%s/%s.ect", dir, ids[0]);
    int fd = open(path, O_WRONLY | O_TRUNC);
    CHECK(fd >= 0);
    if (fd >= 0) close(fd);
    ember_continuation_free(&b);

    ember_continuation_store c;
    ember_continuation_init(&c, 8, 65536);
    CHECK(ember_continuation_enable_persistence(&c, root, identity) == 0);
    CHECK(access(path, F_OK) != 0 && errno == ENOENT);
    ember_continuation_free(&c);
    free(path);
    rmdir(dir);
    rmdir(root);
}

static void test_more_than_sixteen_calls(void) {
    enum { N = 20 };
    char root[] = "/tmp/ember-continuation-many-XXXXXX";
    CHECK(mkdtemp(root) != NULL);
    uint8_t identity[16] = {9, 8, 7, 6};
    ember_continuation_store s;
    ember_continuation_init(&s, 4, 65536);
    CHECK(ember_continuation_enable_persistence(&s, root, identity) == 0);
    const char *ids[N];
    char storage[N][24];
    for (int i = 0; i < N; ++i) {
        snprintf(storage[i], sizeof(storage[i]), "call_many_%d", i);
        ids[i] = storage[i];
    }
    int32_t frontier[] = {1, 2, 3};
    CHECK(ember_continuation_put(&s, EMBER_API_RESPONSES, ids, N,
                                 frontier, 3, NULL, "[]"));
    const ember_continuation_entry *e =
        ember_continuation_get(&s, EMBER_API_RESPONSES, ids, N);
    CHECK(e && e->n_call_ids == N &&
          !strcmp(e->call_ids[N - 1], "call_many_19"));
    char dir[PATH_MAX];
    snprintf(dir, sizeof(dir), "%s", s.persist_dir);
    ember_continuation_free(&s);

    ember_continuation_store reloaded;
    ember_continuation_init(&reloaded, 4, 65536);
    CHECK(ember_continuation_enable_persistence(
              &reloaded, root, identity) == 1);
    e = ember_continuation_get(
        &reloaded, EMBER_API_RESPONSES, ids, N);
    CHECK(e && e->n_call_ids == N && e->frontier_ids[2] == 3);
    ember_continuation_free(&reloaded);

    size_t path_len = strlen(dir) + strlen(ids[0]) + 6;
    char *path = malloc(path_len);
    CHECK(path != NULL);
    if (path) {
        snprintf(path, path_len, "%s/%s.ect", dir, ids[0]);
        unlink(path);
        free(path);
    }
    rmdir(dir);
    rmdir(root);
}

int main(void) {
    test_sets_and_replacement();
    test_persistence();
    test_more_than_sixteen_calls();
    printf("continuation tests: %d passed, %d failed\n", pass, fail);
    return fail != 0;
}
