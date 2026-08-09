#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../src/model/tool_memory.h"
#include "../src/model/tool_parser.h"

static int failures;
#define CHECK(cond) do { if (!(cond)) { \
    printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; \
} } while (0)

static char *join_path(const char *a, const char *b) {
    size_t n = strlen(a) + strlen(b) + 2;
    char *path = malloc(n);
    if (path) snprintf(path, n, "%s/%s", a, b);
    return path;
}

static void cleanup_tree(const char *root) {
    DIR *d = opendir(root);
    if (!d) return;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        if (de->d_name[0] == '.') continue;
        char *dir = join_path(root, de->d_name);
        if (!dir) continue;
        DIR *sub = opendir(dir);
        if (sub) {
            struct dirent *se;
            while ((se = readdir(sub)) != NULL) {
                if (se->d_name[0] == '.') continue;
                char *path = join_path(dir, se->d_name);
                if (path) {
                    unlink(path);
                    free(path);
                }
            }
            closedir(sub);
            rmdir(dir);
        } else {
            unlink(dir);
        }
        free(dir);
    }
    closedir(d);
    rmdir(root);
}

static void test_persistence(void) {
    char root[] = "/tmp/ember-tool-memory-test.XXXXXX";
    CHECK(mkdtemp(root) != NULL);
    uint8_t identity[16], other_identity[16];
    for (int i = 0; i < 16; i++) {
        identity[i] = (uint8_t)(i + 1);
        other_identity[i] = (uint8_t)(0xf0 + i);
    }
    const char *id1 = "call_00112233445566778899aabbccddeeff";
    const char *id2 = "call_ffeeddccbbaa99887766554433221100";
    const char raw[] =
        "<?DSML?tool_calls>"
        "<?DSML?invoke name=\"weather\">"
        "<?DSML?parameter name=\"city\" string=\"true\">Paris</?DSML?parameter>"
        "</?DSML?invoke>"
        "</?DSML?tool_calls>";
    int32_t ids[] = {7, -2, 400000, 9};

    ember_tool_memory src;
    ember_tool_memory_init(&src, 8, 4096);
    CHECK(ember_tool_memory_enable_persistence(&src, root, identity) == 0);
    CHECK(ember_tool_memory_persistence_enabled(&src));
    struct stat persist_st;
    CHECK(stat(src.persist_dir, &persist_st) == 0);
    CHECK((persist_st.st_mode & 0777) == 0700);
    ember_tool_memory_put(&src, id1, raw, strlen(raw), ids, 4);
    char persisted[PATH_MAX], stale_tmp[PATH_MAX];
    snprintf(persisted, sizeof(persisted), "%s/%s.etm", src.persist_dir, id1);
    snprintf(stale_tmp, sizeof(stale_tmp), "%s/%s.etm.tmp.crash",
             src.persist_dir, id2);
    CHECK(access(persisted, F_OK) == 0);
    int tmpfd = open(stale_tmp, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    CHECK(tmpfd >= 0);
    if (tmpfd >= 0) {
        CHECK(write(tmpfd, "partial", 7) == 7);
        close(tmpfd);
    }
    ember_tool_memory_free(&src);

    ember_tool_memory restored;
    ember_tool_memory_init(&restored, 8, 4096);
    CHECK(ember_tool_memory_enable_persistence(&restored, root, identity) == 1);
    CHECK(access(stale_tmp, F_OK) != 0 && errno == ENOENT);
    CHECK(strcmp(ember_tool_memory_get(&restored, id1), raw) == 0);
    int n = 0;
    const int32_t *got = ember_tool_memory_get_tokens(&restored, id1, &n);
    CHECK(n == 4 && got && memcmp(got, ids, sizeof(ids)) == 0);
    ember_tool_calls expected = {0};
    expected.calls = calloc(1, sizeof(*expected.calls));
    expected.len = expected.cap = 1;
    expected.calls[0].name = strdup("weather");
    expected.calls[0].arguments = strdup("{\"city\":\"Paris\"}");
    CHECK(ember_tool_calls_match_raw(ember_tool_memory_get(&restored, id1),
                                     &expected));
    ember_tool_calls_free(&expected);
    ember_tool_memory_free(&restored);

    // A different model identity gets an independent namespace.
    ember_tool_memory isolated;
    ember_tool_memory_init(&isolated, 8, 4096);
    CHECK(ember_tool_memory_enable_persistence(
              &isolated, root, other_identity) == 0);
    CHECK(ember_tool_memory_get(&isolated, id1) == NULL);
    ember_tool_memory_free(&isolated);

    // Corruption is rejected and self-healed instead of reaching the renderer.
    int fd = open(persisted, O_WRONLY | O_TRUNC);
    CHECK(fd >= 0);
    if (fd >= 0) {
        CHECK(write(fd, "bad", 3) == 3);
        close(fd);
    }
    ember_tool_memory corrupt;
    ember_tool_memory_init(&corrupt, 8, 4096);
    CHECK(ember_tool_memory_enable_persistence(&corrupt, root, identity) == 0);
    CHECK(ember_tool_memory_get(&corrupt, id1) == NULL);
    CHECK(access(persisted, F_OK) != 0 && errno == ENOENT);

    // Disk state follows the same count-bounded LRU as RAM.
    corrupt.cap = 1;
    ember_tool_memory_put(&corrupt, id1, raw, strlen(raw), ids, 4);
    char first[PATH_MAX], second[PATH_MAX];
    snprintf(first, sizeof(first), "%s/%s.etm", corrupt.persist_dir, id1);
    snprintf(second, sizeof(second), "%s/%s.etm", corrupt.persist_dir, id2);
    ember_tool_memory_put(&corrupt, id2, raw, strlen(raw), ids, 4);
    CHECK(access(first, F_OK) != 0);
    CHECK(access(second, F_OK) == 0);
    ember_tool_memory_free(&corrupt);

    cleanup_tree(root);
}

int main(void) {
    ember_tool_memory tm;
    ember_tool_memory_init(&tm, 8, 120);
    int32_t ids[] = {1, 2, 3, 4};
    const char raw[] = "0123456789012345678901234567890123456789";

    ember_tool_memory_put(&tm, "call_a", raw, strlen(raw), ids, 4);
    CHECK(ember_tool_memory_get(&tm, "call_a") != NULL);
    ember_tool_memory_put(&tm, "call_b", raw, strlen(raw), ids, 4);
    CHECK(ember_tool_memory_get(&tm, "call_a") == NULL);
    CHECK(ember_tool_memory_get(&tm, "call_b") != NULL);
    CHECK(tm.bytes <= tm.max_bytes);

    char oversized[256];
    memset(oversized, 'x', sizeof(oversized));
    ember_tool_memory_put(&tm, "call_large", oversized, sizeof(oversized), ids, 4);
    CHECK(ember_tool_memory_get(&tm, "call_large") == NULL);
    CHECK(ember_tool_memory_get(&tm, "call_b") != NULL);

    ember_tool_memory_free(&tm);
    test_persistence();
    if (!failures) printf("tool memory: all passed\n");
    return failures ? 1 : 0;
}
