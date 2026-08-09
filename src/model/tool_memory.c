// B3 — exact-token tool replay memory with optional model-scoped persistence.
// Accessed only on the generation worker (and once during startup), so no lock.
#include "tool_memory.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define ETM_HEADER_SIZE 48u
#define ETM_VERSION 2u
#define ETM_SUFFIX ".etm"
#define ETM_MAX_ID 128u

typedef struct {
    char *path;
    char *id;
    struct timespec mtime;
} persisted_candidate;

static uint32_t get_u32le(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 |
           (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static uint64_t get_u64le(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}

static void put_u32le(uint8_t *p, uint32_t v) {
    for (int i = 0; i < 4; ++i) p[i] = (uint8_t)(v >> (i * 8));
}

static void put_u64le(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; ++i) p[i] = (uint8_t)(v >> (i * 8));
}

static uint64_t hash_update(uint64_t h, const void *data, size_t n) {
    const uint8_t *p = (const uint8_t *)data;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static bool valid_persisted_id(const char *id) {
    size_t n = id ? strlen(id) : 0;
    if (n == 0 || n > ETM_MAX_ID) return false;
    for (size_t i = 0; i < n; ++i) {
        char c = id[i];
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '_' || c == '-'))
            return false;
    }
    return true;
}

static char *persisted_path(const ember_tool_memory *tm, const char *id) {
    if (!tm->persist_dir || !valid_persisted_id(id)) return NULL;
    size_t need = strlen(tm->persist_dir) + 1 + strlen(id) +
                  strlen(ETM_SUFFIX) + 1;
    char *path = (char *)malloc(need);
    if (!path) return NULL;
    snprintf(path, need, "%s/%s%s", tm->persist_dir, id, ETM_SUFFIX);
    return path;
}

static char *directory_child_path(const char *dir, const char *name) {
    size_t need = strlen(dir) + 1 + strlen(name) + 1;
    char *path = (char *)malloc(need);
    if (path) snprintf(path, need, "%s/%s", dir, name);
    return path;
}

static bool write_all_fd(int fd, const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    while (len > 0) {
        ssize_t n = write(fd, p, len);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return true;
}

static bool read_all_fd(int fd, void *data, size_t len) {
    uint8_t *p = (uint8_t *)data;
    while (len > 0) {
        ssize_t n = read(fd, p, len);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return true;
}

static size_t entry_bytes(const ember_tool_mem_entry *e) {
    return (e->id ? strlen(e->id) + 1 : 0) +
           (e->bytes ? e->len + 1 : 0) +
           (size_t)e->n_ids * sizeof(int32_t);
}

static void entry_clear(ember_tool_mem_entry *e) {
    free(e->id);
    free(e->bytes);
    free(e->ids);
    e->id = e->bytes = NULL;
    e->ids = NULL;
    e->len = 0;
    e->n_ids = 0;
}

static void persisted_delete(const ember_tool_memory *tm, const char *id) {
    char *path = persisted_path(tm, id);
    if (!path) return;
    (void)unlink(path);
    free(path);
}

static int find_idx(ember_tool_memory *tm, const char *id) {
    for (int i = 0; i < tm->n; ++i)
        if (tm->e[i].id && strcmp(tm->e[i].id, id) == 0) return i;
    return -1;
}

static void remove_idx(ember_tool_memory *tm, int i) {
    tm->bytes -= entry_bytes(&tm->e[i]);
    persisted_delete(tm, tm->e[i].id);
    entry_clear(&tm->e[i]);
    if (i != tm->n - 1) tm->e[i] = tm->e[tm->n - 1];
    memset(&tm->e[tm->n - 1], 0, sizeof(tm->e[tm->n - 1]));
    tm->n--;
}

static int lru_idx(const ember_tool_memory *tm) {
    int lru = 0;
    for (int i = 1; i < tm->n; ++i)
        if (tm->e[i].stamp < tm->e[lru].stamp) lru = i;
    return lru;
}

static uint64_t payload_hash(const char *id, const char *bytes, size_t len,
                             const int32_t *ids, int n_ids) {
    uint64_t h = UINT64_C(1469598103934665603);
    h = hash_update(h, id, strlen(id));
    h = hash_update(h, bytes, len);
    for (int i = 0; i < n_ids; ++i) {
        uint8_t le[4];
        put_u32le(le, (uint32_t)ids[i]);
        h = hash_update(h, le, sizeof(le));
    }
    return h;
}

static bool persist_write(const ember_tool_memory *tm,
                          const char *id, const char *bytes, size_t len,
                          const int32_t *ids, int n_ids) {
    char *path = persisted_path(tm, id);
    if (!path || n_ids < 0 || (uint64_t)n_ids > UINT32_MAX) {
        free(path);
        return false;
    }

    size_t tmp_need = strlen(path) + 48;
    char *tmp = (char *)malloc(tmp_need);
    if (!tmp) {
        free(path);
        return false;
    }
    snprintf(tmp, tmp_need, "%s.tmp.%ld.%lu", path, (long)getpid(),
             tm->clock);
    int fd = open(tmp, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) {
        free(tmp);
        free(path);
        return false;
    }

    uint8_t hdr[ETM_HEADER_SIZE] = {0};
    memcpy(hdr, "ETM1", 4);
    put_u32le(hdr + 4, ETM_VERSION);
    memcpy(hdr + 8, tm->persist_identity, 16);
    put_u64le(hdr + 24, (uint64_t)len);
    put_u32le(hdr + 32, (uint32_t)n_ids);
    put_u64le(hdr + 36, payload_hash(id, bytes, len, ids, n_ids));

    bool ok = write_all_fd(fd, hdr, sizeof(hdr)) &&
              write_all_fd(fd, bytes, len);
    for (int i = 0; ok && i < n_ids; ++i) {
        uint8_t le[4];
        put_u32le(le, (uint32_t)ids[i]);
        ok = write_all_fd(fd, le, sizeof(le));
    }
    if (ok) ok = fsync(fd) == 0;
    if (close(fd) != 0) ok = false;
    if (ok) ok = rename(tmp, path) == 0;
    if (ok) {
        int dfd = open(tm->persist_dir, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (dfd >= 0) {
            (void)fsync(dfd);
            close(dfd);
        }
    }
    if (!ok) (void)unlink(tmp);

    free(tmp);
    free(path);
    return ok;
}

static bool put_internal(ember_tool_memory *tm, const char *id,
                         const char *bytes, size_t len,
                         const int32_t *ids, int n_ids,
                         bool write_disk) {
    if (!tm->e || !id || !id[0] || !bytes || n_ids < 0) return false;
    size_t id_len = strlen(id);
    size_t token_bytes = (size_t)n_ids * sizeof(int32_t);
    if (len >= SIZE_MAX || (size_t)n_ids > SIZE_MAX / sizeof(int32_t) ||
        id_len >= SIZE_MAX || id_len + 1 > tm->max_bytes ||
        len + 1 > tm->max_bytes - (id_len + 1) ||
        token_bytes > tm->max_bytes - (id_len + 1) - (len + 1))
        return false;
    size_t need = id_len + 1 + len + 1 + token_bytes;

    char *b = (char *)malloc(len + 1);
    if (!b) return false;
    memcpy(b, bytes, len);
    b[len] = '\0';

    int32_t *ic = NULL;
    if (ids && n_ids > 0) {
        ic = (int32_t *)malloc((size_t)n_ids * sizeof(int32_t));
        if (!ic) {
            free(b);
            return false;
        }
        memcpy(ic, ids, (size_t)n_ids * sizeof(int32_t));
    } else {
        n_ids = 0;
    }

    char *idc = strdup(id);
    if (!idc) {
        free(b);
        free(ic);
        return false;
    }
    int old = find_idx(tm, id);
    if (old >= 0) remove_idx(tm, old);
    while (tm->n > 0 &&
           (tm->n >= tm->cap || tm->bytes + need > tm->max_bytes))
        remove_idx(tm, lru_idx(tm));

    int i = tm->n++;
    tm->e[i].id = idc;
    tm->e[i].bytes = b;
    tm->e[i].len = len;
    tm->e[i].ids = ic;
    tm->e[i].n_ids = n_ids;
    tm->e[i].stamp = ++tm->clock;
    tm->bytes += need;

    if (write_disk && tm->persist_dir &&
        !persist_write(tm, id, bytes, len, ids, n_ids)) {
        tm->persist_failures++;
        if (tm->persist_failures == 1)
            fprintf(stderr,
                    "[ember] warning: failed to persist exact tool replay;"
                    " continuing in RAM only for affected calls\n");
    }
    return true;
}

void ember_tool_memory_init(ember_tool_memory *tm, int max_entries,
                            size_t max_bytes) {
    memset(tm, 0, sizeof(*tm));
    tm->cap = max_entries > 0 ? max_entries : 512;
    tm->e = (ember_tool_mem_entry *)calloc((size_t)tm->cap, sizeof(*tm->e));
    tm->max_bytes = max_bytes > 0 ? max_bytes : 64u * 1024u * 1024u;
}

void ember_tool_memory_free(ember_tool_memory *tm) {
    if (!tm) return;
    if (tm->e) {
        for (int i = 0; i < tm->n; ++i) entry_clear(&tm->e[i]);
    }
    free(tm->e);
    free(tm->persist_dir);
    memset(tm, 0, sizeof(*tm));
}

static int candidate_cmp(const void *a_, const void *b_) {
    const persisted_candidate *a = (const persisted_candidate *)a_;
    const persisted_candidate *b = (const persisted_candidate *)b_;
    if (a->mtime.tv_sec != b->mtime.tv_sec)
        return a->mtime.tv_sec < b->mtime.tv_sec ? -1 : 1;
    if (a->mtime.tv_nsec != b->mtime.tv_nsec)
        return a->mtime.tv_nsec < b->mtime.tv_nsec ? -1 : 1;
    return strcmp(a->id, b->id);
}

static bool load_persisted_file(ember_tool_memory *tm,
                                const char *path, const char *id) {
    int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return false;
    struct stat st;
    uint8_t hdr[ETM_HEADER_SIZE];
    bool ok = fstat(fd, &st) == 0 && S_ISREG(st.st_mode) &&
              read_all_fd(fd, hdr, sizeof(hdr));
    if (!ok || memcmp(hdr, "ETM1", 4) != 0 ||
        get_u32le(hdr + 4) != ETM_VERSION ||
        memcmp(hdr + 8, tm->persist_identity, 16) != 0) {
        close(fd);
        return false;
    }

    uint64_t raw64 = get_u64le(hdr + 24);
    uint32_t n_ids32 = get_u32le(hdr + 32);
    uint64_t expected_hash = get_u64le(hdr + 36);
    if (raw64 > tm->max_bytes ||
        n_ids32 > INT_MAX ||
        (uint64_t)n_ids32 > tm->max_bytes / sizeof(int32_t) ||
        raw64 > UINT64_MAX - ETM_HEADER_SIZE - (uint64_t)n_ids32 * 4u ||
        (uint64_t)st.st_size != ETM_HEADER_SIZE + raw64 +
                                     (uint64_t)n_ids32 * 4u) {
        close(fd);
        return false;
    }

    size_t raw_len = (size_t)raw64;
    char *raw = (char *)malloc(raw_len + 1);
    uint8_t *token_bytes =
        n_ids32 ? (uint8_t *)malloc((size_t)n_ids32 * 4u) : NULL;
    if (!raw || (n_ids32 && !token_bytes)) {
        free(raw);
        free(token_bytes);
        close(fd);
        return false;
    }
    ok = read_all_fd(fd, raw, raw_len) &&
         (!n_ids32 || read_all_fd(fd, token_bytes, (size_t)n_ids32 * 4u));
    close(fd);
    if (!ok) {
        free(raw);
        free(token_bytes);
        return false;
    }
    raw[raw_len] = '\0';

    uint64_t h = UINT64_C(1469598103934665603);
    h = hash_update(h, id, strlen(id));
    h = hash_update(h, raw, raw_len);
    h = hash_update(h, token_bytes, (size_t)n_ids32 * 4u);
    if (h != expected_hash) {
        free(raw);
        free(token_bytes);
        return false;
    }

    int32_t *ids = n_ids32
                       ? (int32_t *)malloc((size_t)n_ids32 * sizeof(int32_t))
                       : NULL;
    if (n_ids32 && !ids) {
        free(raw);
        free(token_bytes);
        return false;
    }
    for (uint32_t i = 0; i < n_ids32; ++i)
        ids[i] = (int32_t)get_u32le(token_bytes + (size_t)i * 4u);
    bool inserted = put_internal(tm, id, raw, raw_len, ids, (int)n_ids32, false);
    free(ids);
    free(raw);
    free(token_bytes);
    return inserted;
}

int ember_tool_memory_enable_persistence(ember_tool_memory *tm,
                                         const char *root_dir,
                                         const uint8_t identity[16]) {
    if (!tm || !tm->e || !root_dir || !root_dir[0] || !identity ||
        tm->persist_dir)
        return -1;

    char hex[33];
    static const char digits[] = "0123456789abcdef";
    for (int i = 0; i < 16; ++i) {
        hex[i * 2] = digits[identity[i] >> 4];
        hex[i * 2 + 1] = digits[identity[i] & 15];
    }
    hex[32] = '\0';

    size_t need = strlen(root_dir) + strlen("/tool-memory-") + sizeof(hex);
    char *dir = (char *)malloc(need);
    if (!dir) return -1;
    snprintf(dir, need, "%s/tool-memory-%s", root_dir, hex);
    if (mkdir(dir, 0700) != 0 && errno != EEXIST) {
        free(dir);
        return -1;
    }
    struct stat dst;
    if (lstat(dir, &dst) != 0 || !S_ISDIR(dst.st_mode) || S_ISLNK(dst.st_mode)) {
        free(dir);
        return -1;
    }
    // Persisted tool arguments and outputs may contain secrets. Refuse a
    // directory owned by another account and normalize permissions even when
    // it pre-existed with a permissive umask.
    if (dst.st_uid != geteuid() || chmod(dir, 0700) != 0) {
        free(dir);
        return -1;
    }
    tm->persist_dir = dir;
    memcpy(tm->persist_identity, identity, 16);

    DIR *d = opendir(dir);
    if (!d) {
        free(tm->persist_dir);
        tm->persist_dir = NULL;
        return -1;
    }
    persisted_candidate *v = NULL;
    size_t n = 0, cap = 0;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        size_t name_len = strlen(de->d_name);
        size_t suffix_len = strlen(ETM_SUFFIX);
        // Atomic writes may leave an unlinked temp only when the process dies.
        // Remove those bounded-directory leftovers on the next clean start.
        if (strstr(de->d_name, ETM_SUFFIX ".tmp.") != NULL) {
            char *tmp = directory_child_path(dir, de->d_name);
            struct stat tst;
            if (tmp && lstat(tmp, &tst) == 0 && S_ISREG(tst.st_mode) &&
                !S_ISLNK(tst.st_mode))
                (void)unlink(tmp);
            free(tmp);
            continue;
        }
        if (name_len <= suffix_len ||
            strcmp(de->d_name + name_len - suffix_len, ETM_SUFFIX) != 0)
            continue;
        size_t id_len = name_len - suffix_len;
        if (id_len > ETM_MAX_ID) continue;
        char id[ETM_MAX_ID + 1];
        memcpy(id, de->d_name, id_len);
        id[id_len] = '\0';
        if (!valid_persisted_id(id)) continue;

        char *path = persisted_path(tm, id);
        struct stat st;
        if (!path || lstat(path, &st) != 0 || !S_ISREG(st.st_mode) ||
            S_ISLNK(st.st_mode)) {
            free(path);
            continue;
        }
        if (n == cap) {
            size_t next = cap ? cap * 2 : 32;
            persisted_candidate *grown =
                (persisted_candidate *)realloc(v, next * sizeof(*v));
            if (!grown) {
                free(path);
                break;
            }
            v = grown;
            cap = next;
        }
        v[n].path = path;
        v[n].id = strdup(id);
        v[n].mtime = st.st_mtim;
        if (!v[n].id) {
            free(path);
            break;
        }
        n++;
    }
    closedir(d);

    if (n > 1)
        qsort(v, n, sizeof(*v), candidate_cmp);  // oldest first → newest survives
    for (size_t i = 0; i < n; ++i) {
        if (!load_persisted_file(tm, v[i].path, v[i].id))
            (void)unlink(v[i].path);  // self-heal corrupt/incompatible records
        free(v[i].path);
        free(v[i].id);
    }
    free(v);
    return tm->n;
}

bool ember_tool_memory_persistence_enabled(const ember_tool_memory *tm) {
    return tm && tm->persist_dir != NULL;
}

void ember_tool_memory_put(ember_tool_memory *tm, const char *id,
                           const char *bytes, size_t len,
                           const int32_t *ids, int n_ids) {
    (void)put_internal(tm, id, bytes, len, ids, n_ids, true);
}

static void persisted_touch(const ember_tool_memory *tm, const char *id) {
    char *path = persisted_path(tm, id);
    if (!path) return;
    (void)utimensat(AT_FDCWD, path, NULL, AT_SYMLINK_NOFOLLOW);
    free(path);
}

const char *ember_tool_memory_get(ember_tool_memory *tm, const char *id) {
    if (!tm || !tm->e || !id || !id[0]) return NULL;
    int i = find_idx(tm, id);
    if (i < 0) return NULL;
    tm->e[i].stamp = ++tm->clock;
    persisted_touch(tm, id);
    return tm->e[i].bytes;
}

const int32_t *ember_tool_memory_get_tokens(ember_tool_memory *tm,
                                            const char *id, int *n) {
    if (n) *n = 0;
    if (!tm || !tm->e || !id || !id[0]) return NULL;
    int i = find_idx(tm, id);
    if (i < 0 || tm->e[i].n_ids <= 0) return NULL;
    tm->e[i].stamp = ++tm->clock;
    persisted_touch(tm, id);
    if (n) *n = tm->e[i].n_ids;
    return tm->e[i].ids;
}
