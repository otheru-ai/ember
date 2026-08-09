#include "continuation.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define ECT_HEADER_SIZE 64u
#define ECT_VERSION 1u
#define ECT_SUFFIX ".ect"
#define ECT_MAX_ID 128u

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

static bool valid_id(const char *id) {
    size_t n = id ? strlen(id) : 0;
    if (n == 0 || n > ECT_MAX_ID) return false;
    for (size_t i = 0; i < n; ++i) {
        char c = id[i];
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '_' || c == '-'))
            return false;
    }
    return true;
}

static bool write_all(int fd, const void *data, size_t n) {
    const uint8_t *p = (const uint8_t *)data;
    while (n) {
        ssize_t k = write(fd, p, n);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) return false;
        p += (size_t)k;
        n -= (size_t)k;
    }
    return true;
}

static bool read_all(int fd, void *data, size_t n) {
    uint8_t *p = (uint8_t *)data;
    while (n) {
        ssize_t k = read(fd, p, n);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) return false;
        p += (size_t)k;
        n -= (size_t)k;
    }
    return true;
}

static char *child_path(const char *dir, const char *name) {
    size_t n = strlen(dir) + strlen(name) + 2;
    char *p = (char *)malloc(n);
    if (p) snprintf(p, n, "%s/%s", dir, name);
    return p;
}

static char *entry_path(const ember_continuation_store *s, const char *id) {
    if (!s->persist_dir || !valid_id(id)) return NULL;
    size_t n = strlen(s->persist_dir) + strlen(id) + strlen(ECT_SUFFIX) + 2;
    char *p = (char *)malloc(n);
    if (p) snprintf(p, n, "%s/%s%s", s->persist_dir, id, ECT_SUFFIX);
    return p;
}

static size_t entry_bytes(const ember_continuation_entry *e) {
    size_t n = (size_t)e->n_call_ids * sizeof(char *) +
               (size_t)e->n_frontier * sizeof(int32_t);
    for (int i = 0; i < e->n_call_ids; ++i)
        n += strlen(e->call_ids[i]) + 1;
    if (e->visible_text) n += strlen(e->visible_text) + 1;
    if (e->tools_json) n += strlen(e->tools_json) + 1;
    return n;
}

static void entry_clear(ember_continuation_entry *e) {
    if (!e) return;
    for (int i = 0; i < e->n_call_ids; ++i) free(e->call_ids[i]);
    free(e->call_ids);
    free(e->frontier_ids);
    free(e->visible_text);
    free(e->tools_json);
    memset(e, 0, sizeof(*e));
}

static bool has_id(const ember_continuation_entry *e, const char *id) {
    for (int i = 0; i < e->n_call_ids; ++i)
        if (strcmp(e->call_ids[i], id) == 0) return true;
    return false;
}

static bool same_set(const ember_continuation_entry *e,
                     const char *const *ids, int n) {
    if (!e || e->n_call_ids != n) return false;
    for (int i = 0; i < n; ++i)
        if (!ids[i] || !has_id(e, ids[i])) return false;
    return true;
}

static void delete_disk(const ember_continuation_store *s,
                        const ember_continuation_entry *e) {
    if (!e || e->n_call_ids == 0) return;
    char *p = entry_path(s, e->call_ids[0]);
    if (p) {
        (void)unlink(p);
        free(p);
    }
}

static void remove_idx(ember_continuation_store *s, int idx, bool disk) {
    if (disk) delete_disk(s, &s->entries[idx]);
    s->bytes -= entry_bytes(&s->entries[idx]);
    entry_clear(&s->entries[idx]);
    if (idx != s->n - 1) s->entries[idx] = s->entries[s->n - 1];
    memset(&s->entries[s->n - 1], 0, sizeof(s->entries[s->n - 1]));
    s->n--;
}

static int lru_idx(const ember_continuation_store *s) {
    int idx = 0;
    for (int i = 1; i < s->n; ++i)
        if (s->entries[i].stamp < s->entries[idx].stamp) idx = i;
    return idx;
}

static uint64_t payload_hash(ember_api_kind api,
                             const char *const *call_ids, int n_call_ids,
                             const int32_t *frontier, int n_frontier,
                             const char *visible, const char *tools) {
    uint64_t h = UINT64_C(1469598103934665603);
    uint8_t le[4];
    put_u32le(le, (uint32_t)api);
    h = hash_update(h, le, sizeof(le));
    for (int i = 0; i < n_call_ids; ++i) {
        uint8_t lenle[4];
        put_u32le(lenle, (uint32_t)strlen(call_ids[i]));
        h = hash_update(h, lenle, sizeof(lenle));
        h = hash_update(h, call_ids[i], strlen(call_ids[i]));
    }
    for (int i = 0; i < n_frontier; ++i) {
        put_u32le(le, (uint32_t)frontier[i]);
        h = hash_update(h, le, sizeof(le));
    }
    if (visible) h = hash_update(h, visible, strlen(visible));
    if (tools) h = hash_update(h, tools, strlen(tools));
    return h;
}

static bool persist_write(const ember_continuation_store *s,
                          const ember_continuation_entry *e) {
    if (!s->persist_dir || e->n_call_ids <= 0) return true;
    char *path = entry_path(s, e->call_ids[0]);
    if (!path) return false;
    size_t tn = strlen(path) + 48;
    char *tmp = (char *)malloc(tn);
    if (!tmp) {
        free(path);
        return false;
    }
    snprintf(tmp, tn, "%s.tmp.%ld.%lu", path, (long)getpid(), s->clock);
    int fd = open(tmp, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) {
        free(tmp);
        free(path);
        return false;
    }

    const char *visible = e->visible_text ? e->visible_text : "";
    const char *tools = e->tools_json ? e->tools_json : "";
    uint8_t hdr[ECT_HEADER_SIZE] = {0};
    memcpy(hdr, "ECT1", 4);
    put_u32le(hdr + 4, ECT_VERSION);
    memcpy(hdr + 8, s->persist_identity, 16);
    put_u32le(hdr + 24, (uint32_t)e->api);
    put_u32le(hdr + 28, (uint32_t)e->n_call_ids);
    put_u32le(hdr + 32, (uint32_t)e->n_frontier);
    put_u64le(hdr + 40, strlen(visible));
    put_u64le(hdr + 48, strlen(tools));
    put_u64le(hdr + 56,
              payload_hash(e->api, (const char *const *)e->call_ids,
                           e->n_call_ids, e->frontier_ids, e->n_frontier,
                           visible, tools));
    bool ok = write_all(fd, hdr, sizeof(hdr));
    for (int i = 0; ok && i < e->n_call_ids; ++i) {
        uint8_t lenle[4];
        put_u32le(lenle, (uint32_t)strlen(e->call_ids[i]));
        ok = write_all(fd, lenle, sizeof(lenle)) &&
             write_all(fd, e->call_ids[i], strlen(e->call_ids[i]));
    }
    for (int i = 0; ok && i < e->n_frontier; ++i) {
        uint8_t le[4];
        put_u32le(le, (uint32_t)e->frontier_ids[i]);
        ok = write_all(fd, le, sizeof(le));
    }
    if (ok) ok = write_all(fd, visible, strlen(visible));
    if (ok) ok = write_all(fd, tools, strlen(tools));
    if (ok) ok = fsync(fd) == 0;
    if (close(fd) != 0) ok = false;
    if (ok) ok = rename(tmp, path) == 0;
    if (ok) {
        int dfd = open(s->persist_dir, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
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

static bool put_internal(ember_continuation_store *s, ember_api_kind api,
                         const char *const *call_ids, int n_call_ids,
                         const int32_t *frontier_ids, int n_frontier,
                         const char *visible_text, const char *tools_json,
                         bool write_disk) {
    if (!s || !s->entries || n_call_ids <= 0 ||
        n_frontier <= 0 || !frontier_ids)
        return false;
    if ((size_t)n_call_ids > s->max_bytes / sizeof(char *)) return false;
    size_t need = (size_t)n_call_ids * sizeof(char *);
    if ((size_t)n_frontier > (s->max_bytes - need) / sizeof(int32_t))
        return false;
    need += (size_t)n_frontier * sizeof(int32_t);
    for (int i = 0; i < n_call_ids; ++i) {
        if (!valid_id(call_ids[i])) return false;
        for (int j = 0; j < i; ++j)
            if (strcmp(call_ids[i], call_ids[j]) == 0) return false;
        size_t id_bytes = strlen(call_ids[i]) + 1;
        if (id_bytes > s->max_bytes - need) return false;
        need += id_bytes;
    }
    if (visible_text) {
        size_t bytes = strlen(visible_text) + 1;
        if (bytes > s->max_bytes - need) return false;
        need += bytes;
    }
    if (tools_json) {
        size_t bytes = strlen(tools_json) + 1;
        if (bytes > s->max_bytes - need) return false;
        need += bytes;
    }

    ember_continuation_entry e = {0};
    e.api = api;
    e.n_call_ids = n_call_ids;
    e.n_frontier = n_frontier;
    e.call_ids = (char **)calloc((size_t)n_call_ids, sizeof(char *));
    e.frontier_ids =
        (int32_t *)malloc((size_t)n_frontier * sizeof(int32_t));
    if (!e.call_ids || !e.frontier_ids) {
        entry_clear(&e);
        return false;
    }
    for (int i = 0; i < n_call_ids; ++i) {
        e.call_ids[i] = strdup(call_ids[i]);
        if (!e.call_ids[i]) {
            entry_clear(&e);
            return false;
        }
    }
    memcpy(e.frontier_ids, frontier_ids,
           (size_t)n_frontier * sizeof(int32_t));
    if (visible_text && !(e.visible_text = strdup(visible_text))) {
        entry_clear(&e);
        return false;
    }
    if (tools_json && !(e.tools_json = strdup(tools_json))) {
        entry_clear(&e);
        return false;
    }

    // One call id can name only one authoritative frontier.
    const bool delete_evictions = s->persist_dir != NULL;
    for (int i = s->n - 1; i >= 0; --i) {
        bool overlap = false;
        for (int j = 0; j < n_call_ids && !overlap; ++j)
            overlap = has_id(&s->entries[i], call_ids[j]);
        if (overlap) remove_idx(s, i, delete_evictions);
    }
    while (s->n > 0 &&
           (s->n >= s->cap || s->bytes + need > s->max_bytes))
        remove_idx(s, lru_idx(s), delete_evictions);
    if (s->n >= s->cap) {
        entry_clear(&e);
        return false;
    }
    e.stamp = ++s->clock;
    s->entries[s->n++] = e;
    s->bytes += need;
    if (write_disk && !persist_write(s, &e)) {
        s->persist_failures++;
        if (s->persist_failures == 1)
            fprintf(stderr,
                    "[ember] warning: failed to persist continuation binding;"
                    " continuing with the RAM binding\n");
    }
    return true;
}

void ember_continuation_init(ember_continuation_store *s, int max_entries,
                             size_t max_bytes) {
    memset(s, 0, sizeof(*s));
    s->cap = max_entries > 0 ? max_entries : 64;
    s->max_bytes = max_bytes > 0 ? max_bytes : 128u * 1024u * 1024u;
    s->entries =
        (ember_continuation_entry *)calloc((size_t)s->cap, sizeof(*s->entries));
}

void ember_continuation_free(ember_continuation_store *s) {
    if (!s) return;
    for (int i = 0; i < s->n; ++i) entry_clear(&s->entries[i]);
    free(s->entries);
    free(s->persist_dir);
    memset(s, 0, sizeof(*s));
}

bool ember_continuation_put(ember_continuation_store *s, ember_api_kind api,
                            const char *const *call_ids, int n_call_ids,
                            const int32_t *frontier_ids, int n_frontier,
                            const char *visible_text,
                            const char *tools_json) {
    return put_internal(s, api, call_ids, n_call_ids, frontier_ids, n_frontier,
                        visible_text, tools_json, s && s->persist_dir);
}

const ember_continuation_entry *
ember_continuation_get(ember_continuation_store *s, ember_api_kind api,
                       const char *const *call_ids, int n_call_ids) {
    if (!s || n_call_ids <= 0) return NULL;
    for (int i = 0; i < s->n; ++i) {
        if (s->entries[i].api == api &&
            same_set(&s->entries[i], call_ids, n_call_ids)) {
            s->entries[i].stamp = ++s->clock;
            return &s->entries[i];
        }
    }
    return NULL;
}

static bool load_file(ember_continuation_store *s, const char *path) {
    int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return false;
    struct stat st;
    uint8_t hdr[ECT_HEADER_SIZE];
    bool ok = fstat(fd, &st) == 0 && S_ISREG(st.st_mode) &&
              read_all(fd, hdr, sizeof(hdr));
    if (!ok || memcmp(hdr, "ECT1", 4) != 0 ||
        get_u32le(hdr + 4) != ECT_VERSION ||
        memcmp(hdr + 8, s->persist_identity, 16) != 0) {
        close(fd);
        return false;
    }
    ember_api_kind api = (ember_api_kind)get_u32le(hdr + 24);
    uint32_t nc = get_u32le(hdr + 28);
    uint32_t nt = get_u32le(hdr + 32);
    uint64_t nv = get_u64le(hdr + 40);
    uint64_t nj = get_u64le(hdr + 48);
    uint64_t expected = get_u64le(hdr + 56);
    uint64_t id_prefix_bytes = (uint64_t)nc * 4u;
    bool file_bound_overflow =
        (uint64_t)s->max_bytes > UINT64_MAX - ECT_HEADER_SIZE - id_prefix_bytes;
    uint64_t max_file = file_bound_overflow ? 0 :
        ECT_HEADER_SIZE + (uint64_t)s->max_bytes + id_prefix_bytes;
    if ((api < EMBER_API_CHAT || api > EMBER_API_COMPLETIONS) ||
        nc == 0 || nc > INT_MAX ||
        nc > s->max_bytes / sizeof(char *) || nt == 0 || nt > INT_MAX ||
        nv > s->max_bytes || nj > s->max_bytes ||
        (uint64_t)nt * 4u > s->max_bytes ||
        file_bound_overflow || st.st_size < 0 ||
        (uint64_t)st.st_size > max_file) {
        close(fd);
        return false;
    }

    char **ids = (char **)calloc(nc, sizeof(char *));
    int32_t *tokens = (int32_t *)malloc((size_t)nt * sizeof(int32_t));
    char *visible = (char *)malloc((size_t)nv + 1);
    char *tools = (char *)malloc((size_t)nj + 1);
    if (!ids || !tokens || !visible || !tools) ok = false;
    for (uint32_t i = 0; ok && i < nc; ++i) {
        uint8_t le[4];
        ok = read_all(fd, le, 4);
        uint32_t n = ok ? get_u32le(le) : 0;
        if (!ok || n == 0 || n > ECT_MAX_ID) {
            ok = false;
            break;
        }
        ids[i] = (char *)malloc((size_t)n + 1);
        if (!ids[i] || !read_all(fd, ids[i], n)) {
            ok = false;
            break;
        }
        ids[i][n] = '\0';
        if (!valid_id(ids[i])) ok = false;
    }
    for (uint32_t i = 0; ok && i < nt; ++i) {
        uint8_t le[4];
        if (!(ok = read_all(fd, le, 4))) break;
        tokens[i] = (int32_t)get_u32le(le);
    }
    if (ok) ok = read_all(fd, visible, (size_t)nv);
    if (ok) ok = read_all(fd, tools, (size_t)nj);
    if (ok) {
        visible[nv] = '\0';
        tools[nj] = '\0';
        off_t pos = lseek(fd, 0, SEEK_CUR);
        ok = pos == st.st_size &&
             payload_hash(api, (const char *const *)ids, (int)nc,
                          tokens, (int)nt, visible, tools) == expected;
    }
    close(fd);
    if (ok)
        ok = put_internal(s, api, (const char *const *)ids, (int)nc,
                          tokens, (int)nt, visible[0] ? visible : NULL,
                          tools[0] ? tools : NULL, false);
    for (uint32_t i = 0; i < nc; ++i) free(ids ? ids[i] : NULL);
    free(ids);
    free(tokens);
    free(visible);
    free(tools);
    return ok;
}

int ember_continuation_enable_persistence(ember_continuation_store *s,
                                          const char *root_dir,
                                          const uint8_t identity[16]) {
    if (!s || !root_dir || !identity) return -1;
    char hex[33];
    static const char hd[] = "0123456789abcdef";
    for (int i = 0; i < 16; ++i) {
        hex[i * 2] = hd[identity[i] >> 4];
        hex[i * 2 + 1] = hd[identity[i] & 15];
    }
    hex[32] = '\0';
    size_t n = strlen(root_dir) + strlen(hex) + 16;
    char *dir = (char *)malloc(n);
    if (!dir) return -1;
    snprintf(dir, n, "%s/continuations-%s", root_dir, hex);
    if (mkdir(dir, 0700) != 0 && errno != EEXIST) {
        free(dir);
        return -1;
    }
    struct stat st;
    if (lstat(dir, &st) != 0 || !S_ISDIR(st.st_mode) || S_ISLNK(st.st_mode)) {
        free(dir);
        return -1;
    }
    if (st.st_uid != geteuid() || chmod(dir, 0700) != 0) {
        free(dir);
        return -1;
    }
    DIR *d = opendir(dir);
    if (!d) {
        free(dir);
        return -1;
    }
    // Do not replace a working persistence namespace until the new directory
    // has actually been opened. An EMFILE/permission failure must leave the
    // store's prior configuration intact.
    free(s->persist_dir);
    s->persist_dir = dir;
    memcpy(s->persist_identity, identity, 16);
    int loaded = 0;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        size_t ln = strlen(de->d_name), sn = strlen(ECT_SUFFIX);
        if (ln <= sn || strcmp(de->d_name + ln - sn, ECT_SUFFIX) != 0)
            continue;
        char *path = child_path(dir, de->d_name);
        if (!path) continue;
        if (load_file(s, path)) loaded++;
        else (void)unlink(path);
        free(path);
    }
    closedir(d);
    return loaded;
}
