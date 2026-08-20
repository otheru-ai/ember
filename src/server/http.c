// memmem() is a GNU extension; without _GNU_SOURCE it is implicitly declared as
// int-returning, which truncates its pointer result to 32 bits on x86-64 and
// corrupts every parse (segfault). Must precede all includes. Guarded: the
// container build already passes -D_GNU_SOURCE, the CMake stub build does not.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "http.h"

#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

size_t ember_http_parse(char *raw, size_t len, ember_http_request *req) {
    if (!raw || !req) return 0;
    memset(req, 0, sizeof(*req));
    // request line: METHOD SP PATH SP HTTP/x
    char *line_end = memmem(raw, len, "\r\n", 2);
    if (!line_end) return 0;
    *line_end = '\0';
    if (sscanf(raw, "%7s %1023s", req->method, req->path) != 2) return 0;
    // Strip the query string so exact-match routing works (ds4 read_http_request).
    char *q = strchr(req->path, '?');
    if (q) *q = '\0';

    char *p = line_end + 2;
    char *hdr_end = memmem(p, len - (size_t)(p - raw), "\r\n\r\n", 4);
    if (!hdr_end) return 0;

    // headers
    while (p < hdr_end) {
        // Never silently drop headers: a proxy or caller may make a security
        // decision from a field that Ember would otherwise ignore after #48.
        if (req->n_headers >= EMBER_HTTP_MAX_HEADERS) return 0;
        char *eol = memmem(p, (size_t)(hdr_end - p), "\r\n", 2);
        if (!eol) eol = hdr_end;
        char *colon = memchr(p, ':', (size_t)(eol - p));
        if (!colon || colon == p) return 0;
        for (char *name = p; name < colon; ++name)
            if (!isalnum((unsigned char)*name) && *name != '!' &&
                *name != '#' && *name != '$' && *name != '%' &&
                *name != '&' && *name != '\'' && *name != '*' &&
                *name != '+' && *name != '-' && *name != '.' &&
                *name != '^' && *name != '_' && *name != '`' &&
                *name != '|' && *name != '~') return 0;
        *colon = '\0';
        char *val = colon + 1;
        while (val < eol && (*val == ' ' || *val == '\t')) val++;
        *eol = '\0';
        req->headers[req->n_headers].name = p;
        req->headers[req->n_headers].value = val;
        req->n_headers++;
        p = eol + 2;
    }

    size_t body_off = (size_t)(hdr_end + 4 - raw);
    req->body = raw + body_off;
    size_t declared = 0;
    bool saw_cl = false;
    for (int i = 0; i < req->n_headers; ++i) {
        // Chunked request bodies are deliberately unsupported. Reject every
        // Transfer-Encoding spelling rather than create a proxy/backend
        // framing differential.
        if (strcasecmp(req->headers[i].name, "Transfer-Encoding") == 0)
            return 0;
        if (strcasecmp(req->headers[i].name, "Content-Length") != 0) continue;
        // Reject duplicate framing fields rather than leave an HTTP
        // request-smuggling ambiguity for an upstream proxy to interpret.
        if (saw_cl) return 0;
        saw_cl = true;
        const unsigned char *v =
            (const unsigned char *)req->headers[i].value;
        if (!*v) return 0;
        size_t parsed = 0;
        while (*v) {
            if (*v < '0' || *v > '9') return 0;
            unsigned digit = (unsigned)(*v - '0');
            if (parsed > (SIZE_MAX - digit) / 10) return 0;
            parsed = parsed * 10 + digit;
            ++v;
        }
        declared = parsed;
    }
    req->content_length = declared;
    size_t avail = len - body_off;
    req->body_len = declared < avail ? declared : avail;
    return body_off;
}

const char *ember_http_header(const ember_http_request *req, const char *name) {
    for (int i = 0; i < req->n_headers; i++) {
        if (strcasecmp(req->headers[i].name, name) == 0)
            return req->headers[i].value;
    }
    return NULL;
}

int ember_send_all(int fd, const void *data, size_t len) {
    const char *p = (const char *)data;
    size_t off = 0;
    int original_flags = fcntl(fd, F_GETFL, 0);
    if (original_flags < 0) return -1;
    bool restore_flags = !(original_flags & O_NONBLOCK);
    if (restore_flags && fcntl(fd, F_SETFL, original_flags | O_NONBLOCK) != 0)
        return -1;
    struct timespec start;
    if (clock_gettime(CLOCK_MONOTONIC, &start) != 0) {
        if (restore_flags) (void)fcntl(fd, F_SETFL, original_flags);
        return -1;
    }
    const int64_t deadline_ms =
        (int64_t)start.tv_sec * 1000 + start.tv_nsec / 1000000 + 20000;
    while (off < len) {
        ssize_t n = send(fd, p + off, len - off, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR) continue;
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            struct timespec now;
            if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) break;
            int64_t now_ms =
                (int64_t)now.tv_sec * 1000 + now.tv_nsec / 1000000;
            int64_t left = deadline_ms - now_ms;
            if (left <= 0) { errno = ETIMEDOUT; break; }
            struct pollfd wait = {.fd = fd, .events = POLLOUT};
            int rc = poll(&wait, 1, left > INT32_MAX ? INT32_MAX : (int)left);
            if (rc < 0 && errno == EINTR) continue;
            if (rc <= 0 || (wait.revents & (POLLERR | POLLHUP | POLLNVAL))) {
                if (rc == 0) errno = ETIMEDOUT;
                break;
            }
            continue;
        }
        if (n <= 0) break;
        off += (size_t)n;
    }
    int send_errno = off == len ? 0 : errno;
    if (restore_flags && fcntl(fd, F_SETFL, original_flags) != 0) return -1;
    if (off == len) return 0;
    errno = send_errno ? send_errno : EIO;
    return -1;
}

// True once the peer has closed its end, detectable WITHOUT writing anything.
//
// A failed send() is not a sufficient liveness signal for streaming generation:
// prefill can run for minutes with no token to emit, and decode can spend long
// stretches producing bytes the encoder withholds, so there may simply be no
// write to fail. A client that gives up in that window went unnoticed and its
// generation ran to completion holding the slot. POLLRDHUP reports the FIN
// directly. It is deliberately not MSG_PEEK: on a keep-alive connection a
// pipelined next request also makes the socket readable, and that must not be
// mistaken for a hangup.
bool ember_client_gone(int fd) {
    if (fd < 0) return true;
    struct pollfd p = { .fd = fd, .events = POLLRDHUP, .revents = 0 };
    const int rc = poll(&p, 1, 0);  // 0 timeout: never stalls the decode loop
    if (rc <= 0) return false;      // no event, or EINTR — assume still connected
    return (p.revents & (POLLRDHUP | POLLHUP | POLLERR | POLLNVAL)) != 0;
}

typedef struct {
    int                 fd;
    ember_http_handler  handler;
    void               *ud;
} conn_ctx;

// #3: cap concurrent connection threads. Every connection spawns a detached
// thread that can allocate up to a 64MB request buffer; without a ceiling a
// flood of slow/abundant clients exhausts threads and memory. Over the cap we
// send a minimal 503 and close instead of accepting the work.
#define EMBER_MAX_CONNS 64
static atomic_int g_active_conns;
static volatile sig_atomic_t g_stop_requested;
static volatile sig_atomic_t g_listen_fd = -1;
static pthread_mutex_t g_conns_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_conns_cv = PTHREAD_COND_INITIALIZER;

void ember_http_request_stop(void) {
    g_stop_requested = 1;
    if (g_listen_fd >= 0) {
        int fd = (int)g_listen_fd;
        g_listen_fd = -1;
        close(fd);
    }
}

static void connection_release(void) {
    pthread_mutex_lock(&g_conns_mu);
    int before = atomic_fetch_sub(&g_active_conns, 1);
    if (before == 1) pthread_cond_broadcast(&g_conns_cv);
    pthread_mutex_unlock(&g_conns_mu);
}

// #3: minimal overload response for a rejected connection (Content-Length is
// computed so no manual byte-count can drift).
static void send_503(int fd) {
    static const char body[] = "{\"error\":{\"message\":\"server busy, retry later\"}}";
    char resp[192];
    int n = snprintf(resp, sizeof(resp),
        "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\n"
        "Content-Length: %zu\r\nConnection: close\r\n\r\n%s",
        sizeof(body) - 1, body);
    if (n > 0) ember_send_all(fd, resp, (size_t)n);
}

// Read until we have the full request (headers + Content-Length body).
static void *conn_thread(void *arg) {
    conn_ctx *ctx = (conn_ctx *)arg;
    int fd = ctx->fd;
    int flag = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    size_t cap = 8192, len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) goto done;
    size_t body_off = 0, need = 0;
    ember_http_request req;
    int have_headers = 0;

    for (;;) {
        if (len + 1 >= cap) {
            size_t next_cap = cap * 2;
            char *grown = (char *)realloc(buf, next_cap);
            if (!grown) goto done;
            buf = grown;
            cap = next_cap;
        }
        ssize_t n = recv(fd, buf + len, cap - len - 1, 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) goto done;
        len += (size_t)n;
        buf[len] = '\0';
        if (!have_headers && memmem(buf, len, "\r\n\r\n", 4)) {
            // parse once we have full headers; may still need body bytes
            char *tmp = (char *)malloc(len + 1);
            if (!tmp) goto done;
            memcpy(tmp, buf, len + 1);
            body_off = ember_http_parse(tmp, len, &req);
            free(tmp);
            if (body_off == 0) goto done;
            if (req.content_length > (64UL << 20))
                goto done;  // reject oversize body (ds4 64MB cap)
            if (req.content_length > SIZE_MAX - body_off) goto done;
            need = body_off + req.content_length;
            have_headers = 1;
        }
        if (have_headers && len >= need) break;
        if (!have_headers && len > (1u << 20)) goto done;  // header flood guard
    }

    // final parse against the complete buffer (in place)
    body_off = ember_http_parse(buf, len, &req);
    if (body_off) ctx->handler(&req, fd, ctx->ud);

done:
    free(buf);
    close(fd);
    free(ctx);
    connection_release();
    return NULL;
}

bool ember_http_host_valid(const char *host) {
    struct in_addr addr;
    return host && host[0] && inet_pton(AF_INET, host, &addr) == 1;
}

int ember_http_serve(const char *host, int port,
                     ember_http_handler handler, void *ud) {
    if (g_stop_requested) return 0;
    struct sockaddr_in addr = {0};
    if (!ember_http_host_valid(host)) return 1;
    addr.sin_family = AF_INET;
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) return 1;
    addr.sin_port = htons((uint16_t)port);

    int lfd = socket(AF_INET, SOCK_STREAM, 0);
    if (lfd < 0) return 2;
    int yes = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    if (bind(lfd, (struct sockaddr *)&addr, sizeof(addr)) != 0) { close(lfd); return 3; }
    if (listen(lfd, 64) != 0) { close(lfd); return 4; }
    g_listen_fd = lfd;
    fprintf(stderr, "[ember] listening on http://%s:%d\n", host, port);

    while (!g_stop_requested) {
        int cfd = accept(lfd, NULL, NULL);
        if (cfd < 0) {
            if (g_stop_requested) break;
            continue;
        }
        if (g_stop_requested) {
            close(cfd);
            break;
        }
        // Bound both directions so a stalled/slowloris client can't wedge the
        // thread or (via a full write window inside on_token) hold gen_lock and
        // freeze the single-slot server. ds4 uses non-blocking + poll deadlines;
        // SO_*TIMEO achieves the same "never block forever" guarantee simply.
        struct timeval rcv = { .tv_sec = 30, .tv_usec = 0 };
        struct timeval snd = { .tv_sec = 20, .tv_usec = 0 };
        setsockopt(cfd, SOL_SOCKET, SO_RCVTIMEO, &rcv, sizeof(rcv));
        setsockopt(cfd, SOL_SOCKET, SO_SNDTIMEO, &snd, sizeof(snd));
        // #3: refuse work past the concurrency cap. fetch_add returns the prior
        // count; if we were already at the ceiling, undo and shed the client.
        if (atomic_fetch_add(&g_active_conns, 1) >= EMBER_MAX_CONNS) {
            connection_release();
            send_503(cfd);
            close(cfd);
            continue;
        }
        conn_ctx *ctx = (conn_ctx *)malloc(sizeof(conn_ctx));
        if (!ctx) {
            connection_release();
            close(cfd);
            continue;
        }
        ctx->fd = cfd;
        ctx->handler = handler;
        ctx->ud = ud;
        pthread_t t;
        if (pthread_create(&t, NULL, conn_thread, ctx) != 0) {
            connection_release();
            close(cfd);
            free(ctx);
        } else {
            pthread_detach(t);
        }
    }
    if (g_listen_fd >= 0) {
        g_listen_fd = -1;
        close(lfd);
    }

    // No new connection can increment the count after the listener is closed.
    // Drain accepted clients before returning so the handler's `ud` and the
    // generation worker remain alive until every request has finished.
    pthread_mutex_lock(&g_conns_mu);
    while (atomic_load(&g_active_conns) > 0)
        pthread_cond_wait(&g_conns_cv, &g_conns_mu);
    pthread_mutex_unlock(&g_conns_mu);
    return 0;
}
