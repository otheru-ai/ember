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
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

size_t ember_http_parse(char *raw, size_t len, ember_http_request *req) {
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
    while (p < hdr_end && req->n_headers < EMBER_HTTP_MAX_HEADERS) {
        char *eol = memmem(p, (size_t)(hdr_end - p), "\r\n", 2);
        if (!eol) eol = hdr_end;
        char *colon = memchr(p, ':', (size_t)(eol - p));
        if (colon) {
            *colon = '\0';
            char *val = colon + 1;
            while (val < eol && (*val == ' ' || *val == '\t')) val++;
            *eol = '\0';
            req->headers[req->n_headers].name = p;
            req->headers[req->n_headers].value = val;
            req->n_headers++;
        }
        p = eol + 2;
    }

    size_t body_off = (size_t)(hdr_end + 4 - raw);
    req->body = raw + body_off;
    const char *cl = ember_http_header(req, "Content-Length");
    size_t declared = cl ? (size_t)strtoul(cl, NULL, 10) : 0;
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
    while (off < len) {
        ssize_t n = send(fd, p + off, len - off, MSG_NOSIGNAL);
        if (n <= 0) return -1;
        off += (size_t)n;
    }
    return 0;
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
    size_t body_off = 0, need = 0;
    ember_http_request req;
    int have_headers = 0;

    for (;;) {
        if (len + 1 >= cap) { cap *= 2; buf = (char *)realloc(buf, cap); }
        ssize_t n = recv(fd, buf + len, cap - len - 1, 0);
        if (n <= 0) goto done;
        len += (size_t)n;
        buf[len] = '\0';
        if (!have_headers && memmem(buf, len, "\r\n\r\n", 4)) {
            // parse once we have full headers; may still need body bytes
            char *tmp = (char *)malloc(len + 1);
            memcpy(tmp, buf, len + 1);
            body_off = ember_http_parse(tmp, len, &req);
            free(tmp);
            if (body_off == 0) goto done;
            const char *cl_hdr = NULL;
            // recompute Content-Length against the live buffer
            {
                char *saved = (char *)malloc(len + 1);
                memcpy(saved, buf, len + 1);
                ember_http_request r2;
                ember_http_parse(saved, len, &r2);
                cl_hdr = ember_http_header(&r2, "Content-Length");
                unsigned long clv = cl_hdr ? strtoul(cl_hdr, NULL, 10) : 0;
                free(saved);
                if (clv > (64UL << 20)) goto done;  // reject oversize body (ds4 64MB cap)
                need = body_off + clv;
            }
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
    atomic_fetch_sub(&g_active_conns, 1);  // #3: release the connection slot
    return NULL;
}

int ember_http_serve(int port, ember_http_handler handler, void *ud) {
    int lfd = socket(AF_INET, SOCK_STREAM, 0);
    if (lfd < 0) return 1;
    int yes = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons((uint16_t)port);
    if (bind(lfd, (struct sockaddr *)&addr, sizeof(addr)) != 0) { close(lfd); return 2; }
    if (listen(lfd, 64) != 0) { close(lfd); return 3; }
    fprintf(stderr, "[ember] listening on http://127.0.0.1:%d\n", port);

    for (;;) {
        int cfd = accept(lfd, NULL, NULL);
        if (cfd < 0) continue;
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
            atomic_fetch_sub(&g_active_conns, 1);
            send_503(cfd);
            close(cfd);
            continue;
        }
        conn_ctx *ctx = (conn_ctx *)malloc(sizeof(conn_ctx));
        ctx->fd = cfd;
        ctx->handler = handler;
        ctx->ud = ud;
        pthread_t t;
        if (pthread_create(&t, NULL, conn_thread, ctx) != 0) {
            atomic_fetch_sub(&g_active_conns, 1);  // #3: no thread to release it
            close(cfd);
            free(ctx);
        } else {
            pthread_detach(t);
        }
    }
    // unreachable
}
