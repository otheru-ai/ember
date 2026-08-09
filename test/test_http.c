#include <errno.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include "../src/server/http.h"
static int g_pass=0,g_fail=0;
#define CHECK(c,m) do{ if(c)g_pass++; else{g_fail++;printf("  FAIL: %s\n",m);} }while(0)

// ── live accept/read loop ───────────────────────────────────────────────
//
// Everything above exercises ember_http_parse in isolation. The connection
// loop in conn_thread() — buffer growth, the oversize-body cap and the header
// flood guard — had no coverage at all, and those three are the request-side
// resource limits: without them a single client can drive the server to
// allocate without bound. They are only reachable through a real socket.
//
// ember_http_request_stop() latches a process-global, so a second
// ember_http_serve() would return immediately. Every server-backed assertion
// therefore shares one serve lifetime and the stop happens once, at the end.

typedef struct {
    pthread_mutex_t mu;
    int             requests;      // handler invocations
    size_t          last_body_len;
    int             rc;            // ember_http_serve return code
    int             port;
} server_state;

static void record_handler(const ember_http_request *req, int fd, void *ud) {
    server_state *st = (server_state *)ud;
    pthread_mutex_lock(&st->mu);
    st->requests++;
    st->last_body_len = req->body_len;
    pthread_mutex_unlock(&st->mu);
    static const char ok[] =
        "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok";
    ember_send_all(fd, ok, sizeof(ok) - 1);
}

static void *serve_thread(void *ud) {
    server_state *st = (server_state *)ud;
    st->rc = ember_http_serve(st->port, record_handler, st);
    return NULL;
}

// Bind port 0, read back what the kernel chose, release it. SO_REUSEADDR on the
// real listener makes the reuse safe.
static int pick_free_port(void) {
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) return -1;
    struct sockaddr_in a = {0};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = 0;
    int port = -1;
    if (bind(s, (struct sockaddr *)&a, sizeof(a)) == 0) {
        socklen_t n = sizeof(a);
        if (getsockname(s, (struct sockaddr *)&a, &n) == 0)
            port = ntohs(a.sin_port);
    }
    close(s);
    return port;
}

static int connect_retry(int port) {
    for (int attempt = 0; attempt < 200; ++attempt) {
        int c = socket(AF_INET, SOCK_STREAM, 0);
        if (c < 0) return -1;
        struct sockaddr_in a = {0};
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        a.sin_port = htons((uint16_t)port);
        if (connect(c, (struct sockaddr *)&a, sizeof(a)) == 0) return c;
        close(c);
        usleep(5000);
    }
    return -1;
}

// Read to EOF. Returns bytes read; caller frees *out when non-NULL.
static size_t read_all(int fd, char **out) {
    size_t cap = 4096, len = 0;
    char *buf = malloc(cap);
    if (!buf) { if (out) *out = NULL; return 0; }
    for (;;) {
        if (len + 1 >= cap) {
            char *g = realloc(buf, cap * 2);
            if (!g) break;
            buf = g; cap *= 2;
        }
        ssize_t n = read(fd, buf + len, cap - len - 1);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        len += (size_t)n;
    }
    buf[len] = '\0';
    if (out) *out = buf; else free(buf);
    return len;
}

// pick_free_port() releases the port before ember_http_serve() rebinds it, so
// another process can win the race and serve() returns a bind error. Retry on a
// fresh port instead of failing: a flaky test is worse than no test.
static bool start_server(server_state *st, pthread_t *server) {
    for (int attempt = 0; attempt < 5; ++attempt) {
        st->port = pick_free_port();
        if (st->port <= 0) continue;
        st->rc = -1;
        if (pthread_create(server, NULL, serve_thread, st) != 0) return false;
        // A probe connection proves the listener is up. Closing it immediately
        // makes recv() return 0, so conn_thread exits without the handler.
        int probe = connect_retry(st->port);
        if (probe >= 0) {
            close(probe);
            return true;
        }
        pthread_join(*server, NULL);   // bind failed; serve() already returned
    }
    return false;
}

static void test_server_loop(void) {
    // A peer we intentionally overrun will close on us; SIGPIPE would kill the
    // test process rather than surface as a write error.
    signal(SIGPIPE, SIG_IGN);

    server_state st = {.mu = PTHREAD_MUTEX_INITIALIZER};
    pthread_t server;
    CHECK(start_server(&st, &server), "test server listening");
    if (st.port <= 0 || st.rc == 0) return;

    // 1. A body larger than the 8 KiB initial read buffer must drive the
    //    realloc growth loop and still arrive whole.
    const size_t body_len = 20000;
    char *body = malloc(body_len + 1);
    CHECK(body != NULL, "allocated large body");
    if (body) {
        memset(body, 'a', body_len);
        body[body_len] = '\0';
        char head[256];
        int hn = snprintf(head, sizeof(head),
                          "POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
                          "Content-Type: application/json\r\n"
                          "Content-Length: %zu\r\n\r\n", body_len);
        int c = connect_retry(st.port);
        CHECK(c >= 0, "connected to the test server");
        if (c >= 0) {
            CHECK(ember_send_all(c, head, (size_t)hn) == 0, "sent headers");
            CHECK(ember_send_all(c, body, body_len) == 0, "sent large body");
            char *resp = NULL;
            size_t n = read_all(c, &resp);
            CHECK(n > 0 && resp && strstr(resp, "200 OK") != NULL,
                  "oversized-but-legal request was served");
            free(resp);
            close(c);
        }
        free(body);
    }
    pthread_mutex_lock(&st.mu);
    CHECK(st.requests == 1, "handler ran once");
    CHECK(st.last_body_len == body_len,
          "full body survived the buffer growth loop");
    pthread_mutex_unlock(&st.mu);

    // 2. A declared body past the 64 MiB cap must be refused before a single
    //    body byte is buffered — the connection closes with no response and the
    //    handler never runs.
    {
        int c = connect_retry(st.port);
        CHECK(c >= 0, "connected for the oversize-body case");
        if (c >= 0) {
            const char *head =
                "POST / HTTP/1.1\r\nHost: t\r\n"
                "Content-Length: 67108865\r\n\r\n";   // 64 MiB + 1
            (void)ember_send_all(c, head, strlen(head));
            char *resp = NULL;
            size_t n = read_all(c, &resp);
            CHECK(n == 0, "oversize declared body got no response");
            free(resp);
            close(c);
        }
    }

    // 3. Headers that never terminate must trip the 1 MiB flood guard rather
    //    than growing the buffer until the allocator gives up.
    {
        int c = connect_retry(st.port);
        CHECK(c >= 0, "connected for the header-flood case");
        if (c >= 0) {
            const char *start = "GET / HTTP/1.1\r\n";
            (void)ember_send_all(c, start, strlen(start));
            char line[128];
            memset(line, 'a', sizeof(line));
            memcpy(line, "X-Pad: ", 7);
            line[sizeof(line) - 2] = '\r';
            line[sizeof(line) - 1] = '\n';
            // Push past 1 MiB; the server closes mid-way, so a short write or
            // EPIPE here is the expected outcome, not a failure.
            for (int i = 0; i < 12000; ++i)
                if (ember_send_all(c, line, sizeof(line)) != 0) break;
            char *resp = NULL;
            size_t n = read_all(c, &resp);
            CHECK(n == 0, "header flood got no response");
            free(resp);
            close(c);
        }
    }

    pthread_mutex_lock(&st.mu);
    CHECK(st.requests == 1, "malformed connections never reached the handler");
    pthread_mutex_unlock(&st.mu);

    ember_http_request_stop();
    // Nudge accept() in case it is already blocked on a closed listener.
    int wake = connect_retry(st.port);
    if (wake >= 0) close(wake);
    CHECK(pthread_join(server, NULL) == 0, "server thread joined after stop");
    CHECK(st.rc == 0, "serve returned cleanly");
    pthread_mutex_destroy(&st.mu);
}

int main(void){
    printf("ember http tests\n");
    char raw[] = "POST /v1/chat/completions HTTP/1.1\r\n"
                 "Host: localhost\r\nContent-Type: application/json\r\n"
                 "Content-Length: 7\r\n\r\n{\"a\":1}";
    ember_http_request req;
    size_t off = ember_http_parse(raw, strlen(raw), &req);
    CHECK(off>0, "parse ok");
    CHECK(strcmp(req.method,"POST")==0, "method");
    CHECK(strcmp(req.path,"/v1/chat/completions")==0, "path");
    CHECK(strcmp(ember_http_header(&req,"content-type"),"application/json")==0, "hdr case-insensitive");
    CHECK(req.body_len==7 && strncmp(req.body,"{\"a\":1}",7)==0, "body + content-length");
    CHECK(req.content_length==7, "validated content length exposed");

    char bad_alpha[] =
        "POST / HTTP/1.1\r\nContent-Length: 7x\r\n\r\n1234567";
    CHECK(ember_http_parse(bad_alpha, strlen(bad_alpha), &req)==0,
          "non-decimal Content-Length rejected");
    char bad_negative[] =
        "POST / HTTP/1.1\r\nContent-Length: -1\r\n\r\n";
    CHECK(ember_http_parse(bad_negative, strlen(bad_negative), &req)==0,
          "negative Content-Length rejected");
    char duplicate[] =
        "POST / HTTP/1.1\r\nContent-Length: 1\r\nContent-Length: 1\r\n\r\nx";
    CHECK(ember_http_parse(duplicate, strlen(duplicate), &req)==0,
          "duplicate Content-Length rejected");
    char overflow[] =
        "POST / HTTP/1.1\r\nContent-Length: 999999999999999999999999\r\n\r\n";
    CHECK(ember_http_parse(overflow, strlen(overflow), &req)==0,
          "overflowing Content-Length rejected");

    // ember_client_gone: streaming generation must notice a departed client
    // without needing a write to fail, or it runs to completion into a dead
    // socket while holding the only generation slot.
    int sv[2];
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0, "socketpair");
    CHECK(!ember_client_gone(sv[0]), "live peer is not reported gone");
    CHECK(ember_send_all(sv[0], "ok", 2) == 0,
          "bounded write-all succeeds for a live peer");
    char sent[2];
    CHECK(read(sv[1], sent, sizeof(sent)) == 2 && !memcmp(sent, "ok", 2),
          "write-all delivered the complete payload");

    // A keep-alive client that pipelines its next request makes our end
    // readable. That is NOT a hangup — reporting it as one would abort a
    // perfectly healthy stream, which is why this uses POLLRDHUP over MSG_PEEK.
    CHECK(write(sv[1], "GET / HTTP/1.1\r\n", 16) == 16, "peer wrote a request");
    CHECK(!ember_client_gone(sv[0]), "pending inbound data is not a hangup");

    close(sv[1]);
    CHECK(ember_client_gone(sv[0]), "closed peer is detected");
    CHECK(ember_send_all(sv[0], "x", 1) != 0,
          "write-all reports a closed peer");
    CHECK(ember_client_gone(-1), "invalid fd counts as gone");
    close(sv[0]);

    // Runs last: it latches the process-global stop flag.
    test_server_loop();

    printf("──────────────────────────────\n  %d passed, %d failed\n",g_pass,g_fail);
    return g_fail?1:0;
}
