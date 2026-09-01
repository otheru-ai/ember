// Minimal HTTP/1.1 request parsing + a threaded listening server. Enough for a
// local inference endpoint: GET/POST, header lookup, Content-Length body.
#ifndef EMBER_HTTP_H
#define EMBER_HTTP_H

#include <stddef.h>

#define EMBER_HTTP_MAX_HEADERS 48

typedef struct {
    char method[8];
    char path[1024];
    struct { char *name; char *value; } headers[EMBER_HTTP_MAX_HEADERS];
    int  n_headers;
    const char *body;   // points into the caller's raw buffer
    size_t body_len;
} ember_http_request;

// Parse the request line + headers from `raw` (len bytes). Returns the offset
// where the body begins (== header end), or 0 on parse failure. Sets req->body
// to raw+offset and body_len from Content-Length (clamped to available bytes).
// `raw` must outlive `req` (headers/body point into it; the request line and
// path are copied into fixed buffers).
size_t ember_http_parse(char *raw, size_t len, ember_http_request *req);

// Case-insensitive header lookup; NULL if absent.
const char *ember_http_header(const ember_http_request *req, const char *name);

// Handler invoked per request; writes the full response to client_fd itself
// (so it can stream). `ud` is the user pointer passed to ember_http_serve.
typedef void (*ember_http_handler)(const ember_http_request *req, int client_fd,
                                   void *ud);

// Listen on 127.0.0.1:port and dispatch each connection to `handler` on its own
// thread. Blocks. Returns non-zero on listen/bind failure.
int ember_http_serve(int port, ember_http_handler handler, void *ud);

// Blocking write-all helper for handlers.
int ember_send_all(int fd, const void *data, size_t len);

#endif  // EMBER_HTTP_H
