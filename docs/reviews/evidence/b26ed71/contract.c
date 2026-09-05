// Independent integration contract: intended values -> parser -> SSE, all families.
// Compile against the revision under review, never execute tool names/arguments.
#include "model/tool_parser.h"
#include "server/sse.h"
#include "common/json_util.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int checks, failures;
static void check(bool ok, const char *label, int family, int kind) {
    ++checks;
    if (!ok) { ++failures; fprintf(stderr, "FAIL family=%d kind=%d: %s\n", family, kind, label); }
}
typedef struct { int starts; ember_buf names[2], args[2]; } captured;
static void capture_start(void *ud, int index, const char *id, const char *name, ember_buf *out) {
    captured *c = ud; (void)id; (void)out;
    ++c->starts;
    if (index >= 0 && index < 2) ember_buf_puts(&c->names[index], name);
}
static void capture_arg(void *ud, int index, const char *text, size_t len, ember_buf *out) {
    captured *c = ud; (void)out;
    if (index >= 0 && index < 2) ember_buf_append(&c->args[index], text, len);
}
static void parameter(ember_buf *b, const ember_dsml_syntax *sx, const char *name, const char *value) {
    ember_buf_printf(b, "%s name=\"%s\" string=\"true\">%s%s", sx->param_open, name, value, sx->param_close);
}
static void invoke(ember_buf *b, const ember_dsml_syntax *sx, const char *value) {
    ember_buf_printf(b, "%s name=\"write_file\">", sx->invoke_open);
    parameter(b, sx, "content", value);
    parameter(b, sx, "tail", "tail sentinel");
    ember_buf_puts(b, sx->invoke_close);
}
static void stream_case(const char *full, const char *expected, bool valid, int family, int kind) {
    const size_t len = strlen(full);
    ember_tool_calls tc = {0}; ember_tool_parse_report r = {0};
    const int n = ember_parse_dsml_tool_calls_ex(full, &tc, &r);
    check(valid ? (n == 1 && r.complete && !r.malformed && !r.contaminated && !r.mixed_syntax && !r.repaired) : n == 0,
          "parser acceptance", family, kind);
    if (valid && n == 1) check(strcmp(tc.calls[0].arguments, expected) == 0,
                              "parser preserves entire intended value and following parameter", family, kind);
    if (valid) {
        bool premature = false;
        ember_buf prefix = {0};
        for (size_t i = 0; i < len; ++i) {
            ember_buf_putc(&prefix, full[i]);
            const char *end = ember_find_tool_end(prefix.ptr);
            if (i + 1 < len && end) premature = true;
        }
        check(!premature, "no generation stop at an inner closer for any byte prefix", family, kind);
        const char *end = ember_find_tool_end(full);
        check(end == full + len, "outer wrapper defines complete tool/replay boundary", family, kind);
        ember_buf_free(&prefix);
    }
    // Reconstruct every possible fixed-size chunking. Nothing is emitted until
    // the caller commits the validated call, then bytes must equal the oracle.
    for (size_t chunk = 1; chunk <= len; ++chunk) {
        ember_sse_stream st; ember_buf acc = {0}, wire = {0}; captured got = {0};
        ember_sse_init(&st, "review", "synthetic", 0, true, false, false);
        const ember_sse_sink sink = {.tool_start=capture_start, .tool_args_delta=capture_arg};
        ember_sse_set_sink(&st, &sink, &got);
        for (size_t off = 0; off < len;) {
            const size_t take = len-off < chunk ? len-off : chunk;
            ember_buf_append(&acc, full+off, take); off += take;
            ember_sse_update(&st, acc.ptr, acc.len, false, &wire);
        }
        check(got.starts == 0, "unvalidated tool remains buffered", family, kind);
        if (valid && n == 1) {
            ember_sse_update(&st, acc.ptr, acc.len, true, &wire);
            (void)ember_sse_emit_tools(&st, acc.ptr, acc.len, &wire);
            check(got.starts == 1 && got.names[0].ptr && strcmp(got.names[0].ptr, "write_file") == 0,
                  "one outer call emitted", family, kind);
            check(got.args[0].ptr && strcmp(got.args[0].ptr, expected) == 0,
                  "SSE exactly preserves intended and validated arguments", family, kind);
        } else {
            (void)ember_sse_discard_tool_block(&st, acc.len);
            check(got.starts == 0, "rejected nested call is not emitted", family, kind);
        }
        ember_sse_free(&st); ember_buf_free(&acc); ember_buf_free(&wire);
        for (int i=0;i<2;++i) { ember_buf_free(&got.names[i]); ember_buf_free(&got.args[i]); }
    }
    ember_tool_calls_free(&tc);
}
int main(void) {
    const char *flag = getenv("EMBER_DSML_NESTED_VALUES");
    const bool enabled = flag && flag[0] == '1';
    int nf = 0; const ember_dsml_syntax *families = ember_dsml_syntaxes(&nf);
    for (int family=0; family<nf; ++family) {
        const ember_dsml_syntax *sx = &families[family];
        for (int kind=0; kind<5; ++kind) {
            ember_buf value={0}, full={0}, expected={0}, inner={0};
            ember_buf_puts(&value, "BEFORE \xce\xa9 ");
            const ember_dsml_syntax *nested = kind==4 ? &families[(family+1)%nf] : sx;
            if (kind==1) parameter(&value, nested, "inner", "nested parameter");
            if (kind>=2) {
                invoke(&inner, nested, "nested call");
                if (kind>=3) ember_buf_puts(&value, nested->calls_open);
                ember_buf_puts(&value, inner.ptr);
                if (kind>=3) ember_buf_puts(&value, nested->calls_close);
            }
            ember_buf_puts(&value, " AFTER");
            ember_buf_puts(&full, sx->calls_open); invoke(&full, sx, value.ptr); ember_buf_puts(&full, sx->calls_close);
            ember_buf_puts(&expected, "{\"content\":"); ember_json_escape(&expected, value.ptr);
            ember_buf_puts(&expected, ",\"tail\":\"tail sentinel\"}");
            stream_case(full.ptr, expected.ptr, enabled || kind==0, family, kind);
            ember_buf_free(&value); ember_buf_free(&full); ember_buf_free(&expected); ember_buf_free(&inner);
        }
    }
    printf("{\"enabled\":%s,\"families\":%d,\"checks\":%d,\"failures\":%d}\n", enabled?"true":"false", nf, checks, failures);
    return failures ? 1 : 0;
}
