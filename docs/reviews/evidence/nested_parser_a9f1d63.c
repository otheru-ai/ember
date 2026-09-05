// Independent offline reproducer. Link against unmodified a9f1d63 ember_core.
#include "model/tool_parser.h"
#include "server/sse.h"
#include "common/json_util.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { ember_buf name, args; int calls; } capture;
static void start(void *ud, int index, const char *id, const char *name, ember_buf *out) {
    capture *c = ud;
    (void)index; (void)id; (void)out;
    ++c->calls; ember_buf_puts(&c->name, name);
}
static void args(void *ud, int index, const char *text, size_t len, ember_buf *out) {
    capture *c = ud;
    (void)index; (void)out;
    ember_buf_append(&c->args, text, len);
}
int main(int argc, char **argv) {
    (void)argv;
    const bool parameter_only = argc > 1;
    const char *full =
        "<?DSML?tool_calls><?DSML?invoke name=\"write_file\">"
        "<?DSML?parameter name=\"content\" string=\"true\">BEFORE "
        "<?DSML?tool_calls><?DSML?invoke name=\"terminal\">"
        "<?DSML?parameter name=\"command\" string=\"true\">echo nested"
        "</?DSML?parameter></?DSML?invoke></?DSML?tool_calls> AFTER"
        "</?DSML?parameter></?DSML?invoke></?DSML?tool_calls>";
    if (parameter_only) full =
        "<?DSML?tool_calls><?DSML?invoke name=\"write_file\">"
        "<?DSML?parameter name=\"content\" string=\"true\">BEFORE "
        "<?DSML?parameter name=\"inner\" string=\"true\">nested"
        "</?DSML?parameter> AFTER</?DSML?parameter>"
        "</?DSML?invoke></?DSML?tool_calls>";
    ember_tool_calls parsed = {0}; ember_tool_parse_report report = {0};
    const int n = ember_parse_dsml_tool_calls_ex(full, &parsed, &report);
    const char *end = ember_find_tool_end(full);
    if (!end) return 3;
    const size_t stop = (size_t)(end - full), total = strlen(full);
    char *prefix = malloc(stop + 1);
    if (!prefix) return 3;
    memcpy(prefix, full, stop); prefix[stop] = 0;
    ember_tool_calls truncated = {0}; ember_tool_parse_report tr = {0};
    const int prefix_n = ember_parse_dsml_tool_calls_ex(prefix, &truncated, &tr);
    capture got = {0}; ember_sse_stream st; ember_buf wire = {0};
    ember_sse_init(&st, "review", "synthetic", 0, true, false, false);
    const ember_sse_sink sink = {.tool_start = start, .tool_args_delta = args};
    ember_sse_set_sink(&st, &sink, &got);
    if (n > 0 && !report.malformed && !report.contaminated && !report.mixed_syntax) {
        ember_sse_update(&st, full, total, true, &wire);
        (void)ember_sse_emit_tools(&st, full, total, &wire);
    }
    ember_buf result = {0};
    ember_buf_printf(&result, "{\"full_parse_calls\":%d,\"complete\":%s,\"first_end\":%zu,\"full_length\":%zu,\"prefix_parse_calls\":%d,\"prefix_complete\":%s,\"prefix_malformed\":%s,\"sse_calls\":%d,\"validated_arguments\":", n, report.complete?"true":"false", stop, total, prefix_n, tr.complete?"true":"false", tr.malformed?"true":"false", got.calls);
    ember_json_escape(&result, n ? parsed.calls[0].arguments : "");
    ember_buf_puts(&result, ",\"emitted_arguments\":");
    ember_json_escape(&result, got.args.ptr ? got.args.ptr : "");
    const bool unequal = n && got.args.ptr && strcmp(parsed.calls[0].arguments, got.args.ptr) != 0;
    ember_buf_printf(&result, ",\"emission_differs_from_validated\":%s}", unequal?"true":"false");
    puts(result.ptr);
    const char *flag = getenv("EMBER_DSML_NESTED_VALUES");
    const bool enabled = flag && flag[0] == '1';
    const bool reproduced = enabled ? (n == 1 && (parameter_only ? (stop == total && prefix_n == 1) : (stop < total && prefix_n == 0)) && unequal) : (n == 0 && got.calls == 0);
    free(prefix); ember_tool_calls_free(&parsed); ember_tool_calls_free(&truncated);
    ember_sse_free(&st); ember_buf_free(&wire); ember_buf_free(&got.name); ember_buf_free(&got.args); ember_buf_free(&result);
    return reproduced ? 0 : 1;
}
