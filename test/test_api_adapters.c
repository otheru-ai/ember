#include "../src/server/api_adapters.h"
#include "../src/common/json.h"

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

static int pass, fail;
#define CHECK(x) do { if (x) pass++; else { fail++; fprintf(stderr, \
    "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); } } while (0)
#define PIPE "\xef\xbd\x9c"

typedef struct {
    int fd;
    char *data;
    size_t len;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    bool ready;
} reader_ctx;

static void *read_response(void *arg) {
    reader_ctx *ctx = arg;
    pthread_mutex_lock(&ctx->mu);
    ctx->ready = true;
    pthread_cond_signal(&ctx->cv);
    pthread_mutex_unlock(&ctx->mu);
    size_t cap = 4096;
    ctx->data = malloc(cap);
    if (!ctx->data) return NULL;
    for (;;) {
        if (ctx->len + 1 == cap) {
            cap *= 2;
            char *grown = realloc(ctx->data, cap);
            if (!grown) {
                free(ctx->data);
                ctx->data = NULL;
                ctx->len = 0;
                return NULL;
            }
            ctx->data = grown;
        }
        ssize_t n = read(ctx->fd, ctx->data + ctx->len,
                         cap - ctx->len - 1);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        ctx->len += (size_t)n;
    }
    ctx->data[ctx->len] = '\0';
    return NULL;
}

static char *emit_and_read(const ember_chat_request *req,
                           const ember_protocol_result *result) {
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) return NULL;
    reader_ctx ctx = {
        .fd = sv[1],
        .mu = PTHREAD_MUTEX_INITIALIZER,
        .cv = PTHREAD_COND_INITIALIZER,
    };
    pthread_t reader;
    if (pthread_create(&reader, NULL, read_response, &ctx) != 0) {
        close(sv[0]);
        close(sv[1]);
        return NULL;
    }
    pthread_mutex_lock(&ctx.mu);
    while (!ctx.ready) pthread_cond_wait(&ctx.cv, &ctx.mu);
    pthread_mutex_unlock(&ctx.mu);
    CHECK(ember_protocol_emit(sv[0], req, result));
    shutdown(sv[0], SHUT_WR);
    pthread_join(reader, NULL);
    close(sv[0]);
    close(sv[1]);
    pthread_cond_destroy(&ctx.cv);
    pthread_mutex_destroy(&ctx.mu);
    return ctx.data;
}

static void check_responses_frames(const char *response) {
    const char *p = response;
    int expected_sequence = 0;
    while ((p = strstr(p, "data: {")) != NULL) {
        p += 6;
        const char *end = strchr(p, '\n');
        CHECK(end != NULL);
        if (!end) return;
        size_t n = (size_t)(end - p);
        char *line = malloc(n + 1);
        CHECK(line != NULL);
        if (!line) return;
        memcpy(line, p, n);
        line[n] = '\0';
        ember_json *event_json = ember_json_parse(line);
        CHECK(event_json != NULL);
        if (event_json) {
            const ember_json *seq =
                ember_json_get(event_json, "sequence_number");
            CHECK(seq && seq->type == EMBER_JSON_NUMBER);
            CHECK(seq && seq->u.num == (double)expected_sequence);
            expected_sequence++;
            ember_json_free(event_json);
        }
        free(line);
        p = end + 1;
    }
    CHECK(expected_sequence > 1);
}

static void test_responses(void) {
    const char *body =
        "{\"model\":\"deepseek-v4-flash\",\"stream\":true,"
        "\"instructions\":\"follow the repository rules\","
        "\"max_output_tokens\":123,"
        "\"thinking_token_budget\":55,"
        "\"reasoning\":{\"effort\":\"high\",\"summary\":\"auto\"},"
        "\"input\":["
        "{\"type\":\"reasoning\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"why\"}]},"
        "{\"type\":\"function_call\",\"call_id\":\"call_a\",\"name\":\"bash\","
        "\"arguments\":\"{\\\"command\\\":\\\"pwd\\\"}\"},"
        "{\"type\":\"function_call_output\",\"call_id\":\"call_a\",\"output\":\"/tmp\"}],"
        "\"tools\":[{\"type\":\"function\",\"name\":\"bash\","
        "\"parameters\":{\"type\":\"object\"}}]}";
    ember_json *j = ember_json_parse(body);
    ember_chat_request r;
    char err[160] = {0};
    CHECK(j && ember_responses_request_parse(j, &r, err, sizeof(err)));
    CHECK(r.api == EMBER_API_RESPONSES);
    CHECK(r.stream && r.max_tokens_set && r.max_tokens == 123);
    CHECK(r.reasoning_budget_tokens_set &&
          r.reasoning_budget_tokens == 55);
    CHECK(r.reasoning_summary_emit);
    CHECK(r.n_messages == 3);
    CHECK(!strcmp(r.messages[0].role, "system"));
    CHECK(!strcmp(r.messages[0].content, "follow the repository rules"));
    CHECK(r.messages[1].calls.len == 1);
    CHECK(!strcmp(r.messages[1].calls.calls[0].id, "call_a"));
    CHECK(!strcmp(r.messages[2].tool_call_id, "call_a"));
    CHECK(!r.continuation_only);
    CHECK(r.has_tools && strstr(r.tools_json, "\"function\""));
    ember_chat_request_free(&r);
    ember_json_free(j);

    j = ember_json_parse(
        "{\"input\":[{\"type\":\"function_call_output\","
        "\"call_id\":\"call_only\",\"output\":\"ok\"}]}");
    CHECK(j && ember_responses_request_parse(j, &r, err, sizeof(err)));
    CHECK(r.continuation_only && r.n_messages == 1);
    ember_chat_request_free(&r);
    ember_json_free(j);

    j = ember_json_parse(
        "{\"previous_response_id\":\"resp_prior\",\"input\":["
        "{\"type\":\"function_call_output\",\"call_id\":\"call_only\","
        "\"output\":\"ok\"}]}");
    CHECK(j && ember_responses_request_parse(j, &r, err, sizeof(err)));
    CHECK(r.continuation_only &&
          !strcmp(r.continuation_key, "resp_prior"));
    ember_chat_request_free(&r);
    ember_json_free(j);

    j = ember_json_parse(
        "{\"input\":["
        "{\"type\":\"tool_search_call\",\"call_id\":\"search_1\","
        "\"arguments\":{\"query\":\"perplexity\"}},"
        "{\"type\":\"tool_search_output\",\"call_id\":\"search_1\","
        "\"status\":\"completed\",\"tools\":["
        "{\"type\":\"namespace\",\"name\":\"mcp__perplexity__\","
        "\"tools\":[{\"type\":\"function\",\"name\":\"search\","
        "\"parameters\":{\"type\":\"object\"}}]}]}]}");
    memset(err, 0, sizeof(err));
    CHECK(j && ember_responses_request_parse(j, &r, err, sizeof(err)));
    CHECK(r.n_messages == 2);
    CHECK(r.messages[0].calls.len == 1);
    CHECK(!strcmp(r.messages[0].calls.calls[0].name, "tool_search"));
    CHECK(strstr(r.messages[1].content, "mcp__perplexity__") != NULL);
    CHECK(r.has_tools);
    CHECK(strstr(r.tools_json, "mcp__perplexity__search") != NULL);
    ember_chat_request_free(&r);
    ember_json_free(j);

    j = ember_json_parse(
        "{\"input\":[{\"type\":\"local_shell_call\","
        "\"call_id\":\"shell_1\",\"action\":{\"command\":\"pwd\"}},"
        "{\"type\":\"local_shell_call_output\","
        "\"call_id\":\"shell_1\",\"output\":\"/tmp\"}]}");
    memset(err, 0, sizeof(err));
    CHECK(j && ember_responses_request_parse(j, &r, err, sizeof(err)));
    CHECK(r.messages[0].calls.len == 1);
    CHECK(!strcmp(r.messages[0].calls.calls[0].name, "local_shell"));
    CHECK(strstr(r.messages[0].calls.calls[0].arguments, "pwd") != NULL);
    CHECK(!strcmp(r.messages[1].content, "/tmp"));
    ember_chat_request_free(&r);
    ember_json_free(j);

    j = ember_json_parse(
        "{\"input\":[{\"type\":\"unknown_future_item\",\"value\":1}]}");
    memset(err, 0, sizeof(err));
    CHECK(j && !ember_responses_request_parse(j, &r, err, sizeof(err)));
    CHECK(strstr(err, "unsupported Responses input item type") != NULL);
    ember_json_free(j);

    j = ember_json_parse(
        "{\"input\":[{\"type\":\"message\",\"role\":\"user\","
        "\"content\":[{\"type\":\"input_text\",\"text\":\"look\"},"
        "{\"type\":\"input_image\",\"image_url\":\"https://example.invalid/x.png\"}]}]}");
    memset(err, 0, sizeof(err));
    CHECK(j && !ember_responses_request_parse(j, &r, err, sizeof(err)));
    CHECK(strstr(err, "image inputs are not available") != NULL);
    ember_json_free(j);

    j = ember_json_parse(
        "{\"input\":[{\"type\":\"message\",\"role\":\"assistant\","
        "\"status\":\"in_progress\",\"content\":\"partial\"}]}");
    memset(err, 0, sizeof(err));
    CHECK(j && !ember_responses_request_parse(j, &r, err, sizeof(err)));
    CHECK(strstr(err, "completed status") != NULL);
    ember_json_free(j);

    j = ember_json_parse(
        "{\"tool_choice\":\"required\",\"input\":\"call something\","
        "\"tools\":[{\"type\":\"function\",\"name\":\"bash\","
        "\"parameters\":{\"type\":\"object\"}}]}");
    memset(err, 0, sizeof(err));
    CHECK(j && ember_responses_request_parse(j, &r, err, sizeof(err)));
    CHECK(r.tool_choice_required && r.has_tools);
    ember_chat_request_free(&r);
    ember_json_free(j);

    j = ember_json_parse(
        "{\"instructions\":{\"bad\":true},\"input\":\"hello\"}");
    memset(err, 0, sizeof(err));
    CHECK(j && !ember_responses_request_parse(j, &r, err, sizeof(err)));
    CHECK(strstr(err, "instructions must be a string") != NULL);
    ember_json_free(j);
}

static void test_anthropic(void) {
    const char *body =
        "{\"model\":\"deepseek-v4-flash\",\"max_tokens\":200,\"stream\":true,"
        "\"system\":\"be precise\",\"messages\":["
        "{\"role\":\"assistant\",\"content\":["
        "{\"type\":\"thinking\",\"thinking\":\"why\"},"
        "{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"bash\","
        "\"input\":{\"command\":\"pwd\"}}]},"
        "{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\","
        "\"tool_use_id\":\"toolu_1\",\"content\":\"/tmp\"}]}],"
        "\"tools\":[{\"name\":\"bash\",\"input_schema\":{\"type\":\"object\"}}]}";
    ember_json *j = ember_json_parse(body);
    ember_chat_request r;
    char err[160] = {0};
    CHECK(j && ember_anthropic_request_parse(j, &r, err, sizeof(err)));
    CHECK(r.api == EMBER_API_ANTHROPIC);
    CHECK(r.n_messages == 3);
    CHECK(!strcmp(r.messages[0].role, "system"));
    CHECK(r.messages[1].calls.len == 1);
    CHECK(!strcmp(r.messages[2].tool_call_id, "toolu_1"));
    CHECK(!r.continuation_only);
    CHECK(r.has_tools);
    ember_chat_request_free(&r);
    ember_json_free(j);

    j = ember_json_parse(
        "{\"max_tokens\":20,\"tool_choice\":{\"type\":\"any\","
        "\"disable_parallel_tool_use\":true},"
        "\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":12},"
        "\"messages\":[{\"role\":\"user\",\"content\":\"use a tool\"}],"
        "\"tools\":[{\"name\":\"bash\",\"input_schema\":{"
        "\"type\":\"object\"}}]}");
    memset(err, 0, sizeof(err));
    CHECK(j && ember_anthropic_request_parse(j, &r, err, sizeof(err)));
    CHECK(r.tool_choice_required && !r.parallel_tool_calls);
    CHECK(r.reasoning_budget_tokens_set && r.reasoning_budget_tokens == 12);
    ember_chat_request_free(&r);
    ember_json_free(j);

    j = ember_json_parse(
        "{\"max_tokens\":20,\"messages\":[{\"role\":\"user\",\"content\":["
        "{\"type\":\"tool_result\",\"tool_use_id\":\"toolu_only\","
        "\"content\":\"ok\"}]}]}");
    CHECK(j && ember_anthropic_request_parse(j, &r, err, sizeof(err)));
    CHECK(r.continuation_only);
    ember_chat_request_free(&r);
    ember_json_free(j);

    j = ember_json_parse(
        "{\"messages\":[null],\"tools\":[{\"name\":\"broken\"}]}");
    memset(err, 0, sizeof(err));
    CHECK(j && !ember_anthropic_request_parse(j, &r, err, sizeof(err)));
    CHECK(strstr(err, "input_schema") != NULL);
    ember_json_free(j);

    j = ember_json_parse(
        "{\"messages\":[{\"role\":\"user\",\"content\":["
        "{\"type\":\"tool_use\",\"id\":\"x\",\"name\":\"bad\",\"input\":{}}]}]}");
    memset(err, 0, sizeof(err));
    CHECK(j && !ember_anthropic_request_parse(j, &r, err, sizeof(err)));
    CHECK(strstr(err, "unsupported Anthropic content block") != NULL);
    ember_json_free(j);
}

static void test_completions(void) {
    ember_json *j = ember_json_parse(
        "{\"model\":\"deepseek-v4-flash\",\"prompt\":[\"abc\"],"
        "\"max_tokens\":12,\"stream\":true,\"stop\":\"END\"}");
    ember_chat_request r;
    char err[160] = {0};
    CHECK(j && ember_completion_request_parse(j, &r, err, sizeof(err)));
    CHECK(r.api == EMBER_API_COMPLETIONS);
    CHECK(r.raw_prompt && !strcmp(r.raw_prompt, "abc"));
    CHECK(r.stream && r.max_tokens_set && r.max_tokens == 12);
    CHECK(!r.thinking_enabled && r.n_stop == 1);
    ember_chat_request_free(&r);
    ember_json_free(j);

    j = ember_json_parse("{\"prompt\":[\"abc\",\"ignored\"]}");
    memset(err, 0, sizeof(err));
    CHECK(j && !ember_completion_request_parse(j, &r, err, sizeof(err)));
    CHECK(strstr(err, "single-element") != NULL);
    ember_json_free(j);
}

static void test_emitters(void) {
    ember_tool_call call = {
        .id = "call_1", .name = "bash",
        .arguments = "{\"command\":\"pwd\"}",
    };
    ember_tool_calls calls = {.calls = &call, .len = 1, .cap = 1};
    ember_protocol_result result = {
        .id = "resp_test", .model = "deepseek-v4-flash", .created = 1,
        .content = "done", .reasoning = "why", .calls = &calls,
        .finish_reason = "tool_calls", .prompt_tokens = 10,
        .completion_tokens = 5, .cached_tokens = 7,
        .tool_loop_rounds = 4, .tool_loop_identical = true,
        .tool_loop_tool = "bash",
    };
    ember_chat_request req = {
        .api = EMBER_API_RESPONSES,
        .stream = true,
        .reasoning_summary_emit = false,
    };
    char *response = emit_and_read(&req, &result);
    CHECK(response != NULL);
    if (response) {
        check_responses_frames(response);
        CHECK(strstr(response, "event: response.") == NULL);
        CHECK(strstr(response, "\"status\":\"in_progress\"") != NULL);
        CHECK(strstr(response, "\"sequence_number\":0") != NULL);
        CHECK(strstr(response,
                     "response.function_call_arguments.delta") != NULL);
        CHECK(strstr(response, "response.output_item.done") != NULL);
        CHECK(strstr(response, "response.completed") != NULL);
        CHECK(strstr(response,
                     "\"ember_tool_loop\":{\"rounds\":4,"
                     "\"tool\":\"bash\",\"identical_results\":true}") != NULL);
        CHECK(strstr(response, "reasoning_summary") == NULL);
    }
    free(response);

    req.reasoning_summary_emit = true;
    response = emit_and_read(&req, &result);
    CHECK(response != NULL);
    if (response) {
        CHECK(strstr(response,
                     "response.reasoning_summary_text.delta") != NULL);
    }
    free(response);

    req.api = EMBER_API_ANTHROPIC;
    response = emit_and_read(&req, &result);
    CHECK(response != NULL);
    if (response) {
        CHECK(strstr(response,
                     "\"content\":[],\"stop_reason\":null") != NULL);
        CHECK(strstr(response, "\"type\":\"input_json_delta\"") != NULL);
        CHECK(strstr(response, "\"stop_reason\":\"tool_use\"") != NULL);
        CHECK(strstr(response, "\"name\":\"bash\"") != NULL);
        CHECK(strstr(response,
                     "\"ember_tool_loop\":{\"rounds\":4,"
                     "\"tool\":\"bash\",\"identical_results\":true}") != NULL);
        CHECK(strstr(response, "event: message_stop") != NULL);
        CHECK(strstr(response, "\"signature\":\"\"") == NULL);
        CHECK(strstr(response, "thinking_delta") == NULL);
    }
    free(response);

    result.calls = NULL;
    result.tool_loop_rounds = 0;
    result.tool_loop_tool = NULL;
    result.stop_sequence = "END";
    result.finish_reason = "stop";
    response = emit_and_read(&req, &result);
    CHECK(response != NULL);
    if (response) {
        CHECK(strstr(response, "\"stop_reason\":\"stop_sequence\"") != NULL);
        CHECK(strstr(response, "\"stop_sequence\":\"END\"") != NULL);
    }
    free(response);
    result.calls = &calls;
    result.stop_sequence = NULL;

    req.api = EMBER_API_COMPLETIONS;
    response = emit_and_read(&req, &result);
    CHECK(response != NULL);
    if (response) {
        CHECK(strstr(response, "\"object\":\"text_completion\"") != NULL);
        CHECK(strstr(response, "\"text\":\"done\"") != NULL);
        CHECK(strstr(response, "data: [DONE]") != NULL);
    }
    free(response);
}

// ── buffered (stream=false) emitters ────────────────────────────────────
//
// ember_protocol_emit dispatches `stream` FIRST, so emit_responses /
// emit_anthropic / emit_completion are reached only when stream is false —
// which test_emitters never exercises, because it sets stream=true throughout.
// These are the buffered JSON responses every non-streaming client receives.
//
// The body is parsed rather than substring-matched: a malformed envelope that
// still happens to contain the right substrings is exactly the regression a
// protocol test is supposed to catch.

static const char *http_body(const char *response) {
    const char *sep = response ? strstr(response, "\r\n\r\n") : NULL;
    return sep ? sep + 4 : NULL;
}

static ember_json *emit_and_parse(const ember_chat_request *req,
                                  const ember_protocol_result *result,
                                  char **raw_out) {
    char *raw = emit_and_read(req, result);
    if (raw_out) *raw_out = raw;
    const char *body = http_body(raw);
    ember_json *parsed = body ? ember_json_parse(body) : NULL;
    if (!raw_out) free(raw);
    return parsed;
}

// Never returns NULL, so callers can strcmp directly. A missing key and an
// empty-string value are both "" here; every assertion below expects a real
// value, so conflating them cannot mask a failure.
static const char *str_at(const ember_json *o, const char *key) {
    return ember_json_str(ember_json_get(o, key), "");
}

static void test_buffered_responses(void) {
    ember_tool_call call = {
        .id = "call_1", .name = "bash", .arguments = "{\"command\":\"pwd\"}",
    };
    ember_tool_calls calls = {.calls = &call, .len = 1, .cap = 1};
    ember_protocol_result result = {
        .id = "resp_buf", .model = "deepseek-v4-flash", .created = 7,
        .content = "done", .reasoning = "why", .calls = &calls,
        .finish_reason = "tool_calls", .prompt_tokens = 10,
        .completion_tokens = 5, .cached_tokens = 7,
    };
    ember_chat_request req = {
        .api = EMBER_API_RESPONSES, .stream = false,
        .reasoning_summary_emit = false,
    };
    char *raw = NULL;
    ember_json *body = emit_and_parse(&req, &result, &raw);
    CHECK(raw && strstr(raw, "Content-Type: application/json") != NULL);
    // A buffered response must not carry SSE framing.
    CHECK(raw && strstr(raw, "event: ") == NULL);
    CHECK(raw && strstr(raw, "data: ") == NULL);
    CHECK(body != NULL);
    if (body) {
        CHECK(!strcmp(str_at(body, "object"), "response"));
        CHECK(!strcmp(str_at(body, "id"), "resp_buf"));
        CHECK(!strcmp(str_at(body, "status"), "completed"));
        const ember_json *incomplete =
            ember_json_get(body, "incomplete_details");
        CHECK(incomplete && incomplete->type == EMBER_JSON_NULL);
        const ember_json *output = ember_json_get(body, "output");
        CHECK(output && output->type == EMBER_JSON_ARRAY);
        CHECK(output && ember_json_len(output) >= 1);
        // The function_call item carries arguments as a JSON *string*, which is
        // what the Responses API specifies (Anthropic re-dumps them instead).
        bool saw_call = false;
        for (int i = 0; output && i < ember_json_len(output); ++i) {
            const ember_json *item = ember_json_at(output, i);
            if (item && !strcmp(str_at(item, "type"), "function_call")) {
                saw_call = true;
                CHECK(!strcmp(str_at(item, "call_id"), "call_1"));
                CHECK(!strcmp(str_at(item, "name"), "bash"));
                const ember_json *args = ember_json_get(item, "arguments");
                CHECK(args && args->type == EMBER_JSON_STRING);
                CHECK(args && !strcmp(args->u.str, "{\"command\":\"pwd\"}"));
            }
        }
        CHECK(saw_call);
        ember_json_free(body);
    }
    free(raw);

    // finish_reason=length flips status and populates incomplete_details.
    result.finish_reason = "length";
    result.termination_reason = "";
    body = emit_and_parse(&req, &result, NULL);
    CHECK(body != NULL);
    if (body) {
        CHECK(!strcmp(str_at(body, "status"), "incomplete"));
        const ember_json *d = ember_json_get(body, "incomplete_details");
        CHECK(d && d->type == EMBER_JSON_OBJECT);
        CHECK(!strcmp(str_at(d, "reason"), "max_output_tokens"));
        ember_json_free(body);
    }

    // A typed termination reason replaces the default.
    result.termination_reason = "degenerate_decode";
    body = emit_and_parse(&req, &result, NULL);
    CHECK(body != NULL);
    if (body) {
        const ember_json *d = ember_json_get(body, "incomplete_details");
        CHECK(!strcmp(str_at(d, "reason"), "degenerate_decode"));
        ember_json_free(body);
    }
}

static void test_buffered_anthropic(void) {
    ember_tool_call call = {
        .id = "toolu_1", .name = "bash", .arguments = "{\"command\":\"pwd\"}",
    };
    ember_tool_calls calls = {.calls = &call, .len = 1, .cap = 1};
    ember_protocol_result result = {
        .id = "msg_buf", .model = "deepseek-v4-flash", .created = 3,
        .content = "running", .reasoning = "why", .calls = &calls,
        .finish_reason = "tool_calls", .prompt_tokens = 4,
        .completion_tokens = 2,
    };
    ember_chat_request req = {.api = EMBER_API_ANTHROPIC, .stream = false};

    ember_json *body = emit_and_parse(&req, &result, NULL);
    CHECK(body != NULL);
    if (body) {
        CHECK(!strcmp(str_at(body, "type"), "message"));
        CHECK(!strcmp(str_at(body, "role"), "assistant"));
        CHECK(!strcmp(str_at(body, "stop_reason"), "tool_use"));
        const ember_json *content = ember_json_get(body, "content");
        CHECK(content && content->type == EMBER_JSON_ARRAY);
        CHECK(content && ember_json_len(content) == 2);
        const ember_json *text = ember_json_at(content, 0);
        CHECK(!strcmp(str_at(text, "type"), "text"));
        CHECK(!strcmp(str_at(text, "text"), "running"));
        const ember_json *use = ember_json_at(content, 1);
        CHECK(!strcmp(str_at(use, "type"), "tool_use"));
        CHECK(!strcmp(str_at(use, "id"), "toolu_1"));
        CHECK(!strcmp(str_at(use, "name"), "bash"));
        // Anthropic `input` is a re-parsed OBJECT, not the raw argument string.
        const ember_json *input = ember_json_get(use, "input");
        CHECK(input && input->type == EMBER_JSON_OBJECT);
        CHECK(!strcmp(str_at(input, "command"), "pwd"));
        // Reasoning must never be emitted as an authenticated thinking block.
        char *dump = ember_json_dump(body);
        CHECK(dump && !strstr(dump, "\"thinking\""));
        free(dump);
        ember_json_free(body);
    }

    // Unparseable arguments must degrade to an empty object, never leak the raw
    // string into a field the client will treat as structured input.
    call.arguments = "not json";
    body = emit_and_parse(&req, &result, NULL);
    CHECK(body != NULL);
    if (body) {
        const ember_json *use = ember_json_at(ember_json_get(body, "content"), 1);
        const ember_json *input = ember_json_get(use, "input");
        CHECK(input && input->type == EMBER_JSON_OBJECT);
        CHECK(input && ember_json_len(input) == 0);
        ember_json_free(body);
    }
    call.arguments = "{\"command\":\"pwd\"}";

    // Tool-call-only turn: the content array must start with the tool_use
    // block, with no stray leading comma from the skipped text block.
    result.content = "";
    body = emit_and_parse(&req, &result, NULL);
    CHECK(body != NULL);
    if (body) {
        const ember_json *content = ember_json_get(body, "content");
        CHECK(content && ember_json_len(content) == 1);
        CHECK(!strcmp(str_at(ember_json_at(content, 0), "type"),
                      "tool_use"));
        ember_json_free(body);
    }
    result.content = "running";

    // stop_reason mapping across the enum. Tool calls win over everything.
    struct { const char *finish; const char *stop_seq; bool calls_present;
             const char *want; } cases[] = {
        {"tool_calls", NULL, true,  "tool_use"},
        {"length",     NULL, true,  "tool_use"},
        {"length",     NULL, false, "max_tokens"},
        {"stop",       "END", false, "stop_sequence"},
        {"stop",       NULL, false, "end_turn"},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        result.finish_reason = cases[i].finish;
        result.stop_sequence = cases[i].stop_seq;
        result.calls = cases[i].calls_present ? &calls : NULL;
        body = emit_and_parse(&req, &result, NULL);
        CHECK(body && !strcmp(str_at(body, "stop_reason"),
                              cases[i].want));
        if (body) ember_json_free(body);
    }
}

static void test_buffered_completions(void) {
    ember_protocol_result result = {
        .id = "cmpl-buf", .model = "deepseek-v4-flash", .created = 5,
        .content = "hello", .finish_reason = "stop",
        .prompt_tokens = 2, .completion_tokens = 1,
    };
    ember_chat_request req = {.api = EMBER_API_COMPLETIONS, .stream = false};
    char *raw = NULL;
    ember_json *body = emit_and_parse(&req, &result, &raw);
    CHECK(raw && strstr(raw, "data: [DONE]") == NULL);
    CHECK(body != NULL);
    if (body) {
        CHECK(!strcmp(str_at(body, "object"), "text_completion"));
        const ember_json *choices = ember_json_get(body, "choices");
        CHECK(choices && choices->type == EMBER_JSON_ARRAY);
        CHECK(choices && ember_json_len(choices) == 1);
        const ember_json *c0 = ember_json_at(choices, 0);
        CHECK(!strcmp(str_at(c0, "text"), "hello"));
        CHECK(!strcmp(str_at(c0, "finish_reason"), "stop"));
        ember_json_free(body);
    }
    free(raw);
}

static void test_anthropic_text_block_validation(void) {
    ember_chat_request r;
    char err[160];
    // `system` may be a plain string or an array of {type:"text"} blocks.
    ember_json *j = ember_json_parse(
        "{\"max_tokens\":10,\"system\":[{\"type\":\"text\",\"text\":\"a\"},"
        "{\"type\":\"text\",\"text\":\"b\"}],"
        "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}");
    memset(err, 0, sizeof(err));
    CHECK(j && ember_anthropic_request_parse(j, &r, err, sizeof(err)));
    if (j) ember_json_free(j);
    ember_chat_request_free(&r);

    // A non-text block in `system` must be rejected, not silently dropped —
    // dropping it would quietly discard part of the operator's instructions.
    j = ember_json_parse(
        "{\"max_tokens\":10,\"system\":[{\"type\":\"image\",\"text\":\"a\"}],"
        "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}");
    memset(err, 0, sizeof(err));
    CHECK(j && !ember_anthropic_request_parse(j, &r, err, sizeof(err)));
    CHECK(strstr(err, "type=text") != NULL);
    if (j) ember_json_free(j);

    j = ember_json_parse(
        "{\"max_tokens\":10,\"system\":[{\"type\":\"text\",\"text\":5}],"
        "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}");
    memset(err, 0, sizeof(err));
    CHECK(j && !ember_anthropic_request_parse(j, &r, err, sizeof(err)));
    CHECK(strstr(err, "type=text") != NULL);
    if (j) ember_json_free(j);

    j = ember_json_parse(
        "{\"max_tokens\":10,\"system\":7,"
        "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}");
    memset(err, 0, sizeof(err));
    CHECK(j && !ember_anthropic_request_parse(j, &r, err, sizeof(err)));
    CHECK(strstr(err, "string or block array") != NULL);
    if (j) ember_json_free(j);
}

static void append_and_clear(ember_buf *all, ember_buf *part) {
    ember_buf_append(all, part->ptr ? part->ptr : "", part->len);
    part->len = 0;
    if (part->ptr) part->ptr[0] = '\0';
}

static void test_live_protocol_streams(void) {
    ember_chat_request req = {
        .api = EMBER_API_RESPONSES,
        .stream = true,
        .has_tools = true,
        .reasoning_summary_emit = true,
    };
    ember_sse_stream split;
    ember_sse_init(&split, "resp_live", "deepseek-v4-flash", 2,
                   true, true, false);
    ember_protocol_stream native;
    ember_protocol_stream_init(
        &native, &req, "resp_live", "deepseek-v4-flash", 2, 11);
    ember_protocol_stream_bind(&native, &split);
    ember_buf part = {0}, all = {0};
    ember_protocol_stream_begin(&native, &part);
    CHECK(strstr(part.ptr, "response.created") != NULL);
    append_and_clear(&all, &part);

    const char *reasoning = "because this";
    ember_sse_update(&split, reasoning, strlen(reasoning), false, &part);
    CHECK(strstr(part.ptr, "response.reasoning_summary_text.delta") != NULL);
    CHECK(strstr(part.ptr, "response.completed") == NULL);
    append_and_clear(&all, &part);

    const char *text = "because this</think>Hello";
    ember_sse_update(&split, text, strlen(text), false, &part);
    CHECK(strstr(part.ptr, "response.reasoning_summary_part.done") != NULL);
    CHECK(strstr(part.ptr, "response.output_text.delta") != NULL);
    append_and_clear(&all, &part);

    const char *with_tool =
        "because this</think>Hello\n\n"
        "<" PIPE "DSML" PIPE "tool_calls>"
        "<" PIPE "DSML" PIPE "invoke name=\"get_weather\">"
        "<" PIPE "DSML" PIPE "parameter name=\"city\" string=\"true\">Tokyo"
        "</" PIPE "DSML" PIPE "parameter>"
        "</" PIPE "DSML" PIPE "invoke>"
        "</" PIPE "DSML" PIPE "tool_calls>";
    ember_sse_update(
        &split, with_tool, strlen(with_tool), false, &part);
    CHECK(strstr(part.ptr ? part.ptr : "",
                 "response.function_call_arguments.delta") == NULL);
    CHECK(strstr(part.ptr ? part.ptr : "",
                 "\"type\":\"function_call\"") == NULL);
    CHECK(strstr(part.ptr, "response.completed") == NULL);
    append_and_clear(&all, &part);

    ember_sse_update(&split, with_tool, strlen(with_tool), true, &part);
    CHECK(ember_sse_emit_tools(
        &split, with_tool, strlen(with_tool), &part));
    CHECK(strstr(part.ptr, "response.function_call_arguments.delta") != NULL);
    CHECK(strstr(part.ptr, "\"status\":\"in_progress\"") != NULL);
    ember_tool_call call = {
        .id = split.n_tool_ids ? split.tool_ids[0] : "call_missing",
        .name = "get_weather",
        .arguments = "{\"city\":\"Tokyo\"}",
    };
    ember_tool_calls calls = {.calls = &call, .len = 1, .cap = 1};
    ember_protocol_result result = {
        .id = "resp_live",
        .model = "deepseek-v4-flash",
        .created = 2,
        .content = "Hello",
        .reasoning = "because this",
        .calls = &calls,
        .finish_reason = "tool_calls",
        .prompt_tokens = 11,
        .completion_tokens = 7,
        .cached_tokens = 5,
        .tool_loop_rounds = 4, .tool_loop_identical = true,
        .tool_loop_tool = "get_weather",
    };
    ember_protocol_stream_finish(&native, &result, &part);
    CHECK(strstr(part.ptr, "response.function_call_arguments.done") != NULL);
    CHECK(strstr(part.ptr, "response.output_item.done") != NULL);
    CHECK(strstr(part.ptr, "response.completed") != NULL);
    CHECK(strstr(part.ptr,
                 "\"ember_tool_loop\":{\"rounds\":4,"
                 "\"tool\":\"get_weather\","
                 "\"identical_results\":true}") != NULL);
    CHECK(strstr(part.ptr, "\"id\":\"fc_live_0\"") != NULL);
    CHECK(strstr(part.ptr, "\"item_id\":\"fc_live_0\"") != NULL);
    append_and_clear(&all, &part);
    check_responses_frames(all.ptr);
    ember_buf_free(&all);
    ember_buf_free(&part);
    ember_protocol_stream_free(&native);
    ember_sse_free(&split);

    req.api = EMBER_API_ANTHROPIC;
    req.has_tools = false;
    req.reasoning_summary_emit = false;
    ember_sse_init(&split, "msg_live", "deepseek-v4-flash", 3,
                   false, false, false);
    ember_protocol_stream_init(
        &native, &req, "msg_live", "deepseek-v4-flash", 3, 4);
    ember_protocol_stream_bind(&native, &split);
    ember_protocol_stream_begin(&native, &part);
    CHECK(strstr(part.ptr, "event: message_start") != NULL);
    part.len = 0; if (part.ptr) part.ptr[0] = '\0';
    ember_sse_update(&split, "hel", 3, false, &part);
    CHECK(strstr(part.ptr, "\"type\":\"text_delta\"") != NULL);
    CHECK(strstr(part.ptr, "message_stop") == NULL);
    part.len = 0; if (part.ptr) part.ptr[0] = '\0';
    ember_sse_update(&split, "hello", 5, true, &part);
    ember_protocol_result anthropic_result = {
        .id = "msg_live",
        .model = "deepseek-v4-flash",
        .created = 3,
        .content = "hello",
        .finish_reason = "stop",
        .prompt_tokens = 4,
        .completion_tokens = 2,
    };
    ember_protocol_stream_finish(&native, &anthropic_result, &part);
    CHECK(strstr(part.ptr, "event: content_block_stop") != NULL);
    CHECK(strstr(part.ptr, "event: message_delta") != NULL);
    CHECK(strstr(part.ptr, "\"usage\":{\"output_tokens\":2}") != NULL);
    CHECK(strstr(part.ptr, "\"input_tokens\"") == NULL);
    CHECK(strstr(part.ptr, "event: message_stop") != NULL);
    ember_buf_free(&part);
    ember_protocol_stream_free(&native);
    ember_sse_free(&split);

    req.api = EMBER_API_RESPONSES;
    ember_protocol_stream_init(
        &native, &req, "msg_error", "deepseek-v4-flash", 4, 1);
    ember_protocol_stream_error(
        &native, "backend_error", "generation failed", false, &part);
    CHECK(strstr(part.ptr, "event: error") == NULL);
    CHECK(strstr(part.ptr, "\"type\":\"error\"") != NULL);
    CHECK(strstr(part.ptr, "\"sequence_number\":0") != NULL);
    CHECK(strstr(part.ptr, "\"code\":\"backend_error\"") != NULL);
    CHECK(strstr(part.ptr, "generation failed") != NULL);
    ember_buf_free(&part);
    ember_protocol_stream_free(&native);

    req.api = EMBER_API_ANTHROPIC;
    ember_protocol_stream_init(
        &native, &req, "msg_error", "deepseek-v4-flash", 4, 1);
    ember_protocol_stream_error(
        &native, "backend_error", "generation failed", false, &part);
    CHECK(strstr(part.ptr, "event: error") != NULL);
    CHECK(strstr(part.ptr, "\"type\":\"api_error\"") != NULL);
    ember_buf_free(&part);
    ember_protocol_stream_free(&native);
}

int main(void) {
    test_responses();
    test_anthropic();
    test_completions();
    test_emitters();
    test_buffered_responses();
    test_buffered_anthropic();
    test_buffered_completions();
    test_anthropic_text_block_validation();
    test_live_protocol_streams();
    printf("api adapter tests: %d passed, %d failed\n", pass, fail);
    return fail != 0;
}
