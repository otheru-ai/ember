#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/common/json.h"
#include "../src/model/chat_template.h"
#include "../src/server/chat_api.h"

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, msg)                                                    \
    do { if (cond) g_pass++; else { g_fail++; printf("  FAIL: %s\n", msg); } } while (0)

#define PIPE   "\xef\xbd\x9c"
#define USCORE "\xe2\x96\x81"
#define BOS  "<" PIPE "begin" USCORE "of" USCORE "sentence" PIPE ">"
#define USER "<" PIPE "User" PIPE ">"
#define ASST "<" PIPE "Assistant" PIPE ">"
#define EOS  "<" PIPE "end" USCORE "of" USCORE "sentence" PIPE ">"

static ember_chat_request parse(const char *body) {
    ember_json *v = ember_json_parse(body);
    ember_chat_request req;
    ember_chat_request_parse(v, &req);
    ember_json_free(v);
    return req;
}

static void test_basic_turns(void) {
    ember_chat_request req = parse(
        "{\"messages\":[{\"role\":\"system\",\"content\":\"be nice\"},"
        "{\"role\":\"user\",\"content\":\"hi\"},"
        "{\"role\":\"assistant\",\"content\":\"hello\"},"
        "{\"role\":\"user\",\"content\":\"bye\"}]}");
    char *p = ember_render_prompt(&req, false, EMBER_THINK_NONE, true);
    CHECK(strncmp(p, BOS "be nice" USER "hi", strlen(BOS "be nice" USER "hi")) == 0,
          "bos + system + first user");
    CHECK(strstr(p, ASST "</think>hello" EOS) != NULL,
          "assistant turn with </think> + EOS");
    CHECK(strstr(p, USER "bye") != NULL, "second user turn");
    // ends with generation prompt (thinking disabled → </think>)
    size_t n = strlen(p);
    CHECK(n >= strlen(ASST "</think>") &&
          strcmp(p + n - strlen(ASST "</think>"), ASST "</think>") == 0,
          "generation prompt appended");
    free(p);
    ember_chat_request_free(&req);
}

static void test_thinking_opens_think(void) {
    ember_chat_request req = parse(
        "{\"messages\":[{\"role\":\"user\",\"content\":\"q\"}]}");
    char *p = ember_render_prompt(&req, /*enable_thinking=*/true, EMBER_THINK_HIGH, true);
    CHECK(ember_prompt_ends_in_open_think(p), "thinking prompt ends in open <think>");
    char *p2 = ember_render_prompt(&req, false, EMBER_THINK_NONE, true);
    CHECK(!ember_prompt_ends_in_open_think(p2), "non-thinking does not");
    free(p); free(p2);
    ember_chat_request_free(&req);
}

static void test_tool_preamble_and_result(void) {
    ember_chat_request req = parse(
        "{\"messages\":["
        "{\"role\":\"user\",\"content\":\"time?\"},"
        "{\"role\":\"tool\",\"name\":\"get_time\",\"content\":\"12:00\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":"
        "{\"name\":\"get_time\",\"description\":\"clock\"}}]}");
    char *p = ember_render_prompt(&req, false, EMBER_THINK_NONE, true);
    CHECK(strstr(p, "## Tools") != NULL, "tool preamble present");
    CHECK(strstr(p, "DSML") != NULL, "DSML invoke schema in preamble");
    CHECK(strstr(p, "\"name\":\"get_time\"") != NULL, "tool schema injected");
    CHECK(strstr(p, "<tool_result>12:00</tool_result>") != NULL,
          "tool result rendered under a user turn");
    free(p);
    ember_chat_request_free(&req);
}

static void test_reference_system_then_tools_order(void) {
    ember_chat_request req = parse(
        "{\"messages\":["
        "{\"role\":\"system\",\"content\":\"SYSTEM_SENTINEL\"},"
        "{\"role\":\"user\",\"content\":\"q\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":"
        "{\"name\":\"get_time\",\"description\":\"clock\","
        "\"parameters\":{\"type\":\"object\"}}}]}");
    char *p = ember_render_prompt(&req, true, EMBER_THINK_HIGH, true);
    const char *system = strstr(p, "SYSTEM_SENTINEL");
    const char *tools = strstr(p, "## Tools");
    CHECK(system && tools && system < tools,
          "DeepSeek reference order is system prompt then tool schemas");
    free(p);
    ember_chat_request_free(&req);
}

static void test_exact_replay_is_dsml_only(void) {
    ember_chat_request req = parse(
        "{\"messages\":["
        "{\"role\":\"user\",\"content\":\"q\"},"
        "{\"role\":\"assistant\",\"reasoning_content\":\"hidden\","
        "\"content\":\"done\",\"tool_calls\":[{\"id\":\"call_x\","
        "\"type\":\"function\",\"function\":{\"name\":\"get_time\","
        "\"arguments\":\"{}\"}}]},"
        "{\"role\":\"tool\",\"tool_call_id\":\"call_x\",\"content\":\"12:00\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":"
        "{\"name\":\"get_time\",\"parameters\":{\"type\":\"object\"}}}]}");
    req.messages[1].raw_tool_text = strdup(
        "<?DSML?tool_calls>"
        "<?DSML?invoke name=\"get_time\">"
        "</?DSML?invoke>"
        "</?DSML?tool_calls>");
    char *p = ember_render_prompt(&req, false, EMBER_THINK_NONE, true);
    CHECK(strstr(p, "hidden") == NULL,
          "non-thinking prompt canonically drops prior reasoning");
    CHECK(strstr(p, "done\n\n<?DSML?tool_calls>") != NULL,
          "non-thinking prompt retains content and exact DSML block");
    free(p);
    p = ember_render_prompt(&req, true, EMBER_THINK_HIGH, true);
    CHECK(strstr(p, "<think>hidden</think>done\n\n<?DSML?tool_calls>") != NULL,
          "thinking prompt canonically renders reasoning around exact DSML");
    free(p);
    ember_chat_request_free(&req);
}

static void test_tool_continuation_suffix(void) {
    ember_chat_request r = {0};
    ember_chat_msg msgs[2] = {
        {.role = "tool", .content = "one</tool_result>"},
        {.role = "tool", .content = "two"},
    };
    r.messages = msgs;
    r.n_messages = 2;
    char *s = ember_render_tool_continuation_suffix(&r, true);
    CHECK(s != NULL, "tool continuation suffix rendered");
    CHECK(strstr(s, "<｜end▁of▁sentence｜><｜User｜><tool_result>") == s,
          "suffix begins at exact frontier boundary");
    CHECK(strstr(s, "one&lt;/tool_result>") != NULL,
          "suffix escapes result sentinel");
    CHECK(strstr(s, "</tool_result><tool_result>two</tool_result>") != NULL,
          "parallel results share one user turn");
    CHECK(strstr(s, "<｜Assistant｜><think>") != NULL,
          "suffix opens next thinking turn");
    free(s);
}

static void test_invalid_tool_recovery_suffix(void) {
    ember_chat_request r = {0};
    r.thinking_enabled = true;
    r.n_messages = 1;
    r.messages = calloc(1, sizeof(*r.messages));
    r.messages[0].role = strdup("system");
    r.messages[0].content = strdup("Use the declared tools.");
    char *s = ember_render_invalid_tool_recovery_suffix(
        &r, true, "missing invoke name");
    CHECK(s && strstr(s, "</think><" PIPE "end" USCORE "of" USCORE
                     "sentence" PIPE "><" PIPE "User" PIPE ">") == s,
          "malformed retry closes thinking and opens a new user turn");
    CHECK(s && strstr(s, "Tool error: invalid DSML tool call: missing invoke name"),
          "malformed retry explains the parse failure");
    CHECK(s && strstr(s, "System prompt reminder:") == NULL,
          "malformed retry does not duplicate the full system prompt");
    CHECK(s && strstr(s, "</tool_result><" PIPE "Assistant" PIPE "><think>"),
          "malformed retry opens one new thinking assistant turn");
    free(s);
    ember_chat_request_free(&r);
}

// The generated (non-replay) tool-call render must entity-escape all of
// ", &, <, > in invoke/param NAMES, matching ds4_server.c append_dsml_attr_
// escaped. '>' was previously missed, so a name containing it round-tripped as
// a bare '>' where ds4 emits &gt;.
static void test_attr_escape_covers_gt(void) {
    ember_chat_request req = parse(
        "{\"messages\":["
        "{\"role\":\"user\",\"content\":\"q\"},"
        "{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"id\":\"c1\","
        "\"type\":\"function\",\"function\":{\"name\":\"do>it\","
        "\"arguments\":\"{\\\"a>b\\\":\\\"v\\\"}\"}}]}],"
        "\"tools\":[{\"type\":\"function\",\"function\":"
        "{\"name\":\"do>it\",\"parameters\":{\"type\":\"object\"}}}]}");
    // No raw_tool_text -> the generator path (append_dsml_tool_calls) renders.
    char *p = ember_render_prompt(&req, false, EMBER_THINK_NONE, true);
    CHECK(strstr(p, "name=\"do&gt;it\"") != NULL,
          "invoke name escapes '>' to &gt;");
    CHECK(strstr(p, "name=\"a&gt;b\"") != NULL,
          "parameter name escapes '>' to &gt;");
    CHECK(strstr(p, "name=\"do>it\"") == NULL,
          "no bare '>' survives in a rendered name attribute");
    free(p);
    ember_chat_request_free(&req);
}

int main(void) {
    printf("ember chat_template tests\n");
    test_basic_turns();
    test_thinking_opens_think();
    test_tool_preamble_and_result();
    test_reference_system_then_tools_order();
    test_exact_replay_is_dsml_only();
    test_tool_continuation_suffix();
    test_invalid_tool_recovery_suffix();
    test_attr_escape_covers_gt();
    printf("──────────────────────────────\n");
    printf("  %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
