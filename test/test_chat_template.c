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
    static ember_chat_request req;  // leaked in test; fine
    ember_chat_request_parse(v, &req);
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
}

static void test_thinking_opens_think(void) {
    ember_chat_request req = parse(
        "{\"messages\":[{\"role\":\"user\",\"content\":\"q\"}]}");
    char *p = ember_render_prompt(&req, /*enable_thinking=*/true, EMBER_THINK_HIGH, true);
    CHECK(ember_prompt_ends_in_open_think(p), "thinking prompt ends in open <think>");
    char *p2 = ember_render_prompt(&req, false, EMBER_THINK_NONE, true);
    CHECK(!ember_prompt_ends_in_open_think(p2), "non-thinking does not");
    free(p); free(p2);
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
}

int main(void) {
    printf("ember chat_template tests\n");
    test_basic_turns();
    test_thinking_opens_think();
    test_tool_preamble_and_result();
    printf("──────────────────────────────\n");
    printf("  %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
