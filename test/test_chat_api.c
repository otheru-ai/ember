#include <stdio.h>
#include <string.h>

#include "../src/common/json.h"
#include "../src/server/chat_api.h"

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, msg)                                                    \
    do { if (cond) g_pass++; else { g_fail++; printf("  FAIL: %s\n", msg); } } while (0)

static void test_full_request(void) {
    const char *body =
        "{\"model\":\"deepseek-v4-flash\",\"stream\":true,\"max_tokens\":4096,"
        "\"temperature\":0,\"reasoning_effort\":\"high\","
        "\"messages\":["
        "{\"role\":\"system\",\"content\":\"be helpful\"},"
        "{\"role\":\"user\",\"content\":\"hi\"},"
        "{\"role\":\"tool\",\"name\":\"get_time\",\"content\":\"12:00\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_time\"}}]}";
    ember_json *v = ember_json_parse(body);
    ember_chat_request req;
    bool ok = ember_chat_request_parse(v, &req);
    CHECK(ok, "parse ok");
    CHECK(strcmp(req.model, "deepseek-v4-flash") == 0, "model");
    CHECK(req.stream == true, "stream");
    CHECK(req.max_tokens == 4096, "max_tokens");
    CHECK(req.temperature_set && req.temperature == 0, "temperature 0 (set)");
    CHECK(strcmp(req.reasoning_effort, "high") == 0, "reasoning_effort");
    CHECK(req.n_messages == 3, "3 messages");
    CHECK(strcmp(req.messages[2].role, "tool") == 0 &&
          strcmp(req.messages[2].name, "get_time") == 0, "tool msg name");
    CHECK(req.has_tools && strstr(req.tools_json, "get_time") != NULL,
          "tools re-serialized");
    ember_chat_request_free(&req);
    ember_json_free(v);
}

static void test_multimodal_content(void) {
    const char *body =
        "{\"messages\":[{\"role\":\"user\",\"content\":["
        "{\"type\":\"text\",\"text\":\"look: \"},"
        "{\"type\":\"image_url\",\"image_url\":{\"url\":\"x\"}},"
        "{\"type\":\"text\",\"text\":\"a cat\"}]}]}";
    ember_json *v = ember_json_parse(body);
    ember_chat_request req;
    CHECK(ember_chat_request_parse(v, &req), "parse multimodal");
    CHECK(strcmp(req.messages[0].content, "look: a cat") == 0,
          "text parts flattened, non-text dropped");
    CHECK(!req.has_tools, "no tools");
    ember_chat_request_free(&req);
    ember_json_free(v);
}

static void test_reject_no_messages(void) {
    ember_json *v = ember_json_parse("{\"model\":\"x\"}");
    ember_chat_request req;
    CHECK(!ember_chat_request_parse(v, &req), "reject request with no messages");
    ember_json_free(v);
}

int main(void) {
    printf("ember chat_api tests\n");
    test_full_request();
    test_multimodal_content();
    test_reject_no_messages();
    printf("──────────────────────────────\n");
    printf("  %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
