#include "../src/model/chat_template_qwen4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_pass;
static int g_fail;

#define CHECK(cond, msg) do { \
    if (cond) ++g_pass; \
    else { ++g_fail; fprintf(stderr, "FAIL: %s\n", msg); } \
} while (0)

static ember_chat_msg message(const char *role, const char *content) {
    ember_chat_msg msg = {0};
    msg.role = (char *)role;
    msg.content = (char *)content;
    return msg;
}

static void test_default_thinking_exact(void) {
    ember_chat_msg messages[] = {message("user", "  Hello  ")};
    ember_chat_request req = {0};
    req.messages = messages;
    req.n_messages = 1;

    char *prompt = ember_qwen4_render_prompt(&req, true, true);
    const char *expected =
        "<|im_start|>system\n"
        "Reasoning effort is set to xhigh. Please think carefully through the "
        "task, validate key assumptions, consider plausible alternatives, and "
        "prioritize correctness, consistency, and clarity in the final answer."
        "<|im_end|>\n"
        "<|im_start|>user\nHello<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n";
    CHECK(prompt && strcmp(prompt, expected) == 0,
          "default thinking prompt is byte-exact");
    free(prompt);
}

static void test_non_thinking_exact(void) {
    ember_chat_msg messages[] = {
        message("system", "  Be concise.\n"),
        message("user", "Hi"),
    };
    ember_chat_request req = {0};
    req.messages = messages;
    req.n_messages = 2;

    char *prompt = ember_qwen4_render_prompt(&req, false, true);
    const char *expected =
        "<|im_start|>system\nBe concise.<|im_end|>\n"
        "<|im_start|>user\nHi<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    CHECK(prompt && strcmp(prompt, expected) == 0,
          "disabled thinking emits official empty think block");
    free(prompt);
}

static void test_assistant_call_and_grouped_results(void) {
    ember_tool_call call = {
        .name = "weather",
        .arguments = "{\"city\":\"Paris\",\"days\":2}",
    };
    ember_chat_msg messages[] = {
        message("user", "Weather?"),
        message("assistant", ""),
        message("tool", "  sunny  "),
        message("tool", "18 C"),
    };
    messages[1].reasoning = " check forecast ";
    messages[1].calls.calls = &call;
    messages[1].calls.len = 1;
    ember_chat_request req = {0};
    req.messages = messages;
    req.n_messages = 4;
    req.reasoning_effort = "medium";

    char *prompt = ember_qwen4_render_prompt(&req, true, false);
    const char *expected =
        "<|im_start|>user\nWeather?<|im_end|>\n"
        "<|im_start|>assistant\n<think>\ncheck forecast\n</think>\n\n"
        "<tool_call>\n<function=weather>\n"
        "<parameter=city>\nParis\n</parameter>\n"
        "<parameter=days>\n2\n</parameter>\n"
        "</function>\n</tool_call><|im_end|>\n"
        "<|im_start|>user\n<tool_response>\nsunny\n</tool_response>"
        "\n<tool_response>\n18 C\n</tool_response><|im_end|>\n";
    CHECK(prompt && strcmp(prompt, expected) == 0,
          "assistant calls and adjacent results match Qwen XML shape");
    free(prompt);
}

static void test_tool_continuation_exact(void) {
    ember_chat_msg messages[] = {
        message("tool", "one"),
        message("tool", "two"),
    };
    ember_chat_request req = {0};
    req.messages = messages;
    req.n_messages = 2;

    char *suffix = ember_qwen4_render_tool_continuation_suffix(&req, false);
    const char *expected =
        "<|im_end|>\n<|im_start|>user\n<tool_response>\none\n</tool_response>"
        "\n<tool_response>\ntwo\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    CHECK(suffix && strcmp(suffix, expected) == 0,
          "tool continuation closes sampled call and groups results");
    free(suffix);
}

static void test_fail_closed_inputs(void) {
    ember_chat_msg messages[] = {message("developer", "hidden")};
    ember_chat_request req = {0};
    req.messages = messages;
    req.n_messages = 1;
    CHECK(ember_qwen4_render_prompt(&req, true, true) == NULL,
          "unsupported developer role fails closed");

    messages[0] = message("user", "hi");
    req.reasoning_effort = "ultra";
    CHECK(ember_qwen4_render_prompt(&req, true, true) == NULL,
          "unsupported Qwen reasoning effort fails closed");
    req.reasoning_effort = "high";
    CHECK(ember_qwen4_render_prompt(&req, true, true) == NULL,
          "DeepSeek high alias is not accepted as Qwen xhigh");
}

static void test_image_placeholder_order(void) {
    ember_content_part parts[] = {
        {.kind = EMBER_CONTENT_TEXT, .text = "  before "},
        {.kind = EMBER_CONTENT_IMAGE, .text = "https://example.invalid/x.png"},
        {.kind = EMBER_CONTENT_TEXT, .text = " after  "},
    };
    ember_chat_msg messages[] = {message("user", "before  after")};
    messages[0].parts = parts;
    messages[0].n_parts = 3;
    ember_chat_request req = {0};
    req.messages = messages;
    req.n_messages = 1;

    char *prompt = ember_qwen4_render_prompt(&req, false, false);
    const char *expected =
        "<|im_start|>user\nbefore <|vision_start|><|image_pad|>"
        "<|vision_end|> after<|im_end|>\n";
    CHECK(prompt && strcmp(prompt, expected) == 0,
          "image marker remains in exact ordered content position");
    free(prompt);

    messages[0].role = "system";
    CHECK(ember_qwen4_render_prompt(&req, false, false) == NULL,
          "official template rejects images in system messages");
}

int main(void) {
    test_default_thinking_exact();
    test_non_thinking_exact();
    test_assistant_call_and_grouped_results();
    test_tool_continuation_exact();
    test_fail_closed_inputs();
    test_image_placeholder_order();
    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
