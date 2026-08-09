#include <stdio.h>
#include <stdlib.h>
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
        "\"reasoning_budget_tokens\":2048,\"ember_background\":true,"
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
    CHECK(req.background == true, "background scheduling hint");
    CHECK(req.max_tokens == 4096, "max_tokens");
    CHECK(req.temperature_set && req.temperature == 0, "temperature 0 (set)");
    CHECK(strcmp(req.reasoning_effort, "high") == 0, "reasoning_effort");
    CHECK(req.reasoning_budget_tokens_set &&
          req.reasoning_budget_tokens == 2048,
          "explicit reasoning token budget");
    CHECK(req.n_messages == 3, "3 messages");
    CHECK(strcmp(req.messages[2].role, "tool") == 0 &&
          strcmp(req.messages[2].name, "get_time") == 0, "tool msg name");
    CHECK(ember_chat_request_is_tool_result_continuation(&req),
          "immediate tool result requires target AR decode");
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
    CHECK(!ember_chat_request_is_tool_result_continuation(&req),
          "ordinary message history may use speculative decode");
    ember_chat_request_free(&req);
    ember_json_free(v);
}

static void test_historical_tool_result_does_not_poison_decode(void) {
    const char *body =
        "{\"messages\":["
        "{\"role\":\"user\",\"content\":\"write it\"},"
        "{\"role\":\"assistant\",\"content\":\"\"},"
        "{\"role\":\"tool\",\"name\":\"write_file\",\"content\":\"ok\"},"
        "{\"role\":\"assistant\",\"content\":\"done\"},"
        "{\"role\":\"user\",\"content\":\"continue\"}]}";
    ember_json *v = ember_json_parse(body);
    ember_chat_request req;
    CHECK(ember_chat_request_parse(v, &req), "parse historical tool result");
    CHECK(!ember_chat_request_is_tool_result_continuation(&req),
          "a later user turn is not forced to target AR by old tool history");
    ember_chat_request_free(&req);
    ember_json_free(v);
}

static void test_continuation_only_requires_target_decode(void) {
    const char *body =
        "{\"messages\":[{\"role\":\"tool\",\"tool_call_id\":\"call_1\","
        "\"content\":\"ok\"}]}";
    ember_json *v = ember_json_parse(body);
    ember_chat_request req;
    CHECK(ember_chat_request_parse(v, &req), "parse continuation-only result");
    CHECK(req.continuation_only, "tool-only request is continuation-only");
    CHECK(ember_chat_request_is_tool_result_continuation(&req),
          "continuation-only request requires target AR decode");
    ember_chat_request_free(&req);
    ember_json_free(v);
}

static int loop_rounds_for(const char *body, const char **tool) {
    ember_json *v = ember_json_parse(body);
    ember_chat_request req;
    if (!v || !ember_chat_request_parse(v, &req)) {
        if (v) ember_json_free(v);
        return -1;
    }
    int rounds = ember_chat_request_tool_loop_rounds(&req);
    if (tool) {
        const char *name = ember_chat_request_tool_loop_tool(&req);
        *tool = name ? strdup(name) : NULL;
    }
    ember_chat_request_free(&req);
    ember_json_free(v);
    return rounds;
}

static void test_tool_loop_detector(void) {
    const char *identical =
        "{\"messages\":["
        "{\"role\":\"user\",\"content\":\"inspect\"},"
        "{\"role\":\"assistant\",\"tool_calls\":[{\"id\":\"call_1\","
          "\"type\":\"function\",\"function\":{\"name\":\"terminal\","
          "\"arguments\":\"{\\\"command\\\": \\\"pwd\\\"}\"}}]},"
        "{\"role\":\"tool\",\"tool_call_id\":\"call_1\",\"content\":\"/tmp\\n\"},"
        "{\"role\":\"assistant\",\"tool_calls\":[{\"id\":\"call_2\","
          "\"type\":\"function\",\"function\":{\"name\":\"terminal\","
          "\"arguments\":\"{\\\"command\\\":\\\"pwd\\\"}\"}}]},"
        "{\"role\":\"tool\",\"tool_call_id\":\"call_2\",\"content\":\"/tmp\\n\"},"
        "{\"role\":\"assistant\",\"tool_calls\":[{\"id\":\"call_3\","
          "\"type\":\"function\",\"function\":{\"name\":\"terminal\","
          "\"arguments\":\"{ \\\"command\\\" : \\\"pwd\\\" }\"}}]},"
        "{\"role\":\"tool\",\"tool_call_id\":\"call_3\",\"content\":\"/tmp\\n\"},"
        "{\"role\":\"assistant\",\"tool_calls\":[{\"id\":\"call_4\","
          "\"type\":\"function\",\"function\":{\"name\":\"terminal\","
          "\"arguments\":\"{\\\"command\\\":\\\"pwd\\\"}\"}}]},"
        "{\"role\":\"tool\",\"tool_call_id\":\"call_4\",\"content\":\"/tmp\\n\"}]}";
    const char *tool = NULL;
    CHECK(loop_rounds_for(identical, &tool) == 4,
          "distinct ids with identical normalized calls and results count");
    CHECK(tool && !strcmp(tool, "terminal"),
          "detector labels the newest round's first tool");
    free((void *)tool);

    const char *different_result =
        "{\"messages\":["
        "{\"role\":\"assistant\",\"tool_calls\":[{\"id\":\"a\","
          "\"function\":{\"name\":\"poll\",\"arguments\":\"{}\"}}]},"
        "{\"role\":\"tool\",\"tool_call_id\":\"a\",\"content\":\"pending\"},"
        "{\"role\":\"assistant\",\"tool_calls\":[{\"id\":\"b\","
          "\"function\":{\"name\":\"poll\",\"arguments\":\"{}\"}}]},"
        "{\"role\":\"tool\",\"tool_call_id\":\"b\",\"content\":\"done\"}]}";
    CHECK(loop_rounds_for(different_result, NULL) == 0,
          "same call with changing result is legitimate polling");

    const char *different_call =
        "{\"messages\":["
        "{\"role\":\"assistant\",\"tool_calls\":[{\"id\":\"a\","
          "\"function\":{\"name\":\"ls\",\"arguments\":\"{\\\"path\\\":\\\"one\\\"}\"}}]},"
        "{\"role\":\"tool\",\"tool_call_id\":\"a\",\"content\":\"\"},"
        "{\"role\":\"assistant\",\"tool_calls\":[{\"id\":\"b\","
          "\"function\":{\"name\":\"ls\",\"arguments\":\"{\\\"path\\\":\\\"two\\\"}\"}}]},"
        "{\"role\":\"tool\",\"tool_call_id\":\"b\",\"content\":\"\"}]}";
    CHECK(loop_rounds_for(different_call, NULL) == 0,
          "different calls with identical results are legitimate exploration");

    const char *parallel =
        "{\"messages\":["
        "{\"role\":\"assistant\",\"tool_calls\":["
          "{\"id\":\"a1\",\"function\":{\"name\":\"one\",\"arguments\":\"{}\"}},"
          "{\"id\":\"a2\",\"function\":{\"name\":\"two\",\"arguments\":\"{}\"}}]},"
        "{\"role\":\"tool\",\"tool_call_id\":\"a2\",\"content\":\"R2\"},"
        "{\"role\":\"tool\",\"tool_call_id\":\"a1\",\"content\":\"R1\"},"
        "{\"role\":\"assistant\",\"tool_calls\":["
          "{\"id\":\"b1\",\"function\":{\"name\":\"one\",\"arguments\":\"{}\"}},"
          "{\"id\":\"b2\",\"function\":{\"name\":\"two\",\"arguments\":\"{}\"}}]},"
        "{\"role\":\"tool\",\"tool_call_id\":\"b1\",\"content\":\"R1\"},"
        "{\"role\":\"tool\",\"tool_call_id\":\"b2\",\"content\":\"R2\"}]}";
    CHECK(loop_rounds_for(parallel, NULL) == 2,
          "parallel calls compare in call order and bind results by id");

    const char *parallel_reordered =
        "{\"messages\":["
        "{\"role\":\"assistant\",\"tool_calls\":["
          "{\"id\":\"a1\",\"function\":{\"name\":\"one\",\"arguments\":\"{}\"}},"
          "{\"id\":\"a2\",\"function\":{\"name\":\"two\",\"arguments\":\"{}\"}}]},"
        "{\"role\":\"tool\",\"tool_call_id\":\"a1\",\"content\":\"R1\"},"
        "{\"role\":\"tool\",\"tool_call_id\":\"a2\",\"content\":\"R2\"},"
        "{\"role\":\"assistant\",\"tool_calls\":["
          "{\"id\":\"b2\",\"function\":{\"name\":\"two\",\"arguments\":\"{}\"}},"
          "{\"id\":\"b1\",\"function\":{\"name\":\"one\",\"arguments\":\"{}\"}}]},"
        "{\"role\":\"tool\",\"tool_call_id\":\"b2\",\"content\":\"R2\"},"
        "{\"role\":\"tool\",\"tool_call_id\":\"b1\",\"content\":\"R1\"}]}";
    CHECK(loop_rounds_for(parallel_reordered, NULL) == 0,
          "parallel call-set order is significant");
}

static int loop_calls_for(const char *body, const char **tool) {
    ember_json *v = ember_json_parse(body);
    ember_chat_request req;
    if (!v || !ember_chat_request_parse(v, &req)) {
        if (v) ember_json_free(v);
        return -1;
    }
    int calls = ember_chat_request_tool_loop_calls(&req);
    if (tool) {
        const char *name = ember_chat_request_tool_loop_tool(&req);
        *tool = name ? strdup(name) : NULL;
    }
    ember_chat_request_free(&req);
    ember_json_free(v);
    return calls;
}

// The repeated call and the message SHAPE below are the production loop of
// 2026-08-08, captured as /tmp/capture/live-{20,23,24,25,26}-req.json on the
// a production capture: the model re-emitted this byte-identical web_search call on nine
// consecutive tool-calling turns. Result CONTENTS are representative rather
// than verbatim -- the real ones ran to 26,301 bytes and two of them shared a
// 26,107-character prefix, so committing them would add ~100 KB of search
// output whose text is irrelevant. What is reproduced exactly is the property
// under test: which results are byte-identical to each other and which are not,
// and where the two user turns fall. Measured against the real captures, the
// strict signal reported 0/3/0/0/0 and this one reports 5/6/7/8/9.
// Leading commas, and every body opens with HEAD's user turn: a trailing comma
// before "]}" is invalid JSON and makes ember_chat_request_parse fail, which
// surfaces as -1 from the helpers rather than as a wrong count.
#define HEAD "{\"messages\":[{\"role\":\"user\",\"content\":\"research it\"}"
#define TAIL "]}"
#define RPT_CALL(id)                                                          \
    ",{\"role\":\"assistant\",\"tool_calls\":[{\"id\":\"" id "\","            \
      "\"type\":\"function\",\"function\":{\"name\":\"web_search\","          \
      "\"arguments\":\"{\\\"query\\\":\\\"DeepSeek V4 Flash tool calling "    \
      "weakness why prose strong agentic weak\\\"}\"}}]}"
#define RES(id, body)                                                         \
    ",{\"role\":\"tool\",\"tool_call_id\":\"" id "\",\"content\":\"" body "\"}"

static void test_tool_loop_call_signature_detector(void) {
    // Exactly the captured shape: three repeats, assistant prose, a user turn,
    // four more repeats ending in the guardrail blocker, prose, a second user
    // turn, then two more against results that differ only at their tail.
    const char *production_loop =
        HEAD
        // A DIFFERENT, earlier query -- the walk must stop here, not run away.
        ",{\"role\":\"assistant\",\"tool_calls\":["
          "{\"id\":\"c2a\",\"function\":{\"name\":\"web_search\","
            "\"arguments\":\"{\\\"query\\\":\\\"model tool calling\\\"}\"}},"
          "{\"id\":\"c2b\",\"function\":{\"name\":\"web_search\","
            "\"arguments\":\"{\\\"query\\\":\\\"why prose strong\\\"}\"}}]}"
        RES("c2a", "[web_search] query='model tool calling' (28,346 chars result)")
        RES("c2b", "[web_search] query='why prose strong' (12,004 chars result)")
        RPT_CALL("c5")  RES("c5", "[web_search] (28,346 chars result)")
        RPT_CALL("c7")  RES("c7", "[web_search] (28,346 chars result)")
        RPT_CALL("c9")  RES("c9", "[web_search] (923 chars result)")
        ",{\"role\":\"assistant\",\"content\":\"I stopped retrying web_search.\"}"
        ",{\"role\":\"user\",\"content\":\"continue\"}"
        RPT_CALL("c13") RES("c13", "[web_search] (28,346 chars result)")
        RPT_CALL("c15") RES("c15", "[web_search] (28,346 chars result)")
        RPT_CALL("c17") RES("c17", "[web_search] (28,346 chars result)")
        RPT_CALL("c19") RES("c19", "{\\\"error\\\": \\\"Blocked web_search: "
                                   "stop repeating it unchanged\\\"}")
        ",{\"role\":\"assistant\",\"content\":\"The last result explains it.\"}"
        ",{\"role\":\"user\",\"content\":\"continue\"}"
        RPT_CALL("c23") RES("c23", "results A ... differing tail alpha")
        RPT_CALL("c25") RES("c25", "results A ... differing tail beta")
        TAIL;

    const char *tool = NULL;
    CHECK(loop_calls_for(production_loop, &tool) == 9,
          "call-signature signal counts the nine repeated production calls");
    CHECK(tool && !strcmp(tool, "web_search"),
          "call-signature signal is labelled with the repeated tool");
    free((void *)tool);

    // The regression guard. The strict signal must stay exactly as blind as it
    // was: this is the history it could not see, and widening it was NOT the
    // fix. Its final round pair has differing results, so it reports nothing.
    CHECK(loop_rounds_for(production_loop, NULL) == 0,
          "strict round signal is unchanged by the call-signature signal");

    // Each suppressor in isolation, so a regression names its own cause.
    const char *across_user_turn =
        HEAD
        RPT_CALL("a") RES("a", "R")
        ",{\"role\":\"user\",\"content\":\"continue\"}"
        RPT_CALL("b") RES("b", "R")
        TAIL;
    CHECK(loop_calls_for(across_user_turn, NULL) == 2,
          "a user turn between repeats does not stop the call-signature walk");
    CHECK(loop_rounds_for(across_user_turn, NULL) == 0,
          "the strict signal still refuses to match across a user turn");

    const char *changing_results =
        HEAD
        RPT_CALL("a") RES("a", "pending")
        RPT_CALL("b") RES("b", "still pending")
        RPT_CALL("c") RES("c", "done")
        TAIL;
    CHECK(loop_calls_for(changing_results, NULL) == 3,
          "differing results do not stop the call-signature walk");
    CHECK(loop_rounds_for(changing_results, NULL) == 0,
          "the strict signal still treats changing results as legitimate polling");

    // Must NOT fire: these are ordinary agent behaviour.
    const char *different_args =
        HEAD
        ",{\"role\":\"assistant\",\"tool_calls\":[{\"id\":\"a\","
          "\"function\":{\"name\":\"ls\",\"arguments\":\"{\\\"p\\\":\\\"one\\\"}\"}}]}"
        RES("a", "")
        ",{\"role\":\"assistant\",\"tool_calls\":[{\"id\":\"b\","
          "\"function\":{\"name\":\"ls\",\"arguments\":\"{\\\"p\\\":\\\"two\\\"}\"}}]}"
        RES("b", "")
        TAIL;
    CHECK(loop_calls_for(different_args, NULL) == 0,
          "different arguments are exploration, not a loop");

    const char *single = HEAD RPT_CALL("a") RES("a", "R") TAIL;
    CHECK(loop_calls_for(single, NULL) == 0, "a lone call is not a loop");

    // Whitespace-only argument differences are the same call (shared
    // normalization with the strict signal).
    const char *renormalized =
        HEAD
        ",{\"role\":\"assistant\",\"tool_calls\":[{\"id\":\"a\","
          "\"function\":{\"name\":\"t\",\"arguments\":\"{\\\"c\\\":\\\"pwd\\\"}\"}}]}"
        RES("a", "R")
        ",{\"role\":\"assistant\",\"tool_calls\":[{\"id\":\"b\","
          "\"function\":{\"name\":\"t\",\"arguments\":\"{ \\\"c\\\" : \\\"pwd\\\" }\"}}]}"
        RES("b", "R")
        TAIL;
    CHECK(loop_calls_for(renormalized, NULL) == 2,
          "call-signature comparison normalizes argument whitespace");

    const char *continuation =
        "{\"messages\":[{\"role\":\"tool\",\"tool_call_id\":\"a\","
        "\"content\":\"ok\"}]}";
    CHECK(loop_calls_for(continuation, NULL) == 0,
          "continuation-only requests report no call-signature loop");
}

#undef HEAD
#undef TAIL
#undef RPT_CALL
#undef RES

static void test_reject_no_messages(void) {
    ember_json *v = ember_json_parse("{\"model\":\"x\"}");
    ember_chat_request req;
    CHECK(!ember_chat_request_parse(v, &req), "reject request with no messages");
    ember_json_free(v);
}

static void test_reject_invalid_sampler_ranges(void) {
    const char *cases[] = {
        "{\"messages\":[],\"temperature\":-1}",
        "{\"messages\":[],\"top_p\":1.1}",
        "{\"messages\":[],\"min_p\":-0.1}",
        "{\"messages\":[],\"top_k\":-1}",
        "{\"messages\":[],\"repetition_penalty\":0}",
        "{\"messages\":[],\"rep_window\":-1}",
        "{\"messages\":[],\"frequency_penalty\":2.1}",
        "{\"messages\":[],\"presence_penalty\":-2.1}",
        "{\"messages\":[],\"frequency_penalty\":\"zero\"}",
        "{\"messages\":[],\"presence_penalty\":null}",
        "{\"messages\":[],\"max_tokens\":-1}",
        "{\"messages\":[],\"reasoning_budget_tokens\":-1}",
        "{\"messages\":[],\"thinking_token_budget\":\"many\"}",
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        ember_json *v = ember_json_parse(cases[i]);
        ember_chat_request req;
        CHECK(v && !ember_chat_request_parse(v, &req),
              "reject invalid sampler range");
        ember_json_free(v);
    }
}

static void test_explicit_zero_penalty_overrides(void) {
    ember_json *v = ember_json_parse(
        "{\"messages\":[],\"frequency_penalty\":0,"
        "\"presence_penalty\":0}");
    ember_chat_request req;
    CHECK(ember_chat_request_parse(v, &req), "explicit zero penalties parse");
    CHECK(req.freq_pen_set && req.freq_pen == 0.0 &&
          req.pres_pen_set && req.pres_pen == 0.0,
          "explicit zero penalties retain supplied-state");
    ember_chat_request_free(&req);
    ember_json_free(v);

    v = ember_json_parse("{\"messages\":[]}");
    CHECK(ember_chat_request_parse(v, &req), "omitted penalties parse");
    CHECK(!req.freq_pen_set && !req.pres_pen_set,
          "omitted penalties remain distinguishable from zero overrides");
    ember_chat_request_free(&req);
    ember_json_free(v);
}

static void test_reasoning_budget_alias(void) {
    ember_json *v = ember_json_parse(
        "{\"messages\":[],\"thinking_token_budget\":777}");
    ember_chat_request req;
    CHECK(ember_chat_request_parse(v, &req),
          "parse vLLM thinking budget alias");
    CHECK(req.reasoning_budget_tokens_set &&
          req.reasoning_budget_tokens == 777,
          "thinking_token_budget normalizes to reasoning budget");
    ember_chat_request_free(&req);
    ember_json_free(v);
}

static void test_tool_choice_constraints(void) {
    const char *tools =
        "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"a\"}},"
        "{\"type\":\"function\",\"function\":{\"name\":\"b\"}}]";
    char body[1024];
    ember_chat_request req;

    snprintf(body, sizeof(body),
             "{\"messages\":[],%s,\"tool_choice\":\"required\"}", tools);
    ember_json *v = ember_json_parse(body);
    CHECK(v && ember_chat_request_parse(v, &req), "required tool choice parses");
    CHECK(req.tool_choice == EMBER_TOOL_CHOICE_REQUIRED &&
          req.tool_choice_required, "required tool choice is enforced");
    ember_chat_request_free(&req); ember_json_free(v);

    snprintf(body, sizeof(body),
             "{\"messages\":[],%s,\"parallel_tool_calls\":false,"
             "\"tool_choice\":{\"type\":\"function\","
             "\"function\":{\"name\":\"b\"}}}", tools);
    v = ember_json_parse(body);
    CHECK(v && ember_chat_request_parse(v, &req), "named tool choice parses");
    CHECK(req.tool_choice == EMBER_TOOL_CHOICE_NAMED &&
          req.n_tool_choice_names == 1 &&
          !strcmp(req.tool_choice_names[0], "b") &&
          !req.parallel_tool_calls,
          "named and no-parallel constraints are retained");
    ember_chat_request_free(&req); ember_json_free(v);

    snprintf(body, sizeof(body),
             "{\"messages\":[],%s,\"tool_choice\":{"
             "\"type\":\"allowed_tools\",\"mode\":\"auto\","
             "\"tools\":[{\"type\":\"function\",\"name\":\"a\"}]}}",
             tools);
    v = ember_json_parse(body);
    CHECK(v && ember_chat_request_parse(v, &req), "allowed-tools choice parses");
    CHECK(req.tool_choice == EMBER_TOOL_CHOICE_ALLOWED &&
          !req.tool_choice_required && req.n_tool_choice_names == 1,
          "allowed-tools auto constraint is retained");
    ember_chat_request_free(&req); ember_json_free(v);

    snprintf(body, sizeof(body),
             "{\"messages\":[],%s,\"tool_choice\":{\"type\":\"function\","
             "\"function\":{\"name\":\"missing\"}}}", tools);
    v = ember_json_parse(body);
    CHECK(v && !ember_chat_request_parse(v, &req),
          "forced unadvertised tool is rejected");
    ember_json_free(v);
}

int main(void) {
    printf("ember chat_api tests\n");
    test_full_request();
    test_multimodal_content();
    test_historical_tool_result_does_not_poison_decode();
    test_continuation_only_requires_target_decode();
    test_tool_loop_detector();
    test_tool_loop_call_signature_detector();
    test_reject_no_messages();
    test_reject_invalid_sampler_ranges();
    test_explicit_zero_penalty_overrides();
    test_reasoning_budget_alias();
    test_tool_choice_constraints();
    printf("──────────────────────────────\n");
    printf("  %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
