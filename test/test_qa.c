// QA parity gauntlet — the GPU-free half of ds4's QA_BEFORE_RELEASES checklist.
// Adversarial/robustness coverage across every layer the server owns, driven at
// pathological token boundaries. If any of these regress, streaming corruption
// or a crash reaches the client. Runtime-only items (quality fixture, degenerate
// decode and concurrent soak) require GPU runtime validation.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/backend/ember_backend.h"
#include "../src/common/buf.h"
#include "../src/common/json.h"
#include "../src/model/chat_template.h"
#include "../src/model/kv_cache.h"
#include "../src/model/model_card.h"
#include "../src/model/tool_parser.h"
#include "../src/server/chat_api.h"
#include "../src/server/http.h"
#include "../src/server/sse.h"

#define PIPE "\xef\xbd\x9c"
static int g_pass = 0, g_fail = 0;
#define CHECK(c, m) do { if (c) g_pass++; else { g_fail++; printf("  FAIL: %s\n", m); } } while (0)

// ── stream a string at EVERY chunk size 1..len; content must always be exact,
//    never contain a marker fragment, never a replacement char ──
static void fuzz_stream(const char *full, const char *expect_content,
                        bool has_tools, bool started_thinking, const char *label) {
    size_t total = strlen(full);
    for (size_t cz = 1; cz <= total; cz++) {
        ember_sse_stream st;
        ember_sse_init(&st, "cc", "m", 1700000000, has_tools, started_thinking, false);
        ember_buf out = {0}, acc = {0};
        for (size_t i = 0; i < total; i += cz) {
            size_t n = cz < total - i ? cz : total - i;
            ember_buf_append(&acc, full + i, n);
            ember_sse_update(&st, acc.ptr, acc.len, false, &out);
        }
        ember_sse_update(&st, acc.ptr, acc.len, true, &out);
        // collect content
        ember_buf c = {0};
        const char *p = out.ptr ? out.ptr : "";
        const char *needle = "\"content\":\"";
        while ((p = strstr(p, needle))) {
            p += strlen(needle);
            while (*p && *p != '"') {
                if (p[0]=='\\' && p[1]=='n'){ember_buf_putc(&c,'\n');p+=2;}
                else if (p[0]=='\\' && p[1]){ember_buf_putc(&c,p[1]);p+=2;}
                else {ember_buf_putc(&c,*p);p++;}
            }
        }
        bool ok = strcmp(c.ptr ? c.ptr : "", expect_content) == 0;
        bool no_repl = out.ptr ? strstr(out.ptr, "\xef\xbf\xbd") == NULL : true;
        if (!ok || !no_repl) {
            char msg[256];
            snprintf(msg, sizeof(msg), "%s @chunk=%zu (got %s)", label, cz,
                     c.ptr ? c.ptr : "");
            CHECK(false, msg);
            ember_buf_free(&c); ember_buf_free(&out); ember_buf_free(&acc); ember_sse_free(&st);
            return;  // one failure per case is enough
        }
        ember_buf_free(&c); ember_buf_free(&out); ember_buf_free(&acc); ember_sse_free(&st);
    }
    g_pass++;  // survived every chunk size
}

static void qa_streaming(void) {
    fuzz_stream("Hello \xf0\x9f\x91\x8b world \xe4\xbd\xa0\xe5\xa5\xbd!",
                "Hello \xf0\x9f\x91\x8b world \xe4\xbd\xa0\xe5\xa5\xbd!",
                false, false, "emoji+CJK all chunk sizes");
    fuzz_stream("text with <not a tag> and a < b arithmetic",
                "text with <not a tag> and a < b arithmetic", false, false,
                "bare < never mistaken for a marker (no tools)");
    fuzz_stream("preamble here<" PIPE "DSML" PIPE "tool_calls>hidden",
                "preamble here", true, false, "DSML never leaks at any split");
}

static void qa_tool_parser(void) {
    ember_tool_calls tc = {0};
    // malformed: unclosed but NAMED invoke → repaired to a zero-arg call (B#4).
    // A named invoke that hits EOS before its closers is a real call the model
    // intended; DwarfStar repairs rather than drops it. (The unnamed case below
    // still yields nothing.)
    ember_parse_dsml_tool_calls("<" PIPE "DSML" PIPE "tool_calls><" PIPE "DSML" PIPE "invoke name=\"x\">", &tc);
    CHECK(tc.len == 1 && tc.calls[0].name && strcmp(tc.calls[0].name, "x") == 0,
          "unclosed named invoke → repaired zero-arg call (B#4)");
    ember_tool_calls_free(&tc);
    // missing name → skipped
    memset(&tc, 0, sizeof(tc));
    ember_parse_dsml_tool_calls("<" PIPE "DSML" PIPE "tool_calls><" PIPE "DSML" PIPE "invoke></" PIPE "DSML" PIPE "invoke></" PIPE "DSML" PIPE "tool_calls>", &tc);
    CHECK(tc.len == 0, "invoke without name → skipped");
    ember_tool_calls_free(&tc);
    // over-trigger: plain prose with the word tool_call must NOT parse
    memset(&tc, 0, sizeof(tc));
    int n = ember_parse_dsml_tool_calls("I could use a tool_call here but won't.", &tc);
    CHECK(n == 0, "prose mentioning tool_call does not trigger");
    ember_tool_calls_free(&tc);
}

static void qa_kv_churn(void) {
    // hammer the cache past capacity; must never exceed cap or corrupt
    ember_kv_cache c;
    ember_kv_init(&c, 8, 1, 2, 3);
    for (int i = 0; i < 200; i++) {
        int32_t ids[4] = {1, i, i + 1, i + 2};
        int s = ember_kv_reserve(&c);
        ember_kv_commit(&c, s, ids, 3);
        CHECK(c.n_entries <= 8, "cache never exceeds capacity under churn");
        if (c.n_entries > 8) break;
    }
    // a recently-committed prefix is still findable
    int32_t last[4] = {1, 199, 200, 201};
    int slot, len;
    ember_kv_lookup(&c, last, 4, &slot, &len);
    CHECK(slot >= 0, "most-recent entry survives churn");
    ember_kv_free(&c);
}

static void qa_http_robustness(void) {
    ember_http_request req;
    // truncated (no header terminator) → parse returns 0, no crash
    char t1[] = "POST /x HTTP/1.1\r\nHost: a";
    CHECK(ember_http_parse(t1, strlen(t1), &req) == 0, "truncated request → 0");
    // body shorter than Content-Length → body_len clamped to available
    char t2[] = "POST /x HTTP/1.1\r\nContent-Length: 100\r\n\r\nshort";
    ember_http_parse(t2, strlen(t2), &req);
    CHECK(req.body_len == 5, "body clamped to available bytes");
    // missing method/path → 0
    char t3[] = "garbage\r\n\r\n";
    CHECK(ember_http_parse(t3, strlen(t3), &req) == 0, "malformed request line → 0");
}

static void qa_json_adversarial(void) {
    CHECK(ember_json_parse("") == NULL, "empty string → NULL");
    // Unlike the NULL-returning cases below, this one parses successfully, so
    // the tree it returns has to be freed — discarding it inline leaked the
    // whole object (found by the CI sanitizer job).
    ember_json *surrogate = ember_json_parse("{\"a\":\"\\ud800\"}");
    CHECK(surrogate != NULL, "lone high surrogate → parses (lenient)");
    ember_json_free(surrogate);
    ember_json *deep = ember_json_parse("[[[[[[[[[[1]]]]]]]]]]");
    CHECK(deep != NULL, "deep nesting parses");
    ember_json_free(deep);
    CHECK(ember_json_parse("{\"a\":1,}") == NULL, "trailing comma → NULL");
    CHECK(ember_json_parse("[1 2 3]") == NULL, "missing commas → NULL");
    ember_json *big = ember_json_parse("{\"n\":123456789012345}");
    CHECK(big && ember_json_num(ember_json_get(big, "n"), 0) > 1e14, "large integer");
    ember_json_free(big);
}

static void qa_template_edges(void) {
    // empty messages array is rejected upstream; here test unusual sequences
    ember_json *v = ember_json_parse(
        "{\"messages\":[{\"role\":\"assistant\",\"content\":\"orphan\"},"
        "{\"role\":\"user\",\"content\":\"\"}]}");
    ember_chat_request req;
    CHECK(ember_chat_request_parse(v, &req), "assistant-first + empty user parses");
    char *p = ember_render_prompt(&req, false, EMBER_THINK_NONE, true);
    CHECK(p && strstr(p, "orphan") != NULL, "orphan assistant rendered, no crash");
    free(p);
    ember_chat_request_free(&req);
    ember_json_free(v);
}

static void qa_model_card_bounds(void) {
    ember_model_card c;
    ember_model_card_load(&c, NULL);
    CHECK(ember_model_card_think_budget(&c, "high", 0) >= 0, "unset max_tokens → non-negative budget");
    CHECK(ember_model_card_think_budget(&c, "bogus-effort", 4096) > 0, "unknown effort → falls back, positive");
    CHECK(ember_model_card_think_budget(&c, "low", 1) == 0, "tiny budget → 0 (no reply starvation)");
    ember_model_card_free(&c);
}

static void qa_validation_timing_contract(void) {
    char *error = NULL;
    ember_backend_config config = {0};
    config.model_path = "stub";
    config.max_ctx = 128;
    config.model_name = "stub-model";
    ember_backend *backend = ember_backend_load(&config, &error);
    CHECK(backend != NULL, "validation timing stub backend loads");
    free(error);
    if (!backend) return;

    const int32_t prompt[] = {1, 2};
    ember_validation_report report;
    CHECK(ember_backend_validate(backend, prompt, 2, 8, &report) && report.ok,
          "validation timing report crosses the backend ABI");
    CHECK(report.baseline_tokens == 8 && report.baseline_decode_s == 0.0 &&
              report.restored_spec_decode_s == 0.0 &&
              report.spec_decode_s == 0.0 && report.prefill_accepted &&
              !report.prefill_tv_checked && report.prefill_tv_index == -1,
          "GPU-free validation never fabricates AR or MTP timing evidence");
    ember_backend_free(backend);
}

int main(void) {
    printf("ember QA gauntlet (GPU-free half of QA_BEFORE_RELEASES)\n");
    qa_streaming();
    qa_tool_parser();
    qa_kv_churn();
    qa_http_robustness();
    qa_json_adversarial();
    qa_template_edges();
    qa_model_card_bounds();
    qa_validation_timing_contract();
    printf("──────────────────────────────\n  %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
