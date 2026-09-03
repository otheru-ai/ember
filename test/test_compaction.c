// Context compaction tests (src/server/compaction.c), ported policy from ds4.
//
// The policy helpers are pure, so they are checked against the exact thresholds
// ds4 uses. The full exchange runs against backend_stub, which implements the
// same ABI as the real backend — so the rebuild, the ownership transfer, and the
// bounded-output invariant are all exercised without a GPU.
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/server/compaction.h"
#include "../src/model/chat_template.h"

static int checks = 0;
#define CHECK(cond, what) do {                                   \
    checks++;                                                    \
    if (!(cond)) { printf("FAIL: %s\n", (what)); exit(1); }      \
} while (0)

// ── policy ──────────────────────────────────────────────────────────────────

static void test_needed(void) {
    const int ctx = 131072;
    // 85% of 131072 = 111411.
    CHECK(!ember_compaction_needed(111410, ctx), "just under 85% does not compact");
    CHECK(ember_compaction_needed(111411, ctx), "at 85% compacts");
    CHECK(ember_compaction_needed(130000, ctx), "near-full compacts");

    // Second trigger: free <= min(8192, ctx/8). For ctx=131072, ctx/8=16384, so
    // the 8192 floor binds and 85% fires first — matching ds4.
    CHECK(ember_compaction_needed(ctx - 8192, ctx), "free==8192 compacts");

    // Small contexts: the proportional cap (ctx/8) must bind so tiny contexts
    // still compact rather than fail (ds4_agent.c:8020-8021).
    const int tiny = 1024;                  // 85% = 870, ctx/8 = 128
    CHECK(!ember_compaction_needed(800, tiny), "tiny: 800/1024 does not compact");
    CHECK(ember_compaction_needed(870, tiny), "tiny: at 85% compacts");
    CHECK(ember_compaction_needed(896, tiny), "tiny: free==128 compacts");

    // Degenerate inputs must never claim compaction is needed.
    CHECK(!ember_compaction_needed(0, ctx), "zero prompt");
    CHECK(!ember_compaction_needed(100, 0), "zero ctx");
    CHECK(!ember_compaction_needed(-5, ctx), "negative prompt");
}

static void test_tail_budget(void) {
    CHECK(ember_compaction_tail_budget(131072) == 13107, "ctx/10 at 131072");
    // The 50k cap only binds above ctx == 500k.
    CHECK(ember_compaction_tail_budget(1000000) == EMBER_COMPACT_TAIL_CAP_TOKENS,
          "cap binds on huge ctx");
    CHECK(ember_compaction_tail_budget(5) == 1, "never returns 0");
    CHECK(ember_compaction_tail_budget(0) == 1, "degenerate ctx");
}

// ── DSML sigil guard ────────────────────────────────────────────────────────

static void test_sigil(void) {
    CHECK(!ember_compaction_text_has_sigil("a normal summary"), "clean text");
    CHECK(!ember_compaction_text_has_sigil(""), "empty text");
    CHECK(ember_compaction_text_has_sigil("text <?DSML?tool_calls> more"),
          "ascii spelling detected");

    // A trailing PARTIAL opener must be trimmed. This is the guard the DSML
    // content leak needs: a half-emitted sigil must never survive into content.
    char partial[64];
    snprintf(partial, sizeof(partial), "summary text <?DSML?tool_c");
    ember_compaction_trim_partial_sigil(partial);
    CHECK(strcmp(partial, "summary text ") == 0, "partial sigil trimmed");

    // ds4's exact case: a lone trailing '<'.
    char lone[32];
    snprintf(lone, sizeof(lone), "summary<");
    ember_compaction_trim_partial_sigil(lone);
    CHECK(strcmp(lone, "summary") == 0, "lone '<' trimmed");

    // A complete, non-sigil text must be left alone.
    char clean[32];
    snprintf(clean, sizeof(clean), "done.");
    ember_compaction_trim_partial_sigil(clean);
    CHECK(strcmp(clean, "done.") == 0, "clean text untouched");

    // '<' mid-string is not a trailing partial and must survive.
    char mid[32];
    snprintf(mid, sizeof(mid), "a < b");
    ember_compaction_trim_partial_sigil(mid);
    CHECK(strcmp(mid, "a < b") == 0, "interior '<' kept");
}

static void test_prompt_and_wrap(void) {
    char *p = ember_compaction_make_prompt("soft limit");
    CHECK(p != NULL, "prompt allocated");
    CHECK(strstr(p, "durable task-state summary") != NULL, "asks for durable state");
    CHECK(strstr(p, "do not call tools") != NULL, "forbids tools");
    CHECK(strstr(p, "Compaction reason: soft limit") != NULL, "carries the reason");
    free(p);

    char *w = ember_compaction_wrap_summary("STATE");
    CHECK(w != NULL, "wrap allocated");
    CHECK(strstr(w, EMBER_COMPACT_SUMMARY_OPEN) != NULL, "open marker");
    CHECK(strstr(w, EMBER_COMPACT_SUMMARY_CLOSE) != NULL, "close marker");
    CHECK(strstr(w, "STATE") != NULL, "summary body present");
    free(w);
}

// ── message-structure helpers ───────────────────────────────────────────────

static ember_chat_msg mk(const char *role, const char *content) {
    ember_chat_msg m = {0};
    m.role = strdup(role);
    m.content = strdup(content);
    return m;
}

// Build a request: 1 system + `turns` * (user, assistant).
static void build_req(ember_chat_request *r, int turns) {
    memset(r, 0, sizeof(*r));
    r->model = strdup("test-model");
    r->n_messages = 1 + turns * 2;
    r->messages = (ember_chat_msg *)calloc((size_t)r->n_messages,
                                           sizeof(ember_chat_msg));
    r->messages[0] = mk("system", "You are a test harness.");
    for (int i = 0; i < turns; i++) {
        char buf[128];
        snprintf(buf, sizeof(buf), "user turn %d with some filler text to spend tokens", i);
        r->messages[1 + i * 2] = mk("user", buf);
        snprintf(buf, sizeof(buf), "assistant reply %d with some filler text as well", i);
        r->messages[2 + i * 2] = mk("assistant", buf);
    }
    r->thinking_enabled = true;
    r->think_mode = EMBER_THINK_HIGH;
}

static void test_head_and_boundaries(void) {
    ember_chat_request r;
    build_req(&r, 4);
    CHECK(ember_compaction_head_count(&r) == 1, "one leading system message");

    // A second leading system/developer message is also head.
    ember_chat_request r2;
    build_req(&r2, 2);
    free(r2.messages[1].role);
    r2.messages[1].role = strdup("developer");
    CHECK(ember_compaction_head_count(&r2) == 2, "developer counts as head");
    ember_chat_request_free(&r2);

    CHECK(ember_compaction_next_user_msg(&r, 1, 1) == 1, "first user at 1");
    CHECK(ember_compaction_next_user_msg(&r, 1, 2) == 3, "next user after 2 is 3");
    CHECK(ember_compaction_next_user_msg(&r, 1, 100) == -1, "past end returns -1");
    ember_chat_request_free(&r);
}

// A message that arrived as a content array owns a `parts` allocation as well
// as its flattened `content`. Compaction drops middle messages, and dropping
// one must release the parts too -- a leak here is invisible in the report and
// scales with conversation length. ASan in CI is what actually catches it; this
// test exists to put a parts-bearing message in the dropped range at all.
static void test_dropped_message_with_content_parts(void) {
    setenv("EMBER_STUB_REPLY",
           "DURABLE STATE: the parts-bearing turn was dropped cleanly.", 1);
    char *err = NULL;
    ember_backend_config cfg = {0};
    cfg.model_path = "stub";
    cfg.max_ctx = 2048;
    cfg.model_name = "stub-model";
    ember_backend *be = ember_backend_load(&cfg, &err);
    CHECK(be != NULL, "stub backend loads");
    free(err);

    ember_chat_request r;
    build_req(&r, 24);
    // Message 3 is a middle user turn, comfortably inside the dropped range.
    ember_chat_msg *m = &r.messages[3];
    m->n_parts = 2;
    m->parts = (ember_content_part *)calloc(2, sizeof(ember_content_part));
    m->parts[0].kind = EMBER_CONTENT_TEXT;
    m->parts[0].text = strdup("the original text part");
    m->parts[1].kind = EMBER_CONTENT_TEXT;
    m->parts[1].text = strdup("a second part with a detail hint");
    m->parts[1].detail = strdup("auto");

    char *prompt = ember_render_prompt(&r, true, EMBER_THINK_HIGH, true);
    int32_t *ids = NULL;
    int n_prompt = ember_backend_encode(be, prompt, &ids);
    free(prompt); free(ids);
    CHECK(n_prompt > 0, "the parts-bearing fixture encodes");

    ember_compaction_report rep = {0};
    bool ok = ember_compact_request(be, &r, ember_backend_n_ctx(be), n_prompt,
                                    -1, "unit test", &rep);
    CHECK(ok && rep.applied, "compaction applies with a parts-bearing message");
    CHECK(rep.dropped_messages > 0, "the parts-bearing turn was in the drop range");
    ember_chat_request_free(&r);
    ember_backend_free(be);
    unsetenv("EMBER_STUB_REPLY");
}

// ── full exchange against the stub backend ──────────────────────────────────

static void test_exchange(void) {
    // The stub tokenizes one token per BYTE, so sizing matters: ctx 2048 gives a
    // 204-token tail budget (ctx/10), which fits a turn or two, and 24 turns of
    // ~60-byte messages puts the prompt well past the 85% trigger. A canned
    // summary makes the whole exchange deterministic.
    setenv("EMBER_STUB_REPLY",
           "DURABLE STATE: goal is to port ds4 compaction; compaction.c added; "
           "tail budget is ctx/10; next step is wiring usage reporting.", 1);

    char *err = NULL;
    ember_backend_config cfg = {0};
    cfg.model_path = "stub";
    cfg.max_ctx = 2048;
    cfg.model_name = "stub-model";
    ember_backend *be = ember_backend_load(&cfg, &err);
    CHECK(be != NULL, "stub backend loads");
    free(err);

    const int n_ctx = ember_backend_n_ctx(be);

    ember_chat_request r;
    build_req(&r, 24);
    const int before_msgs = r.n_messages;

    char *prompt = ember_render_prompt(&r, true, EMBER_THINK_HIGH, true);
    CHECK(prompt != NULL, "render");
    int32_t *ids = NULL;
    int n_prompt = ember_backend_encode(be, prompt, &ids);
    free(prompt);
    CHECK(n_prompt > 0, "encode");

    CHECK(ember_compaction_needed(n_prompt, n_ctx),
          "the sized fixture actually trips the policy");

    ember_compaction_report rep = {0};
    bool ok = ember_compact_request(be, &r, n_ctx, n_prompt, -1, "unit test", &rep);
    if (!ok)
        printf("  refused: %s (n_prompt=%d ctx=%d tail_budget=%d)\n",
               rep.error, n_prompt, n_ctx, ember_compaction_tail_budget(n_ctx));
    CHECK(ok, "compaction applies on a fixture built to trip it");

    {
        CHECK(rep.applied, "report says applied");
        CHECK(strstr(r.messages[rep.head_messages].content,
                     "DURABLE STATE") != NULL,
              "the model's summary text landed in the rebuilt history");
        // The invariant that matters: the rebuild is bounded by construction, so
        // it MUST be smaller than what it replaced. ds4's design guarantees this;
        // a compaction that can grow is the bug this port exists to remove.
        CHECK(rep.compacted_tokens < rep.original_tokens,
              "compaction strictly shrinks the prompt");
        CHECK(rep.compacted_tokens < n_ctx, "result fits context");
        CHECK(rep.dropped_messages > 0, "some messages were replaced");
        CHECK(r.n_messages < before_msgs, "message count shrank");
        CHECK(r.n_messages == rep.head_messages + 1 +
                              (before_msgs - rep.tail_start_msg),
              "rebuilt as head + summary + tail");
        // Head preserved, summary injected right after it.
        CHECK(strcmp(r.messages[0].role, "system") == 0, "head still first");
        CHECK(strcmp(r.messages[rep.head_messages].role, "system") == 0,
              "summary is a system message");
        CHECK(strstr(r.messages[rep.head_messages].content,
                     EMBER_COMPACT_SUMMARY_OPEN) != NULL,
              "summary carries the open marker");
        // The tail must open on a user turn (ds4 snaps to <｜User｜>).
        CHECK(strcmp(r.messages[rep.head_messages + 1].role, "user") == 0,
              "tail opens on a user turn");
        // Re-render must still tokenize.
        char *p2 = ember_render_prompt(&r, true, EMBER_THINK_HIGH, true);
        CHECK(p2 != NULL, "compacted request re-renders");
        int32_t *ids2 = NULL;
        int n2 = ember_backend_encode(be, p2, &ids2);
        free(p2); free(ids2);
        CHECK(n2 > 0 && n2 < n_ctx, "compacted prompt encodes within ctx");
        printf("  exchange: %d -> %d tokens, dropped %d msgs, summary %d tok\n",
               rep.original_tokens, rep.compacted_tokens, rep.dropped_messages,
               rep.summary_tokens);
    }

    free(ids);
    ember_chat_request_free(&r);
    ember_backend_free(be);
    unsetenv("EMBER_STUB_REPLY");
}

static void test_guards(void) {
    char *err = NULL;
    ember_backend_config cfg = {0};
    cfg.model_path = "stub"; cfg.max_ctx = 512; cfg.model_name = "stub-model";
    ember_backend *be = ember_backend_load(&cfg, &err);
    CHECK(be != NULL, "backend loads");
    free(err);

    // raw_prompt (/v1/completions) has no messages to rebuild.
    ember_chat_request raw = {0};
    raw.raw_prompt = strdup("legacy completion prompt");
    raw.model = strdup("m");
    ember_compaction_report rep = {0};
    CHECK(!ember_compact_request(be, &raw, 512, 400, -1, "x", &rep),
          "raw_prompt is refused");
    CHECK(rep.error[0] != '\0', "raw refusal explained");
    ember_chat_request_free(&raw);

    // System-only history: nothing to summarize.
    ember_chat_request sys = {0};
    sys.model = strdup("m");
    sys.n_messages = 1;
    sys.messages = (ember_chat_msg *)calloc(1, sizeof(ember_chat_msg));
    sys.messages[0] = mk("system", "only a system prompt");
    memset(&rep, 0, sizeof(rep));
    CHECK(!ember_compact_request(be, &sys, 512, 400, -1, "x", &rep),
          "system-only is refused");
    ember_chat_request_free(&sys);

    // A short history whose tail already fits must be refused, not "compacted"
    // into something no smaller.
    ember_chat_request tiny;
    build_req(&tiny, 1);
    memset(&rep, 0, sizeof(rep));
    int n_msgs = tiny.n_messages;
    CHECK(!ember_compact_request(be, &tiny, 131072, 50, -1, "x", &rep),
          "already-small history refused");
    CHECK(tiny.n_messages == n_msgs, "refusal did not mutate");
    ember_chat_request_free(&tiny);

    ember_backend_free(be);
}

static void test_report_json(void) {
    ember_compaction_report r = {0};
    r.applied = true;
    r.original_tokens = 125284;
    r.compacted_tokens = 17203;
    r.summary_tokens = 4096;
    r.tail_tokens = 13107;
    r.dropped_messages = 41;
    snprintf(r.reason, sizeof(r.reason), "soft limit before user turn");

    ember_buf b = {0};
    ember_compaction_append_json(&b, &r);
    CHECK(strstr(b.ptr, "\"applied\":true") != NULL, "applied emitted");
    CHECK(strstr(b.ptr, "\"original_tokens\":125284") != NULL, "original emitted");
    CHECK(strstr(b.ptr, "\"compacted_tokens\":17203") != NULL, "compacted emitted");
    CHECK(strstr(b.ptr, "\"dropped_messages\":41") != NULL, "dropped emitted");
    ember_buf_free(&b);

    // An error must be JSON-safe (quotes escaped, newlines flattened).
    ember_compaction_report e = {0};
    snprintf(e.reason, sizeof(e.reason), "x");
    snprintf(e.error, sizeof(e.error), "bad \"quote\" and\nnewline");
    ember_buf b2 = {0};
    ember_compaction_append_json(&b2, &e);
    CHECK(strstr(b2.ptr, "\\\"quote\\\"") != NULL, "quotes escaped");
    CHECK(strchr(b2.ptr, '\n') == NULL, "no raw newline");
    ember_buf_free(&b2);
}

int main(void) {
    test_needed();
    test_tail_budget();
    test_sigil();
    test_prompt_and_wrap();
    test_head_and_boundaries();
    test_dropped_message_with_content_parts();
    test_exchange();
    test_guards();
    test_report_json();
    printf("compaction: %d checks passed\n", checks);
    return 0;
}
