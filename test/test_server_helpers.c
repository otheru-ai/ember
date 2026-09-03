// Unit tests for ember-server's internal helpers.
//
// These live in main.c as `static`, so the only way to reach them without
// moving production code mid-release is to compile main.c into this
// translation unit with its entry point renamed. That is why the include and
// the `#define main` below look unusual: they are deliberate, and they keep
// the shipped binary byte-for-byte unchanged.
//
// The functions covered here are pure: they take strings and request structs
// and return decisions. The rest of main.c's uncovered surface is the request
// path and the validation arms, which need a live backend and an 85 GiB model,
// so it is not reachable from a unit test and is not attempted here.

#define main ember_server_main_under_test
#include "../src/server/main.c"
#undef main

#include <assert.h>

static int failures;

static void check(bool ok, const char *what) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) failures++;
}

// ---------------------------------------------------------------- think tags

static void test_visible_after_think(void) {
    printf("visible_after_think\n");
    check(strcmp(visible_after_think("plain"), "plain") == 0,
          "text with no think block is returned whole");
    check(strcmp(visible_after_think("<think>hidden</think>shown"), "shown") == 0,
          "everything up to and including the close tag is skipped");
    // The LAST close tag wins: a model that reopens reasoning must not have its
    // final answer truncated at the first tag.
    check(strcmp(visible_after_think("<think>a</think>mid<think>b</think>end"),
                 "end") == 0,
          "the last close tag wins, not the first");
    check(strcmp(visible_after_think("</think>"), "") == 0,
          "a close tag at the very end yields the empty string");
    check(strcmp(visible_after_think("no close <think>open"),
                 "no close <think>open") == 0,
          "an unclosed think block leaves the text untouched");
}

// -------------------------------------------------------- continuation ids

static ember_chat_msg mk_msg(const char *role, const char *id) {
    ember_chat_msg m = {0};
    m.role = (char *)role;
    m.tool_call_id = (char *)id;
    return m;
}

static void test_continuation_request_ids(void) {
    printf("continuation_request_ids\n");
    const char *ids[4];

    ember_chat_msg two[] = { mk_msg("tool", "a"), mk_msg("function", "b") };
    ember_chat_request req = {0};
    req.continuation_only = true;
    req.messages = two;
    req.n_messages = 2;
    check(continuation_request_ids(&req, ids, 4) == 2 &&
          strcmp(ids[0], "a") == 0 && strcmp(ids[1], "b") == 0,
          "tool and function roles both count, in order");

    // Duplicates collapse: two results for one call are one binding.
    ember_chat_msg dup[] = { mk_msg("tool", "a"), mk_msg("tool", "a") };
    req.messages = dup;
    req.n_messages = 2;
    check(continuation_request_ids(&req, ids, 4) == 1,
          "a repeated tool_call_id is counted once");

    // Any non-tool message disqualifies the whole request, because a
    // continuation must be tool results only.
    ember_chat_msg mixed[] = { mk_msg("tool", "a"), mk_msg("user", "b") };
    req.messages = mixed;
    req.n_messages = 2;
    check(continuation_request_ids(&req, ids, 4) == 0,
          "a non-tool message disqualifies the request");

    ember_chat_msg noid[] = { mk_msg("tool", NULL) };
    req.messages = noid;
    req.n_messages = 1;
    check(continuation_request_ids(&req, ids, 4) == 0,
          "a tool message with no call id disqualifies the request");

    ember_chat_msg empty_id[] = { mk_msg("tool", "") };
    req.messages = empty_id;
    req.n_messages = 1;
    check(continuation_request_ids(&req, ids, 4) == 0,
          "an empty call id is not a call id");

    req.messages = two;
    req.n_messages = 2;
    req.continuation_only = false;
    check(continuation_request_ids(&req, ids, 4) == 0,
          "a request that is not continuation_only yields nothing");
    req.continuation_only = true;

    // Overflowing the caller's array returns 0 rather than writing past it.
    ember_chat_msg three[] = { mk_msg("tool", "a"), mk_msg("tool", "b"),
                               mk_msg("tool", "c") };
    req.messages = three;
    req.n_messages = 3;
    check(continuation_request_ids(&req, ids, 2) == 0,
          "more ids than capacity returns 0 rather than overflowing");

    check(continuation_request_ids(NULL, ids, 4) == 0, "NULL request is 0");
}

// -------------------------------------------------------- signature matching

static void test_continuation_signature(void) {
    printf("continuation_ids_signature / continuation_signature_matches\n");
    const char *ids[] = { "call_1", "call_2" };

    char *sig = continuation_ids_signature(ids, 2);
    check(sig && strcmp(sig, "call_1\ncall_2") == 0,
          "signature is the ids joined by newlines");
    check(continuation_signature_matches(sig, ids, 2),
          "a signature matches the ids it was built from");

    // Order must not matter: the binding is a set of call ids.
    const char *swapped[] = { "call_2", "call_1" };
    check(continuation_signature_matches(sig, swapped, 2),
          "order does not matter, the signature is a set");

    const char *subset[] = { "call_1" };
    check(!continuation_signature_matches(sig, subset, 1),
          "a subset does not match: the line count differs");

    const char *other[] = { "call_1", "call_9" };
    check(!continuation_signature_matches(sig, other, 2),
          "an unknown id does not match");

    // A prefix must not match a longer id, or a stale binding could be
    // mistaken for a live one.
    const char *prefix[] = { "call_", "call_2" };
    check(!continuation_signature_matches(sig, prefix, 2),
          "a prefix of an id is not that id");
    free(sig);

    char *one = continuation_ids_signature(ids, 1);
    check(one && strcmp(one, "call_1") == 0, "a single id has no separator");
    check(continuation_signature_matches(one, ids, 1), "single id matches");
    free(one);

    char *none = continuation_ids_signature(ids, 0);
    check(none && none[0] == '\0', "zero ids gives an empty signature");
    check(continuation_signature_matches(none, ids, 0),
          "an empty signature matches zero ids");
    check(!continuation_signature_matches(none, ids, 1),
          "an empty signature does not match one id");
    free(none);

    check(!continuation_signature_matches(NULL, ids, 2),
          "a missing signature never matches");
}

// ------------------------------------------------------- tool-result AR rule

static void test_tool_result_forces_ar(void) {
    printf("tool_result_forces_ar\n");
    // The cache is thread-local and computed once, so each expectation runs on
    // its own thread rather than trying to reset a static.
    unsetenv("EMBER_TOOL_RESULT_AR");
    check(tool_result_forces_ar(), "unset means on");
}

static void *ar_thread(void *arg) {
    const char *value = (const char *)arg;
    if (value) setenv("EMBER_TOOL_RESULT_AR", value, 1);
    else unsetenv("EMBER_TOOL_RESULT_AR");
    return (void *)(intptr_t)(tool_result_forces_ar() ? 1 : 0);
}

static bool ar_on_thread(const char *value) {
    pthread_t t;
    void *r = NULL;
    pthread_create(&t, NULL, ar_thread, (void *)value);
    pthread_join(t, &r);
    return (intptr_t)r != 0;
}

static void test_tool_result_forces_ar_values(void) {
    check(!ar_on_thread("0"), "\"0\" disables it");
    // Only a literal 0 disables: an unset or malformed value keeps today's
    // behaviour rather than silently lifting the rule.
    check(ar_on_thread("1"), "\"1\" keeps it on");
    check(ar_on_thread(""), "an empty value keeps it on");
    check(ar_on_thread("false"), "\"false\" keeps it on, only \"0\" disables");
    check(!ar_on_thread("00"), "\"00\" disables: the check is on the first byte");
}

int main(void) {
    printf("ember-server helper tests\n\n");
    test_visible_after_think();
    test_continuation_request_ids();
    test_continuation_signature();
    test_tool_result_forces_ar();
    test_tool_result_forces_ar_values();
    printf(failures ? "\nFAILURES: %d\n" : "\nALL PASS\n", failures);
    return failures ? 1 : 0;
}
