#include "compaction.h"

#include <stdarg.h>
#include <stdint.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../model/chat_template.h"
#include "../model/tool_parser.h"
#include "../common/json_util.h"

// ── pure policy ─────────────────────────────────────────────────────────────

// ds4 agent_worker_should_compact (ds4_agent.c:8022-8031), verbatim policy.
bool ember_compaction_needed(int n_prompt, int n_ctx) {
    if (n_ctx <= 0 || n_prompt <= 0) return false;
    // long math: ctx * 85 overflows int only above ~25M ctx, but the cast costs
    // nothing and removes the question.
    if ((int64_t)n_prompt >=
        ((int64_t)n_ctx * EMBER_COMPACT_SOFT_PERCENT) / 100)
        return true;
    int free_threshold = EMBER_COMPACT_MIN_FREE_TOKENS;
    int proportional = n_ctx / 8;
    if (free_threshold > proportional) free_threshold = proportional;
    return n_ctx - n_prompt <= free_threshold;
}

// ds4 agent_compact_tail_start's budget (:8044-8048).
int ember_compaction_tail_budget(int n_ctx) {
    if (n_ctx <= 0) return 1;
    int budget = n_ctx / EMBER_COMPACT_TAIL_DIVISOR;
    if (budget > EMBER_COMPACT_TAIL_CAP_TOKENS) budget = EMBER_COMPACT_TAIL_CAP_TOKENS;
    if (budget < 1) budget = 1;
    return budget;
}

static bool role_is_head(const char *role) {
    return role && (!strcmp(role, "system") || !strcmp(role, "developer"));
}

int ember_compaction_head_count(const ember_chat_request *req) {
    if (!req || !req->messages) return 0;
    int n = 0;
    while (n < req->n_messages && role_is_head(req->messages[n].role)) n++;
    return n;
}

int ember_compaction_next_user_msg(const ember_chat_request *req,
                                   int head_count, int from) {
    if (!req || !req->messages) return -1;
    if (from < head_count) from = head_count;
    for (int i = from; i < req->n_messages; i++) {
        const char *r = req->messages[i].role;
        if (r && !strcmp(r, "user")) return i;
    }
    return -1;
}

// ds4 agent_compact_make_prompt (:8073-8092). Kept close to the original wording:
// it is a prompt the model was effectively tuned against in ds4, and the explicit
// prohibitions (no tools, no thinking tags, no DSML) are what keep the result
// consumable as plain text.
char *ember_compaction_make_prompt(const char *reason) {
    ember_buf b = {0};
    ember_buf_puts(&b,
        "Internal ember context compaction request. This is not a user request.\n"
        "Write a durable task-state summary of the conversation so far. Preserve only facts that matter for continuing the work:\n"
        "- user goals, constraints, and preferences\n"
        "- files inspected or edited\n"
        "- commands run and important results\n"
        "- decisions, rejected approaches, known bugs, and pending next steps\n"
        "- reloadable bulky data with exact paths/ranges/commands when available\n\n"
        "Do not invent facts. Do not include generic narration. Do not include raw file contents unless they were essential to a conclusion.\n"
        "After the summary, stop. Do not continue the user task, do not call tools, and do not output thinking tags or DSML markup.\n"
        "Output only the compact summary.\n");
    if (reason && reason[0]) {
        ember_buf_puts(&b, "\nCompaction reason: ");
        ember_buf_puts(&b, reason);
        ember_buf_puts(&b, "\n");
    }
    return b.ptr;  // NUL-terminated by ember_buf; caller frees
}

// ds4 builds this as a system message around the summary (:8242-8249).
char *ember_compaction_wrap_summary(const char *summary) {
    ember_buf b = {0};
    ember_buf_puts(&b, "\n\n" EMBER_COMPACT_SUMMARY_OPEN "\n");
    ember_buf_puts(&b, summary ? summary : "");
    if (b.len && b.ptr[b.len - 1] != '\n') ember_buf_putc(&b, '\n');
    ember_buf_puts(&b, EMBER_COMPACT_SUMMARY_CLOSE "\n\n");
    return b.ptr;
}

bool ember_compaction_text_has_sigil(const char *s) {
    if (!s || !*s) return false;
    int n = 0;
    const ember_dsml_syntax *tab = ember_dsml_syntaxes(&n);
    for (int i = 0; i < n; i++) {
        if (tab[i].calls_open && strstr(s, tab[i].calls_open)) return true;
        // An invoke opener can appear without the calls wrapper in malformed
        // output; treat it as a sigil too rather than letting it reach content.
        if (tab[i].invoke_open && strstr(s, tab[i].invoke_open)) return true;
    }
    return false;
}

// ds4 drops a lone trailing '<' when the DSML token stops the summary
// (:8197-8199). Ember's openers are multi-byte strings, so the general form is:
// remove the longest trailing run that is a proper prefix of any known opener.
void ember_compaction_trim_partial_sigil(char *s) {
    if (!s || !*s) return;
    size_t len = strlen(s);
    int n = 0;
    const ember_dsml_syntax *tab = ember_dsml_syntaxes(&n);

    size_t best = 0;
    for (int i = 0; i < n; i++) {
        const char *opens[2] = { tab[i].calls_open, tab[i].invoke_open };
        for (int k = 0; k < 2; k++) {
            const char *open = opens[k];
            if (!open || !*open) continue;
            size_t olen = strlen(open);
            // Try every proper prefix length of `open`, longest first.
            size_t try_len = olen - 1;
            while (try_len > 0) {
                if (try_len <= len &&
                    memcmp(s + len - try_len, open, try_len) == 0) {
                    if (try_len > best) best = try_len;
                    break;
                }
                try_len--;
            }
        }
    }
    if (best) s[len - best] = '\0';
}

// ── tail measurement ────────────────────────────────────────────────────────

// Render+encode a shallow view of `req` covering messages[start, n) and return
// its token count, or -1 on failure. This is ds4's measure-the-real-thing
// discipline (agent_tool_result_fits_context :6045 literally re-tokenizes rather
// than estimating): a compaction that trusts an estimate is how you end up
// reporting a token count the next request contradicts.
//
// The view borrows message pointers — it must never be freed as a request.
static int measure_range(ember_backend *be, const ember_chat_request *req,
                         int start, bool enable_thinking) {
    if (start < 0 || start > req->n_messages) return -1;
    ember_chat_request view = *req;
    view.messages = req->messages + start;
    view.n_messages = req->n_messages - start;
    view.raw_prompt = NULL;

    char *prompt = ember_render_prompt(&view, enable_thinking, req->think_mode, true);
    if (!prompt) return -1;
    int32_t *ids = NULL;
    int n = ember_backend_encode(be, prompt, &ids);
    free(prompt);
    free(ids);
    return n;
}

// Largest verbatim tail that fits `tail_budget`, expressed as a message index.
// Candidates are user-turn boundaries only, so the rebuilt context always opens
// on a user turn like ds4's (:8053-8059). Tail cost is monotonically
// non-increasing in the start index, so a binary search over candidates is valid
// and costs ~log2(n) encodes instead of n.
static int choose_tail_start(ember_backend *be, const ember_chat_request *req,
                             int head_count, int tail_budget,
                             bool enable_thinking, int *tail_tokens_out) {
    // Collect user-boundary candidates after the head.
    int cap = req->n_messages - head_count;
    if (cap <= 0) return -1;
    int *cand = (int *)malloc((size_t)cap * sizeof(int));
    if (!cand) return -1;
    int n_cand = 0;
    for (int i = head_count; i < req->n_messages; i++) {
        const char *r = req->messages[i].role;
        if (r && !strcmp(r, "user")) cand[n_cand++] = i;
    }
    if (n_cand == 0) { free(cand); return -1; }

    // Find the smallest candidate index (largest tail) whose cost fits.
    int lo = 0, hi = n_cand - 1, best = -1, best_tokens = 0;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        int cost = measure_range(be, req, cand[mid], enable_thinking);
        if (cost < 0) { free(cand); return -1; }
        if (cost <= tail_budget) {
            best = cand[mid];
            best_tokens = cost;
            hi = mid - 1;          // try to keep MORE history
        } else {
            lo = mid + 1;          // too big; start later
        }
    }
    free(cand);
    if (best < 0) return -1;
    if (tail_tokens_out) *tail_tokens_out = best_tokens;
    return best;
}

// Build the private compaction prompt covering head + messages[from, to) + the
// instruction, render it with thinking OFF, and encode. Returns the token count
// (>=0) and, when `ids_out` is non-NULL, the token array (caller frees).
//
// Only the REPLACED range is shown to the model: the verbatim tail survives
// compaction untouched, so including it would just spend context re-reading text
// that is already going to be in the rebuilt prompt.
static int build_compaction_prompt(ember_backend *be,
                                   const ember_chat_request *req,
                                   int head_count, int from, int to,
                                   const char *instruction,
                                   int32_t **ids_out) {
    const int n_body = to - from;
    if (head_count < 0 || from < head_count || to < from ||
        to > req->n_messages || n_body < 0 ||
        head_count > INT_MAX - n_body - 1) return -1;
    const int n_aug = head_count + n_body + 1;
    ember_chat_msg *aug =
        (ember_chat_msg *)calloc((size_t)n_aug, sizeof(ember_chat_msg));
    if (!aug) return -1;

    for (int i = 0; i < head_count; i++) aug[i] = req->messages[i];
    for (int i = 0; i < n_body; i++) aug[head_count + i] = req->messages[from + i];
    // Borrowed literals/pointers: this view is never freed as a request.
    aug[n_aug - 1].role = (char *)"user";
    aug[n_aug - 1].content = (char *)instruction;

    ember_chat_request view = *req;
    view.messages = aug;
    view.n_messages = n_aug;
    view.raw_prompt = NULL;
    view.has_tools = false;      // the summary must never be a tool call
    view.tools_json = NULL;
    view.thinking_enabled = false;
    view.think_mode = EMBER_THINK_NONE;

    char *prompt = ember_render_prompt(&view, false, EMBER_THINK_NONE, true);
    free(aug);
    if (!prompt) return -1;

    int32_t *ids = NULL;
    int n = ember_backend_encode(be, prompt, &ids);
    free(prompt);
    if (ids_out) *ids_out = ids; else free(ids);
    return n;
}

// Earliest user boundary in [head_count, to) whose compaction prompt still
// leaves MIN_SUMMARY_ROOM. Returns -1 if even the smallest window fails.
//
// ds4 does not need this: it compacts at 85%, so its transcript always fits and
// the whole history can be summarized in one pass. Ember can be handed a history
// that already exceeds ctx, and then the full range simply cannot be read. Taking
// the most recent window that fits keeps the most relevant state; whatever falls
// outside it is reported as dropped_unsummarized rather than silently lost.
static int choose_summary_window(ember_backend *be,
                                 const ember_chat_request *req,
                                 int head_count, int to, int n_ctx,
                                 const char *instruction) {
    int cap = to - head_count;
    if (cap <= 0) return -1;
    int *cand = (int *)malloc((size_t)cap * sizeof(int));
    if (!cand) return -1;
    int n_cand = 0;
    for (int i = head_count; i < to; i++) {
        const char *r = req->messages[i].role;
        if (r && !strcmp(r, "user")) cand[n_cand++] = i;
    }
    if (n_cand == 0) { free(cand); return -1; }

    // Cost is non-increasing in the start index, so binary search for the
    // earliest (largest window) start that fits.
    int lo = 0, hi = n_cand - 1, best = -1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        int n = build_compaction_prompt(be, req, head_count, cand[mid], to,
                                        instruction, NULL);
        if (n >= 0 && n_ctx - n - 1 >= EMBER_COMPACT_MIN_SUMMARY_ROOM) {
            best = cand[mid];
            hi = mid - 1;          // try to summarize MORE history
        } else {
            lo = mid + 1;          // too big; start later
        }
    }
    free(cand);
    return best;
}

// ── summary generation ──────────────────────────────────────────────────────

typedef struct {
    ember_backend *be;
    ember_buf      text;
    int            n_tokens;
    bool           sigil;
} summary_sink;

static bool summary_on_token(int32_t tok, void *ud) {
    summary_sink *s = (summary_sink *)ud;
    const char *t = ember_backend_token_text(s->be, tok);
    if (t && *t) ember_buf_puts(&s->text, t);
    s->n_tokens++;
    // ds4 stops the moment the DSML token is sampled (:8193-8201). Ember has no
    // single DSML token, so detect the opener in the decoded text and cancel.
    if (s->text.ptr && ember_compaction_text_has_sigil(s->text.ptr)) {
        s->sigil = true;
        return false;
    }
    return true;
}

// ── the exchange ────────────────────────────────────────────────────────────

static void free_msg(ember_chat_msg *m) {
    free(m->role);
    free(m->content);
    free(m->name);
    free(m->reasoning);
    free(m->tool_call_id);
    free(m->raw_tool_text);
    ember_tool_calls_free(&m->calls);
    memset(m, 0, sizeof(*m));
}

static void set_err(ember_compaction_report *r, const char *fmt, ...) {
    if (!r) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(r->error, sizeof(r->error), fmt, ap);
    va_end(ap);
}

bool ember_compact_request(ember_backend *be,
                           ember_chat_request *req,
                           int n_ctx,
                           int n_prompt,
                           int disk_slot,
                           const char *reason,
                           ember_compaction_report *report) {
    if (!be || !req || !report || n_ctx <= 0 || n_prompt < 0 ||
        req->n_messages < 0 ||
        (req->n_messages > 0 && !req->messages)) return false;

    memset(report, 0, sizeof(*report));
    report->original_tokens = n_prompt;
    snprintf(report->reason, sizeof(report->reason), "%s", reason ? reason : "context");

    // /v1/completions has no message structure to rebuild, and a
    // continuation-only request has no stateless prompt at all.
    if (req->raw_prompt || req->continuation_only || req->n_messages <= 0) {
        set_err(report, "no message history to compact");
        return false;
    }

    const int head_count = ember_compaction_head_count(req);
    if (head_count >= req->n_messages) {
        set_err(report, "nothing but system messages to compact");
        return false;
    }

    const int tail_budget = ember_compaction_tail_budget(n_ctx);
    int tail_tokens = 0;
    const int tail_start = choose_tail_start(be, req, head_count, tail_budget,
                                             req->thinking_enabled, &tail_tokens);
    if (tail_start < 0) {
        set_err(report, "no user-turn boundary fits the %d-token tail budget",
                tail_budget);
        return false;
    }
    if (tail_start <= head_count) {
        // The whole conversation already fits the tail budget, so a summary
        // would replace nothing. Refusing here keeps the "compaction always
        // shrinks" invariant honest instead of burning a generation.
        set_err(report, "tail already spans the full history; nothing to summarize");
        return false;
    }

    // ── ask the model for durable task state ────────────────────────────────
    // The compaction prompt is the CURRENT history plus a private instruction,
    // rendered with thinking OFF (ds4 uses DS4_THINK_NONE, :8119). Thinking here
    // would spend the entire summary budget on reasoning we then discard.
    char *instruction = ember_compaction_make_prompt(reason);
    if (!instruction) { set_err(report, "out of memory"); return false; }

    // The window of replaced history the model can actually read in one pass.
    const int win_start = choose_summary_window(be, req, head_count, tail_start,
                                               n_ctx, instruction);
    if (win_start < 0) {
        free(instruction);
        set_err(report, "no summary window fits: even the shortest leaves under "
                        "%d tokens of room", EMBER_COMPACT_MIN_SUMMARY_ROOM);
        return false;
    }

    int32_t *cids = NULL;
    const int n_cprompt = build_compaction_prompt(be, req, head_count, win_start,
                                                  tail_start, instruction, &cids);
    free(instruction);
    if (n_cprompt < 0) {
        free(cids);
        set_err(report, "compaction prompt render/tokenize failed");
        return false;
    }

    // ds4 refuses below 256 tokens of room (:8135-8141). choose_summary_window
    // already enforced this; re-checking keeps the invariant local and obvious.
    const int summary_room = n_ctx - n_cprompt - 1;
    if (summary_room < EMBER_COMPACT_MIN_SUMMARY_ROOM) {
        free(cids);
        set_err(report, "not enough context left to request compaction summary "
                        "(room=%d, need=%d)", summary_room,
                        EMBER_COMPACT_MIN_SUMMARY_ROOM);
        return false;
    }
    int summary_max = summary_room < EMBER_COMPACT_SUMMARY_MAX_TOKENS
                    ? summary_room : EMBER_COMPACT_SUMMARY_MAX_TOKENS;

    summary_sink sink = {0};
    sink.be = be;

    ember_gen_request greq = {0};
    greq.prompt = cids;
    greq.n_prompt = n_cprompt;
    greq.max_tokens = summary_max;
    greq.greedy = true;          // deterministic: same history -> same summary
    greq.temperature = 0.0f;
    greq.top_p = 1.0f;
    greq.min_p = 0.0f;
    greq.rep_pen = 1.0f;
    // No force-close: thinking is off, so there is no </think> to inject.
    greq.budget_close_ids = NULL;
    greq.n_budget_close = 0;
    greq.reply_budget = 0;
    // Reuse cached KV for the longest cached prefix OF THIS PROMPT. Resolving it
    // here (rather than accepting the caller's slot for the original prompt) is
    // what makes the restore sound: the private instruction changes the tail, so
    // only a genuine prefix lookup against `cids` is safe. Never snapshot — this
    // prompt must not become an entry a later real turn could restore from.
    greq.restore_slot = -1;
    greq.snap_slot = -1;
    greq.snap_pos = -1;
    if (disk_slot >= 0 && ember_backend_disk_enabled(be)) {
        int hit = ember_backend_disk_prefix(be, cids, n_cprompt);
        if (hit > 0 && ember_backend_disk_lookup(be, cids, hit, disk_slot))
            greq.restore_slot = disk_slot;
    }
    greq.on_token = summary_on_token;
    greq.ud = &sink;

    ember_gen_result gres = ember_backend_generate(be, &greq);
    free(cids);

    // A cancel caused by our own sigil stop is success, not failure.
    if (!gres.ok && !(gres.cancelled && sink.sigil)) {
        ember_buf_free(&sink.text);
        set_err(report, "compaction generation failed: %s%s%s",
                gres.error_code[0] ? gres.error_code : "unknown",
                gres.error_detail[0] ? ": " : "",
                gres.error_detail[0] ? gres.error_detail : "");
        return false;
    }

    if (sink.text.ptr) ember_compaction_trim_partial_sigil(sink.text.ptr);
    if (!sink.text.ptr || !sink.text.ptr[0]) {
        ember_buf_free(&sink.text);
        set_err(report, "compaction summary was empty");
        return false;
    }

    char *summary_body = ember_compaction_wrap_summary(sink.text.ptr);
    int summary_gen_tokens = sink.n_tokens;
    bool sigil_stopped = sink.sigil;
    ember_buf_free(&sink.text);
    if (!summary_body) { set_err(report, "out of memory"); return false; }

    // ── rebuild: head + summary + verbatim tail ─────────────────────────────
    const int n_tail = req->n_messages - tail_start;
    const int n_new = head_count + 1 + n_tail;
    ember_chat_msg *rebuilt =
        (ember_chat_msg *)calloc((size_t)n_new, sizeof(ember_chat_msg));
    if (!rebuilt) { free(summary_body); set_err(report, "out of memory"); return false; }

    for (int i = 0; i < head_count; i++) rebuilt[i] = req->messages[i];
    rebuilt[head_count].role = strdup("system");
    rebuilt[head_count].content = summary_body;   // ownership moves here
    if (!rebuilt[head_count].role) {
        free(summary_body);
        free(rebuilt);
        set_err(report, "out of memory");
        return false;
    }
    for (int i = 0; i < n_tail; i++)
        rebuilt[head_count + 1 + i] = req->messages[tail_start + i];

    // Prove the rebuild fits BEFORE committing, so a failure leaves the request
    // untouched. ds4's escalation ladder does the same thing: compact, re-check,
    // and only then proceed (:8713-8733). This encode is for the decision only —
    // the caller re-encodes through its own splice-aware path.
    ember_chat_request rview = *req;
    rview.messages = rebuilt;
    rview.n_messages = n_new;
    rview.raw_prompt = NULL;
    char *rprompt = ember_render_prompt(&rview, req->thinking_enabled,
                                        req->think_mode, true);
    int n_rprompt = -1;
    if (rprompt) {
        int32_t *rids = NULL;
        n_rprompt = ember_backend_encode(be, rprompt, &rids);
        free(rids);
        free(rprompt);
    }
    if (n_rprompt < 0 || n_rprompt >= n_ctx) {
        free(rebuilt[head_count].role);
        free(rebuilt[head_count].content);
        free(rebuilt);
        set_err(report, "compacted prompt still does not fit (%d tokens, ctx=%d)",
                n_rprompt, n_ctx);
        return false;
    }

    // Commit. The messages replaced by the summary are freed here; head and tail
    // entries were moved by value into `rebuilt` and must NOT be freed.
    for (int i = head_count; i < tail_start; i++) free_msg(&req->messages[i]);
    free(req->messages);
    req->messages = rebuilt;
    req->n_messages = n_new;

    report->applied = true;
    report->compacted_tokens = n_rprompt;
    report->summary_tokens = summary_gen_tokens;
    report->tail_tokens = tail_tokens;
    report->head_messages = head_count;
    report->tail_start_msg = tail_start;
    report->dropped_messages = tail_start - head_count;
    // Messages removed WITHOUT being represented in the summary, because the
    // history was too large to read in one pass. Zero in the normal case.
    report->dropped_unsummarized = win_start - head_count;
    report->summary_window_msg = win_start;
    report->sigil_stopped = sigil_stopped;
    return true;
}

void ember_compaction_append_json(ember_buf *b, const ember_compaction_report *r) {
    if (!b || !r) return;
    ember_buf_printf(b,
        "\"compaction\":{\"applied\":%s,\"original_tokens\":%d,"
        "\"compacted_tokens\":%d,\"summary_tokens\":%d,\"tail_tokens\":%d,"
        "\"dropped_messages\":%d,\"reason\":",
        r->applied ? "true" : "false", r->original_tokens, r->compacted_tokens,
        r->summary_tokens, r->tail_tokens, r->dropped_messages);
    ember_json_escape(b, r->reason);
    // Only emitted when nonzero: it means history was lost outright, which a
    // client should be able to notice.
    if (r->dropped_unsummarized > 0)
        ember_buf_printf(b, ",\"dropped_unsummarized\":%d",
                         r->dropped_unsummarized);
    if (r->sigil_stopped) ember_buf_puts(b, ",\"sigil_stopped\":true");
    if (r->error[0]) {
        ember_buf_puts(b, ",\"error\":");
        ember_json_escape(b, r->error);
    }
    ember_buf_putc(b, '}');
}
