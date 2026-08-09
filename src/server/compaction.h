// Context compaction — ported from ds4's agent-side compaction
// (ds4_agent.c:8010-8292, "Context Compaction").
//
// ds4 asks the model for durable task state, then rebuilds the transcript as
// `system prompt + summary + recent verbatim tail`. That keeps the active KV
// usable while bounding transcript growth, and the result is bounded BY
// CONSTRUCTION: a rebuild is at most sys + SUMMARY_MAX + tail_budget, so it
// cannot fail to shrink no matter how poor the summary is. That property is the
// whole point — a compaction pass that summarizes and hopes the output is
// smaller can report "still over budget after compression", which is exactly
// the failure mode this replaces.
//
// Two deliberate differences from ds4, both forced by where the code lives:
//
//   1. ds4's agent OWNS the transcript, so it compacts a raw token array and
//      snaps the tail to a `<｜User｜>` token. Ember is a server: it receives
//      structured messages every request, so it snaps to real message
//      boundaries instead. That is strictly more reliable than scanning for a
//      marker token, and it is why the tail search here is over message indices.
//
//   2. ds4 compacts once and keeps the shrunken transcript. Ember is stateless
//      — the client re-sends the full history next turn — so compaction is
//      recomputed per request. It stays affordable because the summary prompt is
//      a prefix of what the KV/disk cache already holds, and because it only
//      runs once the request is already near the context ceiling.
//
// Nothing here mutates the client's view of its own history: compaction changes
// what THIS generation is conditioned on, is opt-in per server, and is reported
// so it is never silent.
#ifndef EMBER_COMPACTION_H
#define EMBER_COMPACTION_H

#include <stdbool.h>
#include <stdint.h>
#include "chat_api.h"
#include "../common/buf.h"
#include "../backend/ember_backend.h"

// ── policy (ds4_agent.c:5872-5876) ──────────────────────────────────────────
// Compact at 85% of context, or when free space falls below
// min(8192, ctx/8) — the proportional cap keeps tiny test contexts compacting
// rather than failing. Tail is ctx/10 capped at 50k. Summary is capped at 4096
// tokens, and we refuse to try at all with less than 256 tokens of room.
#define EMBER_COMPACT_SOFT_PERCENT        85
#define EMBER_COMPACT_MIN_FREE_TOKENS     8192
#define EMBER_COMPACT_TAIL_DIVISOR        10
#define EMBER_COMPACT_TAIL_CAP_TOKENS     50000
#define EMBER_COMPACT_SUMMARY_MAX_TOKENS  4096
#define EMBER_COMPACT_MIN_SUMMARY_ROOM    256

// Markers wrapping the injected summary. ds4 uses these verbatim so /history can
// find and re-render the latest summary (ds4_agent.c:5104-5106); ember keeps the
// same shape so the text is recognizable in logs and transcripts.
#define EMBER_COMPACT_SUMMARY_OPEN \
    "[ember compacted earlier conversation. Durable task-state summary follows.]"
#define EMBER_COMPACT_SUMMARY_CLOSE \
    "[End compacted summary. Recent conversation continues verbatim below.]"

typedef struct {
    bool applied;
    int  original_tokens;    // prompt length before compaction
    int  compacted_tokens;   // prompt length after (0 when !applied)
    int  summary_tokens;     // tokens the model spent on the summary
    int  tail_tokens;        // measured cost of the verbatim tail
    int  head_messages;      // leading system/developer messages kept
    int  tail_start_msg;     // index of the first verbatim message kept
    int  dropped_messages;   // messages replaced by the summary
    // Messages dropped with NO representation in the summary. Nonzero only when
    // the incoming history was too large to read in a single pass — ember is
    // stateless and can be handed a history that already exceeds ctx, which ds4
    // never sees because it compacts proactively at 85%. Surfaced so that loss is
    // visible in usage rather than silent.
    int  dropped_unsummarized;
    int  summary_window_msg; // first message the summary actually covers
    bool sigil_stopped;      // summary generation hit a DSML opener and stopped
    char reason[48];         // why compaction ran (mirrors ds4's reason string)
    char error[192];         // set when compaction was attempted and failed
} ember_compaction_report;

// ── pure policy helpers (no backend; unit-tested directly) ───────────────────

// ds4 agent_worker_should_compact (:8022).
bool ember_compaction_needed(int n_prompt, int n_ctx);

// ds4 agent_compact_tail_start's budget (:8044-8048).
int  ember_compaction_tail_budget(int n_ctx);

// Count of leading system/developer messages, which are always re-emitted ahead
// of the summary (ds4's agent_worker_build_system_tokens equivalent).
int  ember_compaction_head_count(const ember_chat_request *req);

// Lowest index > head_count that begins a user turn, at or after `from`, or -1.
// ds4 scans forward for `<｜User｜>` so the rebuilt context opens on a natural
// turn (:8057-8059); the message-level equivalent is the next role=="user".
int  ember_compaction_next_user_msg(const ember_chat_request *req,
                                    int head_count, int from);

// The private compaction instruction (ds4 agent_compact_make_prompt :8073).
// Caller frees.
char *ember_compaction_make_prompt(const char *reason);

// Wrap a raw summary in the open/close markers as a system message body.
// Caller frees.
char *ember_compaction_wrap_summary(const char *summary);

// True if `s` contains any recognized DSML tool_calls opener. ds4 stops the
// summary the moment the DSML token is sampled (:8196) because the summary is
// consumed internally and must not carry tool markup.
bool ember_compaction_text_has_sigil(const char *s);

// Drop a trailing PARTIAL DSML opener from `s`, in place. ds4 handles the
// single-byte case (`summary.ptr[len-1] == '<'`, :8197-8199); ember's openers are
// multi-byte, so any trailing prefix of one is trimmed. This is also the guard
// the DSML content leak needs: a partial sigil must never survive into content.
void ember_compaction_trim_partial_sigil(char *s);

// ── the full exchange ────────────────────────────────────────────────────────
// Runs the compaction turn and, on success, rewrites req->messages in place to
// head + summary + tail. It deliberately does NOT hand back a token array: the
// caller must re-render and re-encode through its normal path (encode_with_splices)
// so exact-DSML tool-call replay keeps its token identity. Compaction does encode
// the rebuild once internally, but only to prove it fits before committing —
// report->compacted_tokens is that measurement, and the caller should overwrite
// it with the authoritative count from its own encode.
//
// `disk_slot` may be >= 0 to name a KV slot compaction may load a disk snapshot
// into, so the summary prefill is not paid from scratch. Compaction resolves the
// longest cached PREFIX of its OWN prompt via the backend's prefix lookup, which
// is why the caller must not hand over the restore slot it found for the original
// prompt: appending the private instruction changes the prompt's tail, so a slot
// covering the full original prompt is not a prefix of the compaction prompt and
// restoring it would condition the summary on the wrong state. Pass -1 to skip
// reuse — correct, but it pays a full prefill.
//
// This never snapshots: the compaction prompt is private and must not become a
// cache entry a later real turn could restore from.
//
// On failure returns false, leaves req UNTOUCHED, and sets report->error.
//
// ds4 invalidates live KV on every failure path because the KV then holds the
// private compaction exchange (:8171-8174). Ember gets that structurally: the
// compaction generation never snapshots, so no slot is left describing the
// private prompt, and the caller resolves KV reuse for the real generation only
// after this returns — see the call site in run_chat.
//
// `n_prompt` is the current prompt length, used only for reporting.
bool ember_compact_request(ember_backend *be,
                           ember_chat_request *req,
                           int n_ctx,
                           int n_prompt,
                           int disk_slot,
                           const char *reason,
                           ember_compaction_report *report);

// Append `"compaction":{...}` (no leading comma) describing `r`. Emitted inside
// usage so a compacted generation is observable rather than silent.
void ember_compaction_append_json(ember_buf *b,
                                  const ember_compaction_report *r);

#endif  // EMBER_COMPACTION_H
