#ifndef EMBER_SERVER_METRICS_H
#define EMBER_SERVER_METRICS_H

#include <stdbool.h>

#include "../common/buf.h"

// Scrapeable aggregates for /metrics.
//
// /status answers "what is true now"; a scraper needs monotonic series it can
// difference over time. Establishing prefix-cache behaviour during the
// 2026-09-05 soak meant parsing 3,495 log lines across four boots, and 833
// consecutive generations declined speculation without anything counting it.
// Every value here is already computed per request and was only ever printed.
//
// All entry points are safe to call from any request thread.

// One completed generation. `finish_reason` is mapped onto a fixed label set,
// so a novel reason cannot grow the series cardinality without a code change.
// `ttft_s` is MEASURED: the first token's own arrival minus the moment the
// request was enqueued. It is not queue + prefill. Those are backend durations
// and exclude the FIFO wait behind other generations, image encoding and prompt
// preparation, none of which a caller experiences as free. Pass a negative
// value when no token was produced, and no TTFT is recorded -- a request that
// emitted nothing has no time-to-first-token, and recording zero would pull the
// distribution toward a latency that never happened.
//
// `mean_token_gap_s` is likewise measured, as (last - first) / (tokens - 1),
// and is a PER-REQUEST MEAN rather than a distribution of individual gaps.
// Pass a negative value when fewer than two tokens were produced.
void ember_metrics_record_generation(const char *finish_reason,
                                     int prefill_tokens, int completion_tokens,
                                     double ttft_s, double mean_token_gap_s,
                                     double prefill_s, double decode_s,
                                     bool spec_engaged, double accept_rate,
                                     int n_images);

// Why speculation did not run. #13 recorded 833 of 833 generations declining
// for a single reason with nothing counting it, and the engine already computes
// this string -- it was simply never leaving the backend. NULL or "" means
// speculation ran, which is not counted here.
void ember_metrics_record_spec_decline(const char *reason);

// One image encode. `seconds` spans the whole encode call as the server sees
// it -- preprocessing, a lazy first-use tower load, execution and the result
// copy -- NOT the tower alone. That is deliberately the end-to-end cost a
// caller pays, and it is what would expose a projector silently falling back to
// CPU, the way ggml-org/llama.cpp#22582 was diagnosed. Do not read a change
// here as tower execution alone; a first-use load lands in the same series.
void ember_metrics_record_vision_encode(double seconds, int image_tokens);

// Prompt-cache reuse for one request: tokens presented and tokens restored.
void ember_metrics_record_prefix_cache(int prompt_tokens, int restored_tokens);

// Seconds a request waited before generation began. Ember serialises
// generation, so this is the dominant latency term rather than a footnote.
void ember_metrics_record_queue_wait(double seconds);

// ── Request lifecycle ───────────────────────────────────────────────────
// The queue histogram above records a wait only once it is over. A scraper
// asking "is anything queued right now" got no answer, and the 2026-09-10
// review found a p95 queue wait of 60 s with nothing showing depth at the
// time it was happening. Three calls bracket a job's life on the worker FIFO:
// enqueued (waiting), started (waiting -> running), finished (running -> done,
// with the end-to-end seconds from enqueue to response). They render as one
// gauge family with a `state` label and one histogram.
void ember_metrics_job_enqueued(void);
void ember_metrics_job_started(void);
void ember_metrics_job_finished(double end_to_end_s);

// A job refused at the FIFO because it was full (the 503 shed path). Counted
// apart from the 503 response itself: an overload 503 and a 503 for any other
// reason are different operational facts.
void ember_metrics_record_shed(void);

// One HTTP response written, by status. The label set is closed (200, 204,
// 400, 404, 409, 422, 429, 500, 503, other), so a novel code cannot grow the
// series. Streaming responses count once, when their headers are written.
void ember_metrics_record_response(int status);

// How a generation ended from the server's point of view, which the finish
// reason cannot say: "ok", "client_disconnected", "cancelled",
// "backend_error", "stalled". Closed set; unknown maps to "other".
void ember_metrics_record_outcome(const char *outcome);

// One gap between two consecutive generated tokens, in seconds. This IS a
// distribution of individual gaps, unlike
// ember_request_mean_token_gap_seconds, so a p99 read here is meaningful. The
// ladder is finer than the request-level one because a gap is tens of
// milliseconds and a 50 ms first bucket would put every observation in it.
void ember_metrics_record_token_gap(double seconds);

// Speculative-decode detail for an engaged generation: verify cycles and the
// time split the backend already reports (head, verify, provider block). All
// counters, so a rate() over them is the share of wall time speculation spent
// in each phase. A generation with no cycles records nothing.
void ember_metrics_record_spec_detail(int cycles, double head_s,
                                      double verify_s, double provider_block_s);

// KV snapshot persistence: whether a snapshot slot was requested for this
// generation and whether the backend actually saved into it. The server
// commits its prefix entry only on `saved`; a requested-but-unsaved snapshot is
// the failure the prefix cache silently absorbs.
void ember_metrics_record_snapshot(bool requested, bool saved);

// One accepted chat-style request, attributed by protocol and by client
// family. `api` is the protocol name the adapter already knows; `client` is
// derived from the User-Agent through ember_metrics_client_family so the
// label set stays closed. On a serialising server "who is queueing behind
// whom" is the question this answers.
void ember_metrics_record_request(const char *api, const char *client);

// Map a raw User-Agent onto the closed client label set: "hermes", "openai",
// "anthropic", "curl", "python", "node", "browser", "none" (no header) or
// "other". Substring matches, first hit wins, in that order. Never allocates.
const char *ember_metrics_client_family(const char *user_agent);

// Render the Prometheus text exposition format into `out`. Besides the
// counters above this emits `ember_build_info`, `process_start_time_seconds`
// and the standard `process_*` gauges read from /proc at render time. The
// proc root is `EMBER_METRICS_PROC_DIR` when set (tests point it at a
// fixture), else /proc; when a file is unreadable its samples are omitted
// rather than rendered as zero.
void ember_metrics_render(ember_buf *out);

#endif  // EMBER_SERVER_METRICS_H
