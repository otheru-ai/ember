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

// Render the Prometheus text exposition format into `out`.
void ember_metrics_render(ember_buf *out);

#endif  // EMBER_SERVER_METRICS_H
