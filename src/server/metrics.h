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
// `queue_s` is the wait before generation began. It is taken here rather than
// left to the separate queue observation because time-to-first-token is
// queue + prefill, and a scraper cannot add two independent histograms. Quoting
// prefill duration as TTFT would understate exactly the delay a caller feels.
void ember_metrics_record_generation(const char *finish_reason,
                                     int prefill_tokens, int completion_tokens,
                                     double queue_s, double prefill_s,
                                     double decode_s,
                                     bool spec_engaged, double accept_rate,
                                     int n_images);

// Why speculation did not run. #13 recorded 833 of 833 generations declining
// for a single reason with nothing counting it, and the engine already computes
// this string -- it was simply never leaving the backend. NULL or "" means
// speculation ran, which is not counted here.
void ember_metrics_record_spec_decline(const char *reason);

// One image encode. `seconds` is the vision tower only, separated from LM
// prefill, because a projector that silently falls back to CPU shows up here
// and nowhere else -- that is how ggml-org/llama.cpp#22582 was diagnosed.
void ember_metrics_record_vision_encode(double seconds, int image_tokens);

// Prompt-cache reuse for one request: tokens presented and tokens restored.
void ember_metrics_record_prefix_cache(int prompt_tokens, int restored_tokens);

// Seconds a request waited before generation began. Ember serialises
// generation, so this is the dominant latency term rather than a footnote.
void ember_metrics_record_queue_wait(double seconds);

// Render the Prometheus text exposition format into `out`.
void ember_metrics_render(ember_buf *out);

#endif  // EMBER_SERVER_METRICS_H
