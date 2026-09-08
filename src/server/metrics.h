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
void ember_metrics_record_generation(const char *finish_reason,
                                     int prefill_tokens, int completion_tokens,
                                     double prefill_s, double decode_s,
                                     bool spec_engaged, double accept_rate,
                                     int n_images);

// Prompt-cache reuse for one request: tokens presented and tokens restored.
void ember_metrics_record_prefix_cache(int prompt_tokens, int restored_tokens);

// Seconds a request waited before generation began. Ember serialises
// generation, so this is the dominant latency term rather than a footnote.
void ember_metrics_record_queue_wait(double seconds);

// Render the Prometheus text exposition format into `out`.
void ember_metrics_render(ember_buf *out);

#endif  // EMBER_SERVER_METRICS_H
