#include "metrics.h"

#include <pthread.h>
#include <string.h>

// Fixed bucket ladders. Prometheus histograms are cumulative: bucket i counts
// every observation <= bound[i], and +Inf equals the total count.
static const double kSecondsBounds[] = {
    0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0
};
static const double kTokenBounds[] = {
    64, 256, 1024, 4096, 16384, 65536, 131072
};
#define N_SECONDS_BUCKETS (sizeof(kSecondsBounds) / sizeof(kSecondsBounds[0]))
#define N_TOKEN_BUCKETS   (sizeof(kTokenBounds) / sizeof(kTokenBounds[0]))

// One histogram type carrying its own ladder. Two near-identical structs with
// two observe and two render functions differed only by their bounds array,
// which is the duplication this file exists to help find.
#define MAX_BUCKETS 12
typedef struct {
    const double *bounds;
    size_t n;
    unsigned long long counts[MAX_BUCKETS + 1];  // +1 for +Inf
    double sum;
    unsigned long long total;
} histogram;

// A closed label set. An unrecognised reason maps to "other" rather than
// minting a new series: an unbounded label is how a scrape target degrades a
// monitoring system, and ember's reasons are enumerable.
static const char *const kFinishReasons[] = {
    "stop", "length", "tool_calls", "repetition_detected",
    "reasoning_cycle_detected", "prompt_echo_detected", "other"
};
#define N_FINISH_REASONS (sizeof(kFinishReasons) / sizeof(kFinishReasons[0]))

// ponytail: one global lock for every counter. Ceiling: contention once
// generation stops serialising (batch_sessions > 1). Upgrade path: per-counter
// atomics, or a per-thread shard summed at render.
static struct {
    pthread_mutex_t lock;
    unsigned long long generations;
    unsigned long long finish_reason[N_FINISH_REASONS];
    unsigned long long prefill_tokens;
    unsigned long long completion_tokens;
    // Prefix cache
    unsigned long long cache_prompt_tokens;
    unsigned long long cache_restored_tokens;
    unsigned long long cache_requests;
    // Speculative decode
    unsigned long long spec_engaged;
    double spec_accept_sum;
    // Vision: a served image request and no image traffic are otherwise
    // byte-identical in the logs.
    unsigned long long vision_requests;
    unsigned long long vision_images;
    // Distributions
    histogram prefill_seconds;
    histogram decode_seconds;
    histogram queue_seconds;
    histogram prefill_token_shape;
    histogram prompt_token_shape;
    histogram completion_token_shape;
} g = {
    .lock = PTHREAD_MUTEX_INITIALIZER,
    .prefill_seconds       = {.bounds = kSecondsBounds, .n = N_SECONDS_BUCKETS},
    .decode_seconds        = {.bounds = kSecondsBounds, .n = N_SECONDS_BUCKETS},
    .queue_seconds         = {.bounds = kSecondsBounds, .n = N_SECONDS_BUCKETS},
    .prefill_token_shape   = {.bounds = kTokenBounds,   .n = N_TOKEN_BUCKETS},
    .prompt_token_shape    = {.bounds = kTokenBounds,   .n = N_TOKEN_BUCKETS},
    .completion_token_shape= {.bounds = kTokenBounds,   .n = N_TOKEN_BUCKETS},
};

static void observe(histogram *h, double v) {
    if (v < 0.0) return;
    size_t i = 0;
    while (i < h->n && v > h->bounds[i]) i++;
    h->counts[i]++;
    h->sum += v;
    h->total++;
}

static size_t finish_reason_index(const char *reason) {
    if (reason)
        for (size_t i = 0; i + 1 < N_FINISH_REASONS; ++i)
            if (strcmp(reason, kFinishReasons[i]) == 0) return i;
    return N_FINISH_REASONS - 1;  // "other"
}

void ember_metrics_record_generation(const char *finish_reason,
                                     int prefill_tokens, int completion_tokens,
                                     double prefill_s, double decode_s,
                                     bool spec_engaged, double accept_rate,
                                     int n_images) {
    pthread_mutex_lock(&g.lock);
    g.generations++;
    g.finish_reason[finish_reason_index(finish_reason)]++;
    if (prefill_tokens > 0) {
        g.prefill_tokens += (unsigned long long)prefill_tokens;
        observe(&g.prefill_token_shape, prefill_tokens);
    }
    if (completion_tokens > 0) {
        g.completion_tokens += (unsigned long long)completion_tokens;
        observe(&g.completion_token_shape, completion_tokens);
    }
    observe(&g.prefill_seconds, prefill_s);
    observe(&g.decode_seconds, decode_s);
    if (spec_engaged) {
        g.spec_engaged++;
        if (accept_rate > 0.0) g.spec_accept_sum += accept_rate;
    }
    if (n_images > 0) {
        g.vision_requests++;
        g.vision_images += (unsigned long long)n_images;
    }
    pthread_mutex_unlock(&g.lock);
}

void ember_metrics_record_prefix_cache(int prompt_tokens, int restored_tokens) {
    if (prompt_tokens < 0 || restored_tokens < 0) return;
    pthread_mutex_lock(&g.lock);
    g.cache_requests++;
    g.cache_prompt_tokens += (unsigned long long)prompt_tokens;
    // The only site that sees the prompt as presented, before restore.
    if (prompt_tokens > 0) observe(&g.prompt_token_shape, prompt_tokens);
    g.cache_restored_tokens += (unsigned long long)restored_tokens;
    pthread_mutex_unlock(&g.lock);
}

void ember_metrics_record_queue_wait(double seconds) {
    pthread_mutex_lock(&g.lock);
    observe(&g.queue_seconds, seconds);
    pthread_mutex_unlock(&g.lock);
}

static void render_counter(ember_buf *b, const char *name, const char *help,
                           unsigned long long v) {
    ember_buf_printf(b, "# HELP %s %s\n# TYPE %s counter\n%s %llu\n",
                     name, help, name, name, v);
}

static void render_histogram(ember_buf *b, const char *name, const char *help,
                             const histogram *h) {
    ember_buf_printf(b, "# HELP %s %s\n# TYPE %s histogram\n", name, help, name);
    unsigned long long cumulative = 0;
    for (size_t i = 0; i < h->n; ++i) {
        cumulative += h->counts[i];
        ember_buf_printf(b, "%s_bucket{le=\"%g\"} %llu\n",
                         name, h->bounds[i], cumulative);
    }
    cumulative += h->counts[h->n];
    ember_buf_printf(b, "%s_bucket{le=\"+Inf\"} %llu\n", name, cumulative);
    ember_buf_printf(b, "%s_sum %.6f\n%s_count %llu\n", name, h->sum, name,
                     h->total);
}

void ember_metrics_render(ember_buf *out) {
    if (!out) return;
    pthread_mutex_lock(&g.lock);

    render_counter(out, "ember_generations_total",
                   "Completed generations.", g.generations);

    ember_buf_puts(out,
        "# HELP ember_generations_by_finish_reason_total Generations by termination reason.\n"
        "# TYPE ember_generations_by_finish_reason_total counter\n");
    for (size_t i = 0; i < N_FINISH_REASONS; ++i)
        ember_buf_printf(out,
            "ember_generations_by_finish_reason_total{reason=\"%s\"} %llu\n",
            kFinishReasons[i], g.finish_reason[i]);

    // Named for what the backend actually reports: tokens evaluated AFTER
    // prefix restore, not the size of the prompt the client sent. A restored
    // prefix makes those differ by design, which is the whole point of the
    // prefix-cache counters below.
    render_counter(out, "ember_prefill_tokens_total",
                   "Tokens evaluated after prefix restore.", g.prefill_tokens);
    render_counter(out, "ember_completion_tokens_total",
                   "Tokens generated.", g.completion_tokens);

    render_counter(out, "ember_prefix_cache_requests_total",
                   "Requests that consulted the prompt cache.",
                   g.cache_requests);
    render_counter(out, "ember_prefix_cache_prompt_tokens_total",
                   "Prompt tokens presented to the prompt cache.",
                   g.cache_prompt_tokens);
    render_counter(out, "ember_prefix_cache_restored_tokens_total",
                   "Prompt tokens served from the prompt cache. Divide by "
                   "ember_prefix_cache_prompt_tokens_total for hit rate.",
                   g.cache_restored_tokens);

    // No separate "eligible" series: it was incremented on every generation,
    // so it claimed an eligibility this API cannot determine and duplicated a
    // denominator ember_generations_total already provides.
    render_counter(out, "ember_spec_decode_engaged_total",
                   "Generations where speculative decode actually ran. Divide "
                   "by ember_generations_total for the engagement rate.",
                   g.spec_engaged);
    ember_buf_printf(out,
        "# HELP ember_spec_decode_accept_rate_sum Sum of accept rates over "
        "engaged generations. Divide by ember_spec_decode_engaged_total for "
        "the mean.\n# TYPE ember_spec_decode_accept_rate_sum counter\n"
        "ember_spec_decode_accept_rate_sum %.6f\n", g.spec_accept_sum);

    render_counter(out, "ember_vision_requests_total",
                   "Generations that carried at least one image.",
                   g.vision_requests);
    render_counter(out, "ember_vision_images_total",
                   "Images accepted across all generations.", g.vision_images);

    render_histogram(out, "ember_prefill_seconds",
                             "Prefill duration.", &g.prefill_seconds);
    render_histogram(out, "ember_decode_seconds",
                             "Decode duration.", &g.decode_seconds);
    render_histogram(out, "ember_queue_seconds",
                             "Time a request waited before generation began.",
                             &g.queue_seconds);
    render_histogram(out, "ember_request_prefill_tokens",
                           "Distribution of tokens evaluated after prefix "
                           "restore.", &g.prefill_token_shape);
    render_histogram(out, "ember_request_prompt_tokens",
                           "Distribution of prompt size as presented by the "
                           "client, before prefix restore.",
                           &g.prompt_token_shape);
    render_histogram(out, "ember_request_completion_tokens",
                           "Completion size distribution.",
                           &g.completion_token_shape);

    pthread_mutex_unlock(&g.lock);
}
