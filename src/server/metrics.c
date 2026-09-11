// strcasestr() is a GNU extension. The container build passes -D_GNU_SOURCE;
// the CMake stub build does not (same note as http.c).
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "metrics.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <unistd.h>

#include <dirent.h>
#include <sys/time.h>

// Fixed bucket ladders. Prometheus histograms are cumulative: bucket i counts
// every observation <= bound[i], and +Inf equals the total count.
static const double kSecondsBounds[] = {
    0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0
};
static const double kTokenBounds[] = {
    64, 256, 1024, 4096, 16384, 65536, 131072
};
// Individual token gaps sit at tens of milliseconds; the request ladder's
// 50 ms first bucket would swallow every one of them.
static const double kGapBounds[] = {
    0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.25, 0.5, 1.0, 2.5, 5.0
};
#define N_SECONDS_BUCKETS (sizeof(kSecondsBounds) / sizeof(kSecondsBounds[0]))
#define N_TOKEN_BUCKETS   (sizeof(kTokenBounds) / sizeof(kTokenBounds[0]))
#define N_GAP_BUCKETS     (sizeof(kGapBounds) / sizeof(kGapBounds[0]))

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
// The engine's own decline taxonomy (deepseek4_backend.cpp). Closed for the
// same reason as finish reasons: an unbounded label is how a scrape target
// degrades a monitoring system. A new engine reason must be added here
// deliberately, and lands in "other" until it is.
static const char *const kSpecDeclineReasons[] = {
    "disabled", "no_drafter", "context", "force_ar", "token_mask",
    "sampling", "empty_budget", "short_budget", "profitability_gate",
    // Image turns short-circuit the gate entirely, so they never reach the
    // reasons above. #9 asks specifically why they forgo speculation, and
    // folding them into force_ar would answer a different question.
    "vision",
    // Resident batching decides eligibility in session state rather than
    // through the serial gate, so its declines carry their own reasons.
    "resident_provider", "resident_submit_failed", "resident_shadow_capture",
    "other",
};
#define N_SPEC_DECLINE_REASONS \
    (sizeof(kSpecDeclineReasons) / sizeof(kSpecDeclineReasons[0]))

static const char *const kFinishReasons[] = {
    "stop", "length", "tool_calls", "repetition_detected",
    "reasoning_cycle_detected", "prompt_echo_detected", "other"
};
#define N_FINISH_REASONS (sizeof(kFinishReasons) / sizeof(kFinishReasons[0]))

// Every status respond() and the SSE header path can emit. A code outside the
// list lands in "other" -- the same closure rule as the reasons above.
static const int kStatusCodes[] = {200, 204, 400, 404, 409, 422, 429, 500, 503};
#define N_STATUS_CODES (sizeof(kStatusCodes) / sizeof(kStatusCodes[0]))
#define N_STATUS_LABELS (N_STATUS_CODES + 1)   // + "other"

static const char *const kOutcomes[] = {
    "ok", "client_disconnected", "cancelled", "backend_error", "stalled",
    "other"
};
#define N_OUTCOMES (sizeof(kOutcomes) / sizeof(kOutcomes[0]))

// Protocol names as the adapters spell them. "other" absorbs a new adapter
// until it is added here deliberately.
static const char *const kApis[] = {
    "chat", "responses", "anthropic", "completions", "other"
};
#define N_APIS (sizeof(kApis) / sizeof(kApis[0]))

// Client families and the User-Agent fragments that select them. Order is
// precedence: an SDK's UA usually also names its language runtime, so the
// product fragments come before "python"/"node".
static const struct { const char *label; const char *needle; } kClients[] = {
    {"hermes",    "hermes"},
    {"openai",    "openai"},
    {"anthropic", "anthropic"},
    {"curl",      "curl"},
    {"python",    "python"},
    {"node",      "node"},
    {"browser",   "mozilla"},
};
#define N_CLIENT_NEEDLES (sizeof(kClients) / sizeof(kClients[0]))
static const char *const kClientLabels[] = {
    "hermes", "openai", "anthropic", "curl", "python", "node", "browser",
    "none", "other"
};
#define N_CLIENTS (sizeof(kClientLabels) / sizeof(kClientLabels[0]))

#ifndef EMBER_VERSION_STRING
#define EMBER_VERSION_STRING "dev"
#endif
#ifndef EMBER_GIT_REVISION
#define EMBER_GIT_REVISION ""
#endif

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
    unsigned long long spec_declined[N_SPEC_DECLINE_REASONS];
    // Vision: a served image request and no image traffic are otherwise
    // byte-identical in the logs.
    unsigned long long vision_requests;
    unsigned long long vision_images;
    unsigned long long image_tokens;
    // Distributions
    histogram prefill_seconds;
    histogram decode_seconds;
    histogram queue_seconds;
    histogram ttft_seconds;
    histogram mean_token_gap_seconds;
    histogram vision_encode_seconds;
    histogram image_count;
    histogram prefill_token_shape;
    histogram prompt_token_shape;
    histogram completion_token_shape;
    // Lifecycle
    long long jobs_waiting;
    long long jobs_running;
    unsigned long long shed;
    unsigned long long responses[N_STATUS_LABELS];
    unsigned long long outcomes[N_OUTCOMES];
    unsigned long long requests[N_APIS][N_CLIENTS];
    histogram request_seconds;
    histogram token_gap_seconds;
    // Speculative decode detail
    unsigned long long spec_cycles;
    double spec_head_s;
    double spec_verify_s;
    double spec_provider_block_s;
    // KV snapshots
    unsigned long long snapshots_requested;
    unsigned long long snapshots_saved;
    // Wall-clock start, captured on first use so /metrics can report it
    // without main() having to remember to.
    double start_time_s;
} g = {
    .lock = PTHREAD_MUTEX_INITIALIZER,
    .prefill_seconds       = {.bounds = kSecondsBounds, .n = N_SECONDS_BUCKETS},
    .decode_seconds        = {.bounds = kSecondsBounds, .n = N_SECONDS_BUCKETS},
    .queue_seconds         = {.bounds = kSecondsBounds, .n = N_SECONDS_BUCKETS},
    .ttft_seconds          = {.bounds = kSecondsBounds, .n = N_SECONDS_BUCKETS},
    .mean_token_gap_seconds= {.bounds = kSecondsBounds, .n = N_SECONDS_BUCKETS},
    .vision_encode_seconds = {.bounds = kSecondsBounds, .n = N_SECONDS_BUCKETS},
    .image_count           = {.bounds = kTokenBounds,   .n = N_TOKEN_BUCKETS},
    .prefill_token_shape   = {.bounds = kTokenBounds,   .n = N_TOKEN_BUCKETS},
    .prompt_token_shape    = {.bounds = kTokenBounds,   .n = N_TOKEN_BUCKETS},
    .completion_token_shape= {.bounds = kTokenBounds,   .n = N_TOKEN_BUCKETS},
    .request_seconds       = {.bounds = kSecondsBounds, .n = N_SECONDS_BUCKETS},
    .token_gap_seconds     = {.bounds = kGapBounds,     .n = N_GAP_BUCKETS},
};

static double wall_now(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1e6;
}

// Called under g.lock. The start time is the first moment anything touched the
// registry, which is process start to within the model-load window and, more
// to the point, resets exactly when the counters do.
static void note_started(void) {
    if (g.start_time_s <= 0.0) g.start_time_s = wall_now();
}

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

static size_t spec_decline_index(const char *reason) {
    if (reason) {
        for (size_t i = 0; i < N_SPEC_DECLINE_REASONS; ++i) {
            if (strcmp(reason, kSpecDeclineReasons[i]) == 0) return i;
        }
    }
    return N_SPEC_DECLINE_REASONS - 1;  // "other"
}

void ember_metrics_record_spec_decline(const char *reason) {
    if (!reason || !reason[0]) return;   // speculation ran; nothing declined
    pthread_mutex_lock(&g.lock);
    g.spec_declined[spec_decline_index(reason)]++;
    pthread_mutex_unlock(&g.lock);
}

void ember_metrics_record_vision_encode(double seconds, int image_tokens) {
    pthread_mutex_lock(&g.lock);
    observe(&g.vision_encode_seconds, seconds);
    if (image_tokens > 0) g.image_tokens += (unsigned long long)image_tokens;
    pthread_mutex_unlock(&g.lock);
}

void ember_metrics_record_generation(const char *finish_reason,
                                     int prefill_tokens, int completion_tokens,
                                     double ttft_s, double mean_token_gap_s,
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
    // Both are measured by the caller from token arrivals. A negative value
    // means the request had no such measurement -- no token at all, or only
    // one -- and is skipped rather than recorded as zero.
    if (ttft_s >= 0.0) observe(&g.ttft_seconds, ttft_s);
    if (mean_token_gap_s >= 0.0) observe(&g.mean_token_gap_seconds, mean_token_gap_s);
    if (spec_engaged) {
        g.spec_engaged++;
        if (accept_rate > 0.0) g.spec_accept_sum += accept_rate;
    }
    if (n_images > 0) {
        g.vision_requests++;
        g.vision_images += (unsigned long long)n_images;
        observe(&g.image_count, n_images);
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

static size_t closed_index(const char *const *labels, size_t n,
                           const char *value) {
    if (value)
        for (size_t i = 0; i + 1 < n; ++i)
            if (strcmp(value, labels[i]) == 0) return i;
    return n - 1;  // "other"
}

void ember_metrics_job_enqueued(void) {
    pthread_mutex_lock(&g.lock);
    note_started();
    g.jobs_waiting++;
    pthread_mutex_unlock(&g.lock);
}

void ember_metrics_job_started(void) {
    pthread_mutex_lock(&g.lock);
    if (g.jobs_waiting > 0) g.jobs_waiting--;
    g.jobs_running++;
    pthread_mutex_unlock(&g.lock);
}

void ember_metrics_job_finished(double end_to_end_s) {
    pthread_mutex_lock(&g.lock);
    if (g.jobs_running > 0) g.jobs_running--;
    observe(&g.request_seconds, end_to_end_s);
    pthread_mutex_unlock(&g.lock);
}

void ember_metrics_record_shed(void) {
    pthread_mutex_lock(&g.lock);
    g.shed++;
    pthread_mutex_unlock(&g.lock);
}

void ember_metrics_record_response(int status) {
    size_t idx = N_STATUS_CODES;  // "other"
    for (size_t i = 0; i < N_STATUS_CODES; ++i)
        if (kStatusCodes[i] == status) { idx = i; break; }
    pthread_mutex_lock(&g.lock);
    note_started();
    g.responses[idx]++;
    pthread_mutex_unlock(&g.lock);
}

void ember_metrics_record_outcome(const char *outcome) {
    pthread_mutex_lock(&g.lock);
    g.outcomes[closed_index(kOutcomes, N_OUTCOMES, outcome)]++;
    pthread_mutex_unlock(&g.lock);
}

void ember_metrics_record_token_gap(double seconds) {
    pthread_mutex_lock(&g.lock);
    observe(&g.token_gap_seconds, seconds);
    pthread_mutex_unlock(&g.lock);
}

void ember_metrics_record_spec_detail(int cycles, double head_s,
                                      double verify_s, double provider_block_s) {
    if (cycles <= 0) return;
    pthread_mutex_lock(&g.lock);
    g.spec_cycles += (unsigned long long)cycles;
    if (head_s > 0.0) g.spec_head_s += head_s;
    if (verify_s > 0.0) g.spec_verify_s += verify_s;
    if (provider_block_s > 0.0) g.spec_provider_block_s += provider_block_s;
    pthread_mutex_unlock(&g.lock);
}

void ember_metrics_record_snapshot(bool requested, bool saved) {
    if (!requested) return;
    pthread_mutex_lock(&g.lock);
    g.snapshots_requested++;
    if (saved) g.snapshots_saved++;
    pthread_mutex_unlock(&g.lock);
}

const char *ember_metrics_client_family(const char *user_agent) {
    if (!user_agent || !user_agent[0]) return "none";
    for (size_t i = 0; i < N_CLIENT_NEEDLES; ++i)
        if (strcasestr(user_agent, kClients[i].needle)) return kClients[i].label;
    return "other";
}

void ember_metrics_record_request(const char *api, const char *client) {
    const size_t a = closed_index(kApis, N_APIS, api);
    const size_t c = closed_index(kClientLabels, N_CLIENTS, client);
    pthread_mutex_lock(&g.lock);
    note_started();
    g.requests[a][c]++;
    pthread_mutex_unlock(&g.lock);
}

// ── process metrics ─────────────────────────────────────────────────────
// The C server links no client library, so the conventional process_* family
// is read straight from procfs at scrape time. Every value is optional: an
// unreadable file drops its samples instead of rendering a zero that would be
// mistaken for a measurement.
static const char *proc_dir(void) {
    const char *d = getenv("EMBER_METRICS_PROC_DIR");
    return d && d[0] ? d : "/proc";
}

static void render_process_metrics(ember_buf *b) {
    char path[512];
    snprintf(path, sizeof(path), "%s/self/stat", proc_dir());
    FILE *f = fopen(path, "r");
    if (f) {
        char line[4096];
        if (fgets(line, sizeof(line), f)) {
            // Field 2 (comm) may contain spaces; everything after its closing
            // paren is the fixed layout proc(5) documents.
            const char *p = strrchr(line, ')');
            unsigned long long utime = 0, stime = 0, vsize = 0;
            long long rss_pages = 0, starttime = 0;
            long num_threads = 0;
            // Fields 3.. after the paren: state ppid pgrp session tty tpgid
            // flags minflt cminflt majflt cmajflt utime stime cutime cstime
            // priority nice num_threads itrealvalue starttime vsize rss
            if (p && sscanf(p + 2,
                    "%*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u "
                    "%llu %llu %*d %*d %*d %*d %ld %*d %lld %llu %lld",
                    &utime, &stime, &num_threads, &starttime, &vsize,
                    &rss_pages) == 6) {
                const double hz = (double)sysconf(_SC_CLK_TCK);
                const double page = (double)sysconf(_SC_PAGESIZE);
                ember_buf_printf(b,
                    "# HELP process_cpu_seconds_total Total user and system "
                    "CPU time spent in seconds.\n"
                    "# TYPE process_cpu_seconds_total counter\n"
                    "process_cpu_seconds_total %.3f\n",
                    (double)(utime + stime) / hz);
                ember_buf_printf(b,
                    "# HELP process_resident_memory_bytes Resident memory "
                    "size in bytes.\n"
                    "# TYPE process_resident_memory_bytes gauge\n"
                    "process_resident_memory_bytes %.0f\n",
                    (double)rss_pages * page);
                ember_buf_printf(b,
                    "# HELP process_virtual_memory_bytes Virtual memory size "
                    "in bytes.\n"
                    "# TYPE process_virtual_memory_bytes gauge\n"
                    "process_virtual_memory_bytes %llu\n", vsize);
                ember_buf_printf(b,
                    "# HELP process_threads Number of OS threads in the "
                    "process.\n# TYPE process_threads gauge\n"
                    "process_threads %ld\n", num_threads);
                (void)starttime;  // relative to boot; the wall clock below is
                                  // what Prometheus convention expects
            }
        }
        fclose(f);
    }
    snprintf(path, sizeof(path), "%s/self/fd", proc_dir());
    DIR *d = opendir(path);
    if (d) {
        unsigned long n = 0;
        struct dirent *e;
        while ((e = readdir(d)) != NULL)
            if (e->d_name[0] != '.') n++;
        closedir(d);
        ember_buf_printf(b,
            "# HELP process_open_fds Number of open file descriptors.\n"
            "# TYPE process_open_fds gauge\nprocess_open_fds %lu\n", n);
    }
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
    // #13: 833 of 833 generations declined for ONE reason and no signal said
    // so. The engine computed this string already and only printed it behind a
    // debug env; this is that value, counted.
    ember_buf_puts(out,
        "# HELP ember_spec_decode_declined_total Generations where speculation did not run, by reason.\n"
        "# TYPE ember_spec_decode_declined_total counter\n");
    for (size_t i = 0; i < N_SPEC_DECLINE_REASONS; ++i)
        ember_buf_printf(out,
            "ember_spec_decode_declined_total{reason=\"%s\"} %llu\n",
            kSpecDeclineReasons[i], g.spec_declined[i]);

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
    render_counter(out, "ember_image_tokens_total",
                   "Prompt tokens contributed by encoded images.",
                   g.image_tokens);
    render_counter(out, "ember_vision_images_total",
                   "Images accepted across all generations.", g.vision_images);

    render_histogram(out, "ember_prefill_seconds",
                             "Prefill duration.", &g.prefill_seconds);
    render_histogram(out, "ember_decode_seconds",
                             "Decode duration.", &g.decode_seconds);
    render_histogram(out, "ember_queue_seconds",
                             "HTTP enqueue to dispatcher admission, including "
                             "serial lock wait; excludes later resident "
                             "engine admission and prompt preparation.",
                             &g.queue_seconds);
    // Deliberately distinct from ember_prefill_seconds: TTFT includes the
    // queue wait, and on a serialising server that term dominates.
    render_histogram(out, "ember_time_to_first_token_seconds",
                             "Measured first-token arrival minus request "
                             "enqueue: includes FIFO wait, image encoding and "
                             "prompt preparation, not only backend prefill. "
                             "Includes hidden recovery token callbacks; not "
                             "time to first visible content. "
                             "Requests producing no token are excluded.",
                             &g.ttft_seconds);
    // Named a mean, because it is one. Dividing a span by a count is not a
    // distribution of individual gaps, and calling it inter-token latency would
    // invite a p99 read off a series that cannot express one.
    render_histogram(out, "ember_request_mean_token_gap_seconds",
                             "Per-request mean gap between generated tokens, "
                             "(last - first) / (tokens - 1). A distribution of "
                             "per-request means, NOT of individual gaps. "
                             "Includes hidden recovery attempts and the gaps "
                             "between them. "
                             "Requests with fewer than two tokens are excluded.",
                             &g.mean_token_gap_seconds);
    render_histogram(out, "ember_vision_encoder_seconds",
                             "End-to-end image encode as the server sees it: "
                             "preprocessing, lazy first-use tower load, "
                             "execution and result copy. Separated from LM "
                             "prefill, but NOT tower execution alone.",
                             &g.vision_encode_seconds);
    render_histogram(out, "ember_request_image_count",
                             "Images per IMAGE-BEARING generation. Text-only "
                             "requests are not observed here at all, so this is "
                             "not a per-request distribution over all traffic.",
                             &g.image_count);
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

    // ── lifecycle ──
    ember_buf_printf(out,
        "# HELP ember_jobs Generation jobs on the worker FIFO right now, by "
        "state.\n# TYPE ember_jobs gauge\n"
        "ember_jobs{state=\"waiting\"} %lld\n"
        "ember_jobs{state=\"running\"} %lld\n",
        g.jobs_waiting, g.jobs_running);
    render_counter(out, "ember_requests_shed_total",
                   "Generation requests refused with 503 because the worker "
                   "FIFO was full.", g.shed);
    ember_buf_puts(out,
        "# HELP ember_http_responses_total HTTP responses written, by status "
        "code. Streaming responses count when their headers are sent.\n"
        "# TYPE ember_http_responses_total counter\n");
    for (size_t i = 0; i < N_STATUS_CODES; ++i)
        ember_buf_printf(out, "ember_http_responses_total{status=\"%d\"} %llu\n",
                         kStatusCodes[i], g.responses[i]);
    ember_buf_printf(out, "ember_http_responses_total{status=\"other\"} %llu\n",
                     g.responses[N_STATUS_CODES]);
    ember_buf_puts(out,
        "# HELP ember_generation_outcomes_total How generations ended as the "
        "server saw them; a finish reason cannot express a disconnect or a "
        "backend failure.\n"
        "# TYPE ember_generation_outcomes_total counter\n");
    for (size_t i = 0; i < N_OUTCOMES; ++i)
        ember_buf_printf(out,
            "ember_generation_outcomes_total{outcome=\"%s\"} %llu\n",
            kOutcomes[i], g.outcomes[i]);
    ember_buf_puts(out,
        "# HELP ember_requests_total Accepted generation requests by protocol "
        "and client family (User-Agent mapped onto a closed set).\n"
        "# TYPE ember_requests_total counter\n");
    for (size_t a = 0; a < N_APIS; ++a)
        for (size_t c = 0; c < N_CLIENTS; ++c)
            if (g.requests[a][c])
                ember_buf_printf(out,
                    "ember_requests_total{api=\"%s\",client=\"%s\"} %llu\n",
                    kApis[a], kClientLabels[c], g.requests[a][c]);
    render_histogram(out, "ember_request_seconds",
                     "End-to-end seconds from HTTP enqueue to the response "
                     "being finished: queue wait, prompt preparation, prefill "
                     "and decode together. The number a caller experiences.",
                     &g.request_seconds);
    render_histogram(out, "ember_token_gap_seconds",
                     "Individual gaps between consecutive generated tokens. "
                     "A real distribution, so p99 is meaningful here; compare "
                     "ember_request_mean_token_gap_seconds, which is not.",
                     &g.token_gap_seconds);

    // ── speculative decode detail ──
    render_counter(out, "ember_spec_decode_cycles_total",
                   "Draft/verify cycles run across engaged generations.",
                   g.spec_cycles);
    ember_buf_printf(out,
        "# HELP ember_spec_decode_seconds_total Wall seconds speculation spent "
        "per phase, summed over engaged generations.\n"
        "# TYPE ember_spec_decode_seconds_total counter\n"
        "ember_spec_decode_seconds_total{phase=\"head\"} %.6f\n"
        "ember_spec_decode_seconds_total{phase=\"verify\"} %.6f\n"
        "ember_spec_decode_seconds_total{phase=\"provider_block\"} %.6f\n",
        g.spec_head_s, g.spec_verify_s, g.spec_provider_block_s);

    // ── KV snapshots ──
    render_counter(out, "ember_kv_snapshots_requested_total",
                   "Generations that reserved a KV snapshot slot.",
                   g.snapshots_requested);
    render_counter(out, "ember_kv_snapshots_saved_total",
                   "Generations whose backend actually persisted the snapshot. "
                   "Requested minus saved is the silent prefix-cache miss "
                   "source.", g.snapshots_saved);

    // ── identity ──
    ember_buf_printf(out,
        "# HELP ember_build_info Build identity; the value is always 1.\n"
        "# TYPE ember_build_info gauge\n"
        "ember_build_info{version=\"%s\",revision=\"%s\"} 1\n",
        EMBER_VERSION_STRING, EMBER_GIT_REVISION);
    if (g.start_time_s > 0.0)
        ember_buf_printf(out,
            "# HELP process_start_time_seconds Unix time the metrics registry "
            "first recorded anything; resets with the counters.\n"
            "# TYPE process_start_time_seconds gauge\n"
            "process_start_time_seconds %.3f\n", g.start_time_s);
    render_process_metrics(out);

    pthread_mutex_unlock(&g.lock);
}
