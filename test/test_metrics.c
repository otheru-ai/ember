#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../src/server/metrics.h"

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, msg)                                                    \
    do { if (cond) g_pass++; else { g_fail++; printf("  FAIL: %s\n", msg); } } while (0)

// Extract the integer value of a single-line `name value` sample.
static long long sample(const char *text, const char *name) {
    char needle[256];
    snprintf(needle, sizeof(needle), "\n%s ", name);
    const char *p = strstr(text, needle);
    if (!p) return -1;
    return atoll(p + strlen(needle));
}

static void test_shape_and_counters(void) {
    ember_metrics_record_generation("stop", 100, 20, 0.35, 0.06, 0.30, 1.20, false, 0.0, 0);
    ember_metrics_record_generation("length", 5000, 400, 3.25, 0.05, 3.00, 20.0, true, 0.9, 2);
    ember_metrics_record_prefix_cache(1000, 250);
    ember_metrics_record_queue_wait(0.75);

    ember_buf b = {0};
    ember_metrics_render(&b);
    const char *t = b.ptr ? b.ptr : "";

    CHECK(strstr(t, "# HELP ember_generations_total") != NULL, "HELP line present");
    CHECK(strstr(t, "# TYPE ember_generations_total counter") != NULL, "TYPE line present");
    CHECK(sample(t, "ember_generations_total") == 2, "generations counted");
    CHECK(sample(t, "ember_prefill_tokens_total") == 5100, "prefill tokens summed");
    CHECK(sample(t, "ember_completion_tokens_total") == 420, "completion tokens summed");

    // The counters the soak actually needed.
    CHECK(sample(t, "ember_prefix_cache_prompt_tokens_total") == 1000, "cache prompt tokens");
    CHECK(sample(t, "ember_prefix_cache_restored_tokens_total") == 250, "cache restored tokens");
    CHECK(sample(t, "ember_spec_decode_engaged_total") == 1, "spec engaged counted separately");

    // Vision: a served image request must be distinguishable from none.
    CHECK(sample(t, "ember_vision_requests_total") == 1, "vision requests counted");
    CHECK(sample(t, "ember_vision_images_total") == 2, "vision images counted");

    CHECK(strstr(t, "ember_generations_by_finish_reason_total{reason=\"stop\"} 1") != NULL,
          "finish reason stop labelled");
    CHECK(strstr(t, "ember_generations_by_finish_reason_total{reason=\"length\"} 1") != NULL,
          "finish reason length labelled");

    // Histograms must be cumulative and terminate at +Inf == count.
    CHECK(strstr(t, "ember_queue_seconds_bucket{le=\"+Inf\"} 1") != NULL,
          "queue histogram +Inf equals count");
    CHECK(strstr(t, "ember_queue_seconds_count 1") != NULL, "queue histogram count");
    CHECK(strstr(t, "ember_request_prefill_tokens_bucket{le=\"+Inf\"} 2") != NULL,
          "prefill token histogram +Inf equals count");
    // Prompt size and prefill work are DIFFERENT numbers whenever a prefix is
    // restored. Only the prefix-cache site sees the prompt as presented, so
    // exactly one observation (1000) reached the prompt histogram while two
    // generations reached the prefill histogram.
    CHECK(strstr(t, "ember_request_prompt_tokens_bucket{le=\"+Inf\"} 1") != NULL,
          "prompt histogram counts the presented prompt, not the prefill");
    CHECK(strstr(t, "ember_request_prompt_tokens_sum 1000.000000") != NULL,
          "prompt histogram sums the presented prompt size");
    CHECK(strstr(t, "ember_spec_decode_eligible_total") == NULL,
          "no eligible series: it claimed an eligibility the API cannot determine");
    ember_buf_free(&b);
}

static void test_spec_decline_reasons(void) {
    // The metrics state is process-global and earlier tests have already
    // written to it, so every assertion here is a DELTA. Absolute counts would
    // make this test depend on the order the suite happens to run in.
    ember_buf before = {0};
    ember_metrics_render(&before);
    const char *b0 = before.ptr ? before.ptr : "";
    const long long ctx0   = sample(b0, "ember_spec_decode_declined_total{reason=\"context\"}");
    const long long ar0    = sample(b0, "ember_spec_decode_declined_total{reason=\"force_ar\"}");
    const long long other0 = sample(b0, "ember_spec_decode_declined_total{reason=\"other\"}");
    const long long vis0   = sample(b0, "ember_spec_decode_declined_total{reason=\"vision\"}");

    ember_metrics_record_spec_decline("context");
    ember_metrics_record_spec_decline("context");
    ember_metrics_record_spec_decline("force_ar");
    // Speculation ran: nothing to attribute. Both forms must be ignored rather
    // than folded into "other", which would invent declines that never happened.
    ember_metrics_record_spec_decline(NULL);
    ember_metrics_record_spec_decline("");
    // An engine reason this build does not know must not mint a series.
    // Image turns short-circuit the gate and are attributed separately; #9
    // asks specifically why they forgo speculation.
    ember_metrics_record_spec_decline("vision");
    ember_metrics_record_spec_decline("some_future_gate");

    ember_buf b = {0};
    ember_metrics_render(&b);
    const char *t = b.ptr ? b.ptr : "";
    CHECK(sample(t, "ember_spec_decode_declined_total{reason=\"context\"}") - ctx0 == 2,
          "declines counted per reason");
    CHECK(sample(t, "ember_spec_decode_declined_total{reason=\"force_ar\"}") - ar0 == 1,
          "second reason counted separately");
    CHECK(sample(t, "ember_spec_decode_declined_total{reason=\"vision\"}") - vis0 == 1,
          "image turns attributed to vision, not folded into force_ar");
    CHECK(strstr(t, "reason=\"some_future_gate\"") == NULL,
          "an unknown decline reason does not mint a series");
    CHECK(sample(t, "ember_spec_decode_declined_total{reason=\"other\"}") - other0 == 1,
          "unknown reason folds into other, and NULL/empty are not counted");
    ember_buf_free(&before);
    ember_buf_free(&b);
}

static void test_latency_and_vision_series(void) {
    ember_buf before = {0};
    ember_metrics_render(&before);
    const char *b0 = before.ptr ? before.ptr : "";
    const long long ttft0 = sample(b0, "ember_time_to_first_token_seconds_count");
    const long long gap0  = sample(b0, "ember_request_mean_token_gap_seconds_count");
    const long long ven0  = sample(b0, "ember_vision_encoder_seconds_count");
    const long long imt0  = sample(b0, "ember_image_tokens_total");
    const long long imc0  = sample(b0, "ember_request_image_count_count");

    // Measured values, as the server now computes them from token arrivals.
    ember_metrics_record_generation("stop", 10, 21, 1.50, 0.10, 1.0, 2.0, false, 0.0, 1);
    // One token: a TTFT exists, but there is no gap to measure.
    ember_metrics_record_generation("stop", 10, 1, 0.80, -1.0, 0.5, 0.4, false, 0.0, 0);
    // No token at all: neither is measurable. Recording zero would invent a
    // latency that never happened and drag both distributions down.
    ember_metrics_record_generation("stop", 10, 0, -1.0, -1.0, 0.5, 0.0, false, 0.0, 0);
    ember_metrics_record_vision_encode(0.25, 131);

    ember_buf b = {0};
    ember_metrics_render(&b);
    const char *t = b.ptr ? b.ptr : "";
    CHECK(sample(t, "ember_time_to_first_token_seconds_count") - ttft0 == 2,
          "ttft recorded for token-producing requests only, not the empty one");
    CHECK(sample(t, "ember_request_mean_token_gap_seconds_count") - gap0 == 1,
          "token-gap mean needs two tokens; one-token and empty excluded");
    CHECK(sample(t, "ember_vision_encoder_seconds_count") - ven0 == 1,
          "vision encode duration recorded separately from prefill");
    CHECK(sample(t, "ember_image_tokens_total") - imt0 == 131,
          "image tokens counted");
    CHECK(sample(t, "ember_request_image_count_count") - imc0 == 1,
          "images-per-request observed only for image requests");
    ember_buf_free(&before);
    ember_buf_free(&b);
}

static void test_unknown_reason_does_not_grow_cardinality(void) {
    ember_metrics_record_generation("a_brand_new_reason", 1, 1, -1.0, -1.0, 0.0, 0.0, false, 0.0, 0);
    ember_buf b = {0};
    ember_metrics_render(&b);
    const char *t = b.ptr ? b.ptr : "";
    CHECK(strstr(t, "reason=\"a_brand_new_reason\"") == NULL,
          "an unrecognised reason does not mint a new series");
    CHECK(strstr(t, "ember_generations_by_finish_reason_total{reason=\"other\"} 1") != NULL,
          "unrecognised reason folds into other");
    ember_buf_free(&b);
}

static void test_cumulative_buckets_are_monotonic(void) {
    ember_buf b = {0};
    ember_metrics_render(&b);
    const char *t = b.ptr ? b.ptr : "";
    // Walk the queue histogram buckets; cumulative counts must never decrease.
    const char *p = strstr(t, "ember_queue_seconds_bucket");
    long long prev = -1; int checked = 0; bool ok = true;
    while (p) {
        const char *v = strstr(p, "} ");
        if (!v) break;
        long long cur = atoll(v + 2);
        if (cur < prev) ok = false;
        prev = cur; checked++;
        p = strstr(v, "ember_queue_seconds_bucket");
    }
    CHECK(checked > 1 && ok, "queue histogram buckets are cumulative");
    ember_buf_free(&b);
}


// Extract a floating sample; the process metrics are not integers.
static double fsample(const char *text, const char *name) {
    char needle[256];
    snprintf(needle, sizeof(needle), "\n%s ", name);
    const char *p = strstr(text, needle);
    if (!p) return -1.0;
    return atof(p + strlen(needle));
}

static void test_lifecycle_gauges_and_e2e(void) {
    ember_buf before = {0};
    ember_metrics_render(&before);
    const char *b0 = before.ptr ? before.ptr : "";
    const long long shed0 = sample(b0, "ember_requests_shed_total");
    const long long e2e0  = sample(b0, "ember_request_seconds_count");

    ember_metrics_job_enqueued();
    ember_metrics_job_enqueued();
    ember_metrics_job_started();
    ember_buf mid = {0};
    ember_metrics_render(&mid);
    const char *tm = mid.ptr ? mid.ptr : "";
    CHECK(sample(tm, "ember_jobs{state=\"waiting\"}") == 1, "one job still waiting");
    CHECK(sample(tm, "ember_jobs{state=\"running\"}") == 1, "one job running");

    ember_metrics_job_finished(12.5);
    ember_metrics_job_started();
    ember_metrics_job_finished(0.2);
    // A finish or start with nothing outstanding must clamp at zero rather
    // than render a negative gauge: the counters bracket real jobs, and a
    // bookkeeping slip must not read as a phantom.
    ember_metrics_job_finished(0.1);
    ember_metrics_job_started();
    ember_metrics_job_finished(0.1);
    ember_metrics_record_shed();

    ember_buf b = {0};
    ember_metrics_render(&b);
    const char *t = b.ptr ? b.ptr : "";
    CHECK(sample(t, "ember_jobs{state=\"waiting\"}") == 0, "waiting drained");
    CHECK(sample(t, "ember_jobs{state=\"running\"}") == 0, "running drained, never negative");
    CHECK(sample(t, "ember_requests_shed_total") - shed0 == 1, "shed counted");
    CHECK(sample(t, "ember_request_seconds_count") - e2e0 == 4, "end-to-end observed per finish");
    CHECK(strstr(t, "ember_request_seconds_bucket{le=\"+Inf\"}") != NULL,
          "end-to-end histogram terminates at +Inf");
    ember_buf_free(&before);
    ember_buf_free(&mid);
    ember_buf_free(&b);
}

static void test_responses_outcomes_and_requests(void) {
    ember_buf before = {0};
    ember_metrics_render(&before);
    const char *b0 = before.ptr ? before.ptr : "";
    const long long ok0    = sample(b0, "ember_http_responses_total{status=\"200\"}");
    const long long s503_0 = sample(b0, "ember_http_responses_total{status=\"503\"}");
    const long long oth0   = sample(b0, "ember_http_responses_total{status=\"other\"}");
    const long long disc0  = sample(b0, "ember_generation_outcomes_total{outcome=\"client_disconnected\"}");
    const long long ooth0  = sample(b0, "ember_generation_outcomes_total{outcome=\"other\"}");

    ember_metrics_record_response(200);
    ember_metrics_record_response(200);
    ember_metrics_record_response(503);
    ember_metrics_record_response(418);   // not in the closed set
    ember_metrics_record_outcome("client_disconnected");
    ember_metrics_record_outcome("not_an_outcome");
    ember_metrics_record_outcome(NULL);

    ember_metrics_record_request("chat", ember_metrics_client_family("python-requests/2.32"));
    ember_metrics_record_request("chat", ember_metrics_client_family("OpenAI/Python 1.40 python"));
    ember_metrics_record_request("anthropic", ember_metrics_client_family(NULL));
    ember_metrics_record_request("responses", ember_metrics_client_family("Hermes-Agent/0.9"));
    ember_metrics_record_request("something_new", ember_metrics_client_family("Mozilla/5.0"));
    ember_metrics_record_request("completions", ember_metrics_client_family("wget/1.21"));

    ember_buf b = {0};
    ember_metrics_render(&b);
    const char *t = b.ptr ? b.ptr : "";
    CHECK(sample(t, "ember_http_responses_total{status=\"200\"}") - ok0 == 2, "200s counted");
    CHECK(sample(t, "ember_http_responses_total{status=\"503\"}") - s503_0 == 1, "503 counted");
    CHECK(sample(t, "ember_http_responses_total{status=\"other\"}") - oth0 == 1,
          "a status outside the closed set folds into other");
    CHECK(strstr(t, "status=\"418\"") == NULL, "no series minted for a novel status");
    CHECK(sample(t, "ember_generation_outcomes_total{outcome=\"client_disconnected\"}") - disc0 == 1,
          "disconnect outcome counted");
    CHECK(sample(t, "ember_generation_outcomes_total{outcome=\"other\"}") - ooth0 == 2,
          "unknown and NULL outcomes fold into other");

    // Client family precedence: an SDK UA names its runtime too, and the
    // product must win; the browser fragment and absence both have labels.
    CHECK(strcmp(ember_metrics_client_family("OpenAI/Python 1.40 python"), "openai") == 0,
          "openai before python");
    CHECK(strcmp(ember_metrics_client_family("anthropic-sdk-python/0.3"), "anthropic") == 0,
          "anthropic before python");
    CHECK(strcmp(ember_metrics_client_family("curl/8.5"), "curl") == 0, "curl");
    CHECK(strcmp(ember_metrics_client_family("node-fetch/3"), "node") == 0, "node");
    CHECK(strcmp(ember_metrics_client_family(""), "none") == 0, "empty UA is none");
    CHECK(strstr(t, "ember_requests_total{api=\"chat\",client=\"python\"} 1") != NULL,
          "chat/python attributed");
    CHECK(strstr(t, "ember_requests_total{api=\"chat\",client=\"openai\"} 1") != NULL,
          "chat/openai attributed");
    CHECK(strstr(t, "ember_requests_total{api=\"anthropic\",client=\"none\"} 1") != NULL,
          "missing UA attributed as none");
    CHECK(strstr(t, "ember_requests_total{api=\"responses\",client=\"hermes\"} 1") != NULL,
          "hermes attributed case-insensitively");
    CHECK(strstr(t, "ember_requests_total{api=\"other\",client=\"browser\"} 1") != NULL,
          "unknown api folds into other; browser UA labelled");
    CHECK(strstr(t, "ember_requests_total{api=\"completions\",client=\"other\"} 1") != NULL,
          "unknown UA folds into other");
    CHECK(strstr(t, "api=\"something_new\"") == NULL, "no api series minted");
    ember_buf_free(&before);
    ember_buf_free(&b);
}

static void test_token_gaps_spec_detail_snapshots(void) {
    ember_buf before = {0};
    ember_metrics_render(&before);
    const char *b0 = before.ptr ? before.ptr : "";
    const long long gap0   = sample(b0, "ember_token_gap_seconds_count");
    const long long gap20  = sample(b0, "ember_token_gap_seconds_bucket{le=\"0.02\"}");
    const long long cyc0   = sample(b0, "ember_spec_decode_cycles_total");
    const double    head0  = fsample(b0, "ember_spec_decode_seconds_total{phase=\"head\"}");
    const long long sreq0  = sample(b0, "ember_kv_snapshots_requested_total");
    const long long ssav0  = sample(b0, "ember_kv_snapshots_saved_total");

    ember_metrics_record_token_gap(0.015);
    ember_metrics_record_token_gap(0.065);
    ember_metrics_record_token_gap(-1.0);   // clock skew: skipped, not zero
    ember_metrics_record_spec_detail(3, 0.5, 1.25, 0.0);
    ember_metrics_record_spec_detail(0, 9.0, 9.0, 9.0);   // no cycles: nothing
    ember_metrics_record_snapshot(true, true);
    ember_metrics_record_snapshot(true, false);
    ember_metrics_record_snapshot(false, true);   // no slot: not a request

    ember_buf b = {0};
    ember_metrics_render(&b);
    const char *t = b.ptr ? b.ptr : "";
    CHECK(sample(t, "ember_token_gap_seconds_count") - gap0 == 2, "two real gaps observed");
    CHECK(sample(t, "ember_token_gap_seconds_bucket{le=\"0.02\"}") - gap20 == 1,
          "gap ladder resolves tens of milliseconds");
    CHECK(sample(t, "ember_spec_decode_cycles_total") - cyc0 == 3, "cycles summed");
    CHECK(fsample(t, "ember_spec_decode_seconds_total{phase=\"head\"}") - head0 > 0.49,
          "head seconds summed only for engaged generations");
    CHECK(sample(t, "ember_kv_snapshots_requested_total") - sreq0 == 2, "snapshot requests");
    CHECK(sample(t, "ember_kv_snapshots_saved_total") - ssav0 == 1, "snapshot saves");
    ember_buf_free(&before);
    ember_buf_free(&b);
}

static void write_file(const char *path, const char *text) {
    FILE *f = fopen(path, "w");
    if (f) { fputs(text, f); fclose(f); }
}

static void test_identity_and_process_metrics(void) {
    // Real /proc first: on Linux every series must be present and sane.
    ember_buf b = {0};
    ember_metrics_render(&b);
    const char *t = b.ptr ? b.ptr : "";
    CHECK(strstr(t, "ember_build_info{version=\"") != NULL, "build info rendered");
    CHECK(fsample(t, "process_start_time_seconds") > 1.6e9, "start time is wall-clock unix seconds");
    CHECK(fsample(t, "process_resident_memory_bytes") > 0.0, "rss read from /proc");
    CHECK(fsample(t, "process_cpu_seconds_total") >= 0.0, "cpu seconds read from /proc");
    CHECK(fsample(t, "process_open_fds") >= 3.0, "fd count read from /proc");
    CHECK(fsample(t, "process_threads") >= 1.0, "thread count read from /proc");
    ember_buf_free(&b);

    // A fixture proc root: values must come from the files, not the process.
    char root[] = "/tmp/ember-metrics-proc-XXXXXX";
    CHECK(mkdtemp(root) != NULL, "fixture dir");
    char path[512];
    snprintf(path, sizeof(path), "%s/self", root); mkdir(path, 0700);
    snprintf(path, sizeof(path), "%s/self/fd", root); mkdir(path, 0700);
    snprintf(path, sizeof(path), "%s/self/fd/7", root); write_file(path, "");
    snprintf(path, sizeof(path), "%s/self/stat", root);
    // pid (comm with a space) state ppid pgrp session tty tpgid flags minflt
    // cminflt majflt cmajflt utime=100 stime=50 cutime cstime prio nice
    // threads=4 itreal starttime=12345 vsize=999 rss=10 ...
    write_file(path, "42 (ember server) S 1 1 1 0 -1 4194304 0 0 0 0 "
                     "100 50 0 0 20 0 4 0 12345 999 10 0 0 0 0 0 0 0 0 0\n");
    setenv("EMBER_METRICS_PROC_DIR", root, 1);
    ember_buf f = {0};
    ember_metrics_render(&f);
    const char *tf = f.ptr ? f.ptr : "";
    CHECK(fsample(tf, "process_virtual_memory_bytes") == 999.0, "vsize parsed from fixture");
    CHECK(fsample(tf, "process_threads") == 4.0, "threads parsed from fixture");
    CHECK(fsample(tf, "process_open_fds") == 1.0, "fds counted from fixture dir");
    CHECK(fsample(tf, "process_cpu_seconds_total") > 0.0, "cpu ticks converted");
    ember_buf_free(&f);

    // An unparseable stat drops its samples; a missing fd dir drops its own.
    write_file(path, "garbage\n");
    snprintf(path, sizeof(path), "%s/self/fd/7", root); unlink(path);
    snprintf(path, sizeof(path), "%s/self/fd", root); rmdir(path);
    ember_buf u = {0};
    ember_metrics_render(&u);
    const char *tu = u.ptr ? u.ptr : "";
    CHECK(strstr(tu, "process_virtual_memory_bytes") == NULL, "unparseable stat renders nothing");
    CHECK(strstr(tu, "process_open_fds") == NULL, "missing fd dir renders nothing");
    CHECK(strstr(tu, "ember_build_info") != NULL, "identity survives a bad proc");
    ember_buf_free(&u);

    // An absent proc root at all: same rule, and nothing crashes.
    snprintf(path, sizeof(path), "%s/self/stat", root); unlink(path);
    snprintf(path, sizeof(path), "%s/self", root); rmdir(path);
    rmdir(root);
    ember_buf a = {0};
    ember_metrics_render(&a);
    CHECK(strstr(a.ptr ? a.ptr : "", "process_resident_memory_bytes") == NULL,
          "no proc, no process samples");
    ember_buf_free(&a);
    unsetenv("EMBER_METRICS_PROC_DIR");
}

int main(void) {
    printf("ember metrics tests\n");
    test_shape_and_counters();
    test_spec_decline_reasons();
    test_latency_and_vision_series();
    test_unknown_reason_does_not_grow_cardinality();
    test_cumulative_buckets_are_monotonic();
    test_lifecycle_gauges_and_e2e();
    test_responses_outcomes_and_requests();
    test_token_gaps_spec_detail_snapshots();
    test_identity_and_process_metrics();
    printf("──────────────────────────────\n");
    printf("  %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
