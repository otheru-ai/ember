#include <stdio.h>
#include <string.h>
#include <stdlib.h>

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
    ember_metrics_record_generation("stop", 100, 20, 0.05, 0.30, 1.20, false, 0.0, 0);
    ember_metrics_record_generation("length", 5000, 400, 0.25, 3.00, 20.0, true, 0.9, 2);
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

    ember_metrics_record_spec_decline("context");
    ember_metrics_record_spec_decline("context");
    ember_metrics_record_spec_decline("force_ar");
    // Speculation ran: nothing to attribute. Both forms must be ignored rather
    // than folded into "other", which would invent declines that never happened.
    ember_metrics_record_spec_decline(NULL);
    ember_metrics_record_spec_decline("");
    // An engine reason this build does not know must not mint a series.
    ember_metrics_record_spec_decline("some_future_gate");

    ember_buf b = {0};
    ember_metrics_render(&b);
    const char *t = b.ptr ? b.ptr : "";
    CHECK(sample(t, "ember_spec_decode_declined_total{reason=\"context\"}") - ctx0 == 2,
          "declines counted per reason");
    CHECK(sample(t, "ember_spec_decode_declined_total{reason=\"force_ar\"}") - ar0 == 1,
          "second reason counted separately");
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
    const long long itl0  = sample(b0, "ember_inter_token_seconds_count");
    const long long ven0  = sample(b0, "ember_vision_encoder_seconds_count");
    const long long imt0  = sample(b0, "ember_image_tokens_total");
    const long long imc0  = sample(b0, "ember_request_image_count_count");

    // 21 tokens over 2.0s decode, 0.5s queue, 1.0s prefill:
    // ttft = queue + prefill = 1.5; inter-token = 2.0 / (21-1) = 0.1
    ember_metrics_record_generation("stop", 10, 21, 0.5, 1.0, 2.0, false, 0.0, 1);
    // A single token has no interval, so it must not enter inter-token latency:
    // recording decode_s/1 would report a first-token cost as a steady-state one.
    ember_metrics_record_generation("stop", 10, 1, 0.0, 0.5, 0.4, false, 0.0, 0);
    ember_metrics_record_vision_encode(0.25, 131);

    ember_buf b = {0};
    ember_metrics_render(&b);
    const char *t = b.ptr ? b.ptr : "";
    CHECK(sample(t, "ember_time_to_first_token_seconds_count") - ttft0 == 2,
          "ttft observed for every generation");
    CHECK(sample(t, "ember_inter_token_seconds_count") - itl0 == 1,
          "single-token generation excluded from inter-token latency");
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
    ember_metrics_record_generation("a_brand_new_reason", 1, 1, 0.0, 0.0, 0.0, false, 0.0, 0);
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

int main(void) {
    printf("ember metrics tests\n");
    test_shape_and_counters();
    test_spec_decline_reasons();
    test_latency_and_vision_series();
    test_unknown_reason_does_not_grow_cardinality();
    test_cumulative_buckets_are_monotonic();
    printf("──────────────────────────────\n");
    printf("  %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
