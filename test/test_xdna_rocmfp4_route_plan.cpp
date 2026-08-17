#include "rocmfp4_route_plan.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) do {                                        \
    if (cond) { ++g_pass; } else {                                  \
        ++g_fail; std::fprintf(stderr, "FAIL: %s\n", msg);         \
    }                                                               \
} while (0)

int main() {
    using namespace ember::xdna2;

    const int32_t selected[] = {
        7, 2, 9,
        2, 3, 7,
        9, 4, 2,
        5, 7, 4,
        3, 2, 8,
    };
    const float weights[] = {
        0.50f, 0.25f, 0.75f,
        0.20f, 0.30f, 0.40f,
        0.60f, 0.10f, 0.80f,
        0.90f, 0.70f, 0.35f,
        0.45f, 0.55f, 0.65f,
    };
    Rocmfp4RoutePlan plan;
    std::string error;
    CHECK(build_rocmfp4_route_plan(selected, weights, 5, 3, 10,
                                    plan, &error),
          "colliding five-row route builds");
    CHECK(error.empty(), "successful route clears error");
    CHECK(plan.n_tokens == 5 && plan.top_k == 3,
          "plan retains fixed shape");
    CHECK(plan.runs.size() == 7, "collisions group into seven expert runs");
    bool sorted = true;
    for (size_t i = 1; i < plan.runs.size(); ++i)
        sorted = sorted && plan.runs[i - 1].expert_id < plan.runs[i].expert_id;
    CHECK(sorted, "expert runs are resident-offset ordered");

    const Rocmfp4ExpertRun & expert2 = plan.runs[0];
    CHECK(expert2.expert_id == 2, "first active expert is two");
    CHECK(expert2.active_rows == 0x17u,
          "expert two owns rows zero, one, two, and four");
    CHECK(expert2.row_weights[0] == 0.25f &&
          expert2.row_weights[1] == 0.20f &&
          expert2.row_weights[2] == 0.80f &&
          expert2.row_weights[3] == 0.0f &&
          expert2.row_weights[4] == 0.55f,
          "expert two carries token-specific router weights");

    int32_t unique_ids[30];
    float unique_weights[30];
    for (int i = 0; i < 30; ++i) {
        unique_ids[i] = i;
        unique_weights[i] = 0.05f * static_cast<float>((i % 6) + 1);
    }
    CHECK(build_rocmfp4_route_plan(unique_ids, unique_weights, 5, 6, 256,
                                    plan, &error),
          "worst-case thirty-expert route builds");
    CHECK(plan.runs.size() == 30, "worst case emits thirty runs");
    bool one_row_each = true;
    for (const Rocmfp4ExpertRun & run : plan.runs)
        one_row_each = one_row_each && (run.active_rows & (run.active_rows - 1)) == 0;
    CHECK(one_row_each, "worst-case runs each own one row");

    const int32_t zero_ids[] = {1, 2};
    const float zero_weights[] = {0.0f, 0.5f};
    CHECK(build_rocmfp4_route_plan(zero_ids, zero_weights, 1, 2, 3,
                                    plan, &error),
          "zero route weight is safely omitted");
    CHECK(plan.runs.size() == 1 && plan.runs[0].expert_id == 2,
          "only nonzero route becomes a run");

    const int32_t duplicate_ids[] = {4, 4};
    const float duplicate_weights[] = {0.2f, 0.3f};
    CHECK(!build_rocmfp4_route_plan(duplicate_ids, duplicate_weights,
                                     1, 2, 10, plan, &error),
          "duplicate per-token expert is rejected");
    CHECK(plan.runs.empty(), "failed route clears partial runs");

    const int32_t bad_id[] = {256};
    const float good_weight[] = {1.0f};
    CHECK(!build_rocmfp4_route_plan(bad_id, good_weight, 1, 1, 256,
                                     plan, &error),
          "out-of-range expert is rejected");

    const int32_t valid_id[] = {0};
    const float negative[] = {-0.1f};
    CHECK(!build_rocmfp4_route_plan(valid_id, negative, 1, 1, 1,
                                     plan, &error),
          "negative router weight is rejected");
    const float nan_weight[] = {std::numeric_limits<float>::quiet_NaN()};
    CHECK(!build_rocmfp4_route_plan(valid_id, nan_weight, 1, 1, 1,
                                     plan, &error),
          "non-finite router weight is rejected");
    const float all_zero[] = {0.0f};
    CHECK(!build_rocmfp4_route_plan(valid_id, all_zero, 1, 1, 1,
                                     plan, &error),
          "all-zero route is rejected");
    CHECK(!build_rocmfp4_route_plan(nullptr, good_weight, 1, 1, 1,
                                     plan, &error),
          "null route input is rejected");
    CHECK(!build_rocmfp4_route_plan(valid_id, good_weight, 6, 1, 1,
                                     plan, &error),
          "more than five rows is rejected");
    CHECK(!build_rocmfp4_route_plan(valid_id, good_weight, 1, 7, 1,
                                     plan, &error),
          "more than top-six is rejected");

    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
