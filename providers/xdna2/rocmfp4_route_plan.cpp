#include "rocmfp4_route_plan.h"

#include <cmath>

namespace ember::xdna2 {
namespace {

bool fail(Rocmfp4RoutePlan & plan, std::string * error,
          const char * message) {
    plan = {};
    if (error) *error = message;
    return false;
}

}  // namespace

bool build_rocmfp4_route_plan(const int32_t * selected_ids,
                              const float * router_weights,
                              int n_tokens,
                              int top_k,
                              int n_experts,
                              Rocmfp4RoutePlan & plan,
                              std::string * error) {
    plan = {};
    if (error) error->clear();
    if (!selected_ids || !router_weights || n_tokens <= 0 ||
        n_tokens > kRocmfp4RouteMaxTokens || top_k <= 0 ||
        top_k > kRocmfp4RouteMaxTopK || n_experts <= 0 ||
        n_experts > kRocmfp4RouteMaxExperts) {
        return fail(plan, error, "invalid ROCMFP4 route-plan shape");
    }

    std::array<Rocmfp4ExpertRun, kRocmfp4RouteMaxExperts> grouped{};
    std::array<bool, kRocmfp4RouteMaxExperts> used{};
    for (int expert = 0; expert < n_experts; ++expert)
        grouped[static_cast<size_t>(expert)].expert_id = expert;

    for (int token = 0; token < n_tokens; ++token) {
        std::array<bool, kRocmfp4RouteMaxExperts> token_seen{};
        for (int slot = 0; slot < top_k; ++slot) {
            const size_t index = static_cast<size_t>(token) *
                                     static_cast<size_t>(top_k) +
                                 static_cast<size_t>(slot);
            const int32_t expert = selected_ids[index];
            const float weight = router_weights[index];
            if (expert < 0 || expert >= n_experts)
                return fail(plan, error, "route expert id is out of range");
            if (token_seen[static_cast<size_t>(expert)])
                return fail(plan, error,
                            "route contains a duplicate expert for one token");
            token_seen[static_cast<size_t>(expert)] = true;
            if (!std::isfinite(weight) || weight < 0.0f)
                return fail(plan, error,
                            "route weight must be finite and nonnegative");
            if (weight == 0.0f) continue;

            Rocmfp4ExpertRun & run = grouped[static_cast<size_t>(expert)];
            run.row_weights[static_cast<size_t>(token)] = weight;
            run.active_rows |= uint32_t{1} << static_cast<unsigned>(token);
            used[static_cast<size_t>(expert)] = true;
        }
    }

    plan.n_tokens = n_tokens;
    plan.top_k = top_k;
    plan.runs.reserve(static_cast<size_t>(n_tokens) *
                      static_cast<size_t>(top_k));
    for (int expert = 0; expert < n_experts; ++expert) {
        if (used[static_cast<size_t>(expert)])
            plan.runs.push_back(grouped[static_cast<size_t>(expert)]);
    }
    if (plan.runs.empty())
        return fail(plan, error, "route plan contains no active experts");
    return true;
}

}  // namespace ember::xdna2
