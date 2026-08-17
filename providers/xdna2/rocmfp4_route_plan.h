// Build the fixed-shape routed-expert run plan consumed by the Gen8 DSpark
// kernel. The router itself may execute on another processor, but this step is
// deliberately CPU control work: group five token-major top-k selections into
// one resident-weight run per unique expert and carry each row's normalized
// router weight into that run. Keeping this logic independent of XRT gives the
// hardware scheduler a deterministic, GPU-free correctness surface.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace ember::xdna2 {

constexpr int kRocmfp4RouteMaxTokens = 5;
constexpr int kRocmfp4RouteMaxTopK = 6;
constexpr int kRocmfp4RouteMaxExperts = 256;

struct Rocmfp4ExpertRun {
    int32_t expert_id = -1;
    uint32_t active_rows = 0;
    std::array<float, kRocmfp4RouteMaxTokens> row_weights{};
};

struct Rocmfp4RoutePlan {
    int n_tokens = 0;
    int top_k = 0;
    std::vector<Rocmfp4ExpertRun> runs;
};

// selected_ids and router_weights are token-major [n_tokens, top_k]. Expert
// IDs must be unique within a token. Zero-weight selections are validated but
// omitted; all nonzero weights must be finite and positive. Runs are emitted
// in ascending expert-id order so parent-BO sub-buffer selection is stable.
bool build_rocmfp4_route_plan(const int32_t * selected_ids,
                              const float * router_weights,
                              int n_tokens,
                              int top_k,
                              int n_experts,
                              Rocmfp4RoutePlan & plan,
                              std::string * error = nullptr);

}  // namespace ember::xdna2
