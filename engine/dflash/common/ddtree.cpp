// DDTree implementation.
// See ddtree.h for public interface.

#include "ddtree.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace dflash::common {

void extract_draft_topk(const float * logits,
                        int n_positions, int vocab, int K,
                        float * out_log_probs,
                        int32_t * out_token_ids,
                        float temperature) {
    if (!logits || !out_log_probs || !out_token_ids ||
        n_positions <= 0 || vocab <= 0 || K <= 0 || K > vocab) {
        return;
    }
    if (!std::isfinite(temperature) || temperature <= 0.0f)
        temperature = 1.0f;

    struct Entry { float logit; int32_t id; };
    auto cmp_greater = [](const Entry & a, const Entry & b) {
        return a.logit > b.logit;
    };

    const float inv_t = 1.0f / std::max(1e-3f, temperature);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_positions; i++) {
        const float * li = logits + (size_t)i * vocab;
        std::vector<Entry> heap;
        heap.reserve(K);

        float running_max     = -INFINITY;
        float running_sum_exp = 0.0f;
        for (int j = 0; j < vocab; j++) {
            const float l = li[j] * inv_t;

            if (l > running_max) {
                if (running_max > -INFINITY) {
                    running_sum_exp = running_sum_exp * std::exp(running_max - l);
                }
                running_sum_exp += 1.0f;
                running_max = l;
            } else {
                running_sum_exp += std::exp(l - running_max);
            }

            if ((int)heap.size() < K) {
                heap.push_back({l, (int32_t)j});
                std::push_heap(heap.begin(), heap.end(), cmp_greater);
            } else if (l > heap.front().logit) {
                std::pop_heap(heap.begin(), heap.end(), cmp_greater);
                heap.back() = {l, (int32_t)j};
                std::push_heap(heap.begin(), heap.end(), cmp_greater);
            }
        }
        const float log_z = running_max + std::log(running_sum_exp);

        std::sort_heap(heap.begin(), heap.end(), cmp_greater);
        for (int k = 0; k < K; k++) {
            out_log_probs[(size_t)i * K + k] = heap[k].logit - log_z;
            out_token_ids[(size_t)i * K + k] = heap[k].id;
        }
    }
}

DDTree build_ddtree(const float * top_log_probs,
                    const int32_t * top_token_ids,
                    int L, int K, int budget,
                    bool chain_seed) {
    DDTree tree;
    if (!top_log_probs || !top_token_ids || budget <= 0 || L <= 0 || K <= 0) {
        tree.parents.push_back(-1);
        tree.child_maps.emplace_back();
        tree.visibility.assign(1, 1);
        return tree;
    }

    struct HeapEntry {
        float                neg_logw;
        std::vector<int>     ranks;
        int                  parent_index;
        int                  depth;
        int                  rank;
        float                logw;
    };
    struct HeapCmp {
        bool operator()(const HeapEntry & a, const HeapEntry & b) const {
            return a.neg_logw > b.neg_logw;
        }
    };
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapCmp> heap;

    tree.token_ids.reserve(budget);
    tree.depths.reserve(budget);
    tree.parents.reserve(budget + 1);
    tree.parents.push_back(-1);
    tree.child_maps.emplace_back();

    if (chain_seed) {
        const int chain_depth = std::min(L, budget);
        float cum_logw = 0.0f;
        int   prev_idx = 0;
        for (int d = 1; d <= chain_depth; d++) {
            const int32_t tok_id = top_token_ids[(size_t)(d - 1) * K + 0];
            cum_logw += top_log_probs[(size_t)(d - 1) * K + 0];

            const int cur_idx = tree.n_nodes + 1;
            tree.token_ids.push_back(tok_id);
            tree.depths.push_back(d);
            tree.parents.push_back(prev_idx);
            tree.child_maps.emplace_back();
            tree.child_maps[prev_idx][tok_id] = cur_idx;
            tree.n_nodes++;

            if (K > 1) {
                const float sibling_logw = cum_logw
                    - top_log_probs[(size_t)(d - 1) * K + 0]
                    + top_log_probs[(size_t)(d - 1) * K + 1];
                heap.push({
                    -sibling_logw,
                    {1},
                    prev_idx,
                    d,
                    1,
                    sibling_logw,
                });
            }
            prev_idx = cur_idx;
        }
    } else {
        const float root_logw = top_log_probs[0 * K + 0];
        heap.push({
            -root_logw,
            {0},
            0,
            1,
            0,
            root_logw,
        });
    }

    while (!heap.empty() && tree.n_nodes < budget) {
        HeapEntry top = heap.top();
        heap.pop();

        const int    depth_minus_1 = top.depth - 1;
        const int    rank          = top.rank;
        const int32_t token_id     = top_token_ids[(size_t)depth_minus_1 * K + rank];

        const int current_index = tree.n_nodes + 1;
        tree.token_ids.push_back(token_id);
        tree.depths.push_back(top.depth);
        tree.parents.push_back(top.parent_index);
        tree.child_maps.emplace_back();
        tree.child_maps[top.parent_index][token_id] = current_index;
        tree.n_nodes++;

        if (rank + 1 < K) {
            const float sibling_logw = top.logw
                - top_log_probs[(size_t)depth_minus_1 * K + rank]
                + top_log_probs[(size_t)depth_minus_1 * K + rank + 1];
            std::vector<int> sibling_ranks = top.ranks;
            sibling_ranks.back() = rank + 1;
            heap.push({
                -sibling_logw,
                std::move(sibling_ranks),
                top.parent_index,
                top.depth,
                rank + 1,
                sibling_logw,
            });
        }

        if (top.depth < L) {
            const float child_logw = top.logw
                + top_log_probs[(size_t)top.depth * K + 0];
            std::vector<int> child_ranks = top.ranks;
            child_ranks.push_back(0);
            heap.push({
                -child_logw,
                std::move(child_ranks),
                current_index,
                top.depth + 1,
                0,
                child_logw,
            });
        }
    }

    // Build ancestor-only visibility mask.
    const int N = 1 + tree.n_nodes;
    const size_t ns = static_cast<size_t>(N);
    if (ns > std::numeric_limits<size_t>::max() / ns)
        throw std::length_error("DDTree visibility matrix is too large");
    tree.visibility.assign(ns * ns, 0);
    tree.visibility[0 * N + 0] = 1;
    for (int i = 1; i < N; i++) {
        const int p = tree.parents[i];
        for (int j = 0; j < i; j++) {
            tree.visibility[(size_t)i * N + j] = tree.visibility[(size_t)p * N + j];
        }
        tree.visibility[(size_t)i * N + i] = 1;
    }

    return tree;
}

std::vector<int> follow_verified_tree(const DDTree & tree,
                                      const int32_t * posterior,
                                      int & out_next_token,
                                      int * out_node_idx) {
    std::vector<int> accepted;
    if (!posterior || tree.child_maps.empty()) {
        out_next_token = -1;
        if (out_node_idx) *out_node_idx = -1;
        return accepted;
    }
    accepted.reserve(tree.n_nodes + 1);
    accepted.push_back(0);

    int current_index = 0;
    int next_token    = posterior[current_index];
    while (true) {
        const auto & children = tree.child_maps[current_index];
        auto it = children.find(next_token);
        if (it == children.end()) break;
        if (it->second < 0 ||
            static_cast<size_t>(it->second) >= tree.child_maps.size()) {
            break;
        }
        current_index = it->second;
        accepted.push_back(current_index);
        next_token = posterior[current_index];
    }
    out_next_token = next_token;
    if (out_node_idx) *out_node_idx = current_index;
    return accepted;
}

}  // namespace dflash::common
