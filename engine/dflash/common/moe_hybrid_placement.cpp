#include "moe_hybrid_placement.h"

#include <algorithm>
#include <limits>

namespace dflash::common {

namespace {

bool placement_valid(const MoeHybridPlacement & p) {
    if (p.n_layer <= 0 || p.n_expert <= 0 || p.n_expert_used <= 0 ||
        p.n_expert_used > p.n_expert ||
        p.hot_counts.size() != static_cast<size_t>(p.n_layer) ||
        p.hot_expert_ids.size() != static_cast<size_t>(p.n_layer)) {
        return false;
    }

    int64_t total = 0;
    std::vector<uint8_t> seen(static_cast<size_t>(p.n_expert));
    for (int il = 0; il < p.n_layer; ++il) {
        const auto & ids = p.hot_expert_ids[static_cast<size_t>(il)];
        if (ids.size() > static_cast<size_t>(p.n_expert) ||
            p.hot_counts[static_cast<size_t>(il)] !=
                static_cast<int>(ids.size())) {
            return false;
        }
        std::fill(seen.begin(), seen.end(), 0);
        for (int32_t id : ids) {
            if (id < 0 || id >= p.n_expert || seen[static_cast<size_t>(id)]) {
                return false;
            }
            seen[static_cast<size_t>(id)] = 1;
        }
        total += static_cast<int64_t>(ids.size());
    }
    return total <= std::numeric_limits<int>::max() &&
           p.total_hot == static_cast<int>(total);
}

}  // namespace

bool MoeHybridPlacement::matches(
        int n_layer_, int n_expert_, int n_expert_used_) const {
    return n_layer == n_layer_ &&
           n_expert == n_expert_ &&
           n_expert_used == n_expert_used_ &&
           placement_valid(*this);
}

bool MoeHybridPlacement::matches(const MoeHybridConfig & cfg) const {
    return matches(cfg.n_layer, cfg.n_expert, cfg.n_expert_used);
}

}  // namespace dflash::common
