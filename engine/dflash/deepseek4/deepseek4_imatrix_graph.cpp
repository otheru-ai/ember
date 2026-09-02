#include "deepseek4_imatrix.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <vector>

// The ggml-touching half of the collector. Kept apart from the statistic and
// the .dat writer so those stay host-testable with no ggml link and no stubs:
// a stubbed test proves the stub, not the code.

namespace ds4 {

void ImatrixCollector::register_site(const std::string & name, ggml_tensor * input,
                                     ggml_tensor * ids, int n_in, bool per_slot) {
    if (!input || !ids) return;
    // gallocr reuses the buffer of any tensor that is not a graph output, so
    // without this the reads in drain() would return whatever ran next.
    ggml_set_output(input);
    ggml_set_output(ids);
    sites_.push_back(ImatrixSite{name, input, ids, n_in, per_slot});
}

void ImatrixCollector::drain() {
    std::vector<float>   host_in;
    std::vector<int32_t> host_ids;
    for (const ImatrixSite & s : sites_) {
        const int64_t n_used   = s.ids->ne[0];
        const int64_t n_tokens = s.ids->ne[1];
        if (n_used <= 0 || n_tokens <= 0) continue;

        host_ids.resize((size_t) n_used * (size_t) n_tokens);
        ggml_backend_tensor_get(s.ids, host_ids.data(), 0,
                                sizeof(int32_t) * host_ids.size());

        const size_t n_rows = s.per_slot
            ? (size_t) n_used * (size_t) n_tokens
            : (size_t) n_tokens;
        host_in.resize(n_rows * (size_t) s.n_in);
        ggml_backend_tensor_get(s.input, host_in.data(), 0,
                                sizeof(float) * host_in.size());

        accumulate(s.name, host_in.data(), host_ids.data(), s.n_in,
                   (int) n_used, (int) n_tokens, s.per_slot);
    }
    sites_.clear();
}


}  // namespace ds4
