#include "deepseek4_imatrix.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <vector>

// The ggml-touching half of the collector. Kept apart from the statistic and
// the .dat writer so those stay host-testable with no ggml link and no stubs:
// a stubbed test proves the stub, not the code.

namespace ds4 {

void ImatrixCollector::begin_scope() {
    std::lock_guard<std::mutex> lock(mu_);
    // Anything still registered belongs to a graph that was never drained.
    // Reading it later would dereference a freed context, so drop it.
    sites_.clear();
    scope_ = true;
}

void ImatrixCollector::end_scope() {
    std::lock_guard<std::mutex> lock(mu_);
    sites_.clear();
    scope_ = false;
}

void ImatrixCollector::register_site(const std::string & name, ggml_tensor * input,
                                     ggml_tensor * ids, int n_in, bool per_slot) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!scope_ || !input || !ids) return;
    // gallocr reuses the buffer of any tensor that is not a graph output, so
    // without this the reads in drain() would return whatever ran next.
    //
    // The flag has to go on the BASE tensor, not the view. ggml_gallocr_free_node
    // (ggml-alloc.c:690) tests GGML_TENSOR_FLAG_OUTPUT on the node it is freeing,
    // and the view path (:805-817) calls it with `view_src` -- so flagging a
    // reshape view leaves its base free to be reused, and drain() then reads
    // whatever a later node wrote. ggml_gallocr_alloc_graph's inplace check
    // (:645) does look through view_src, which makes the asymmetry easy to miss.
    // cur_3d is exactly such a reshape view of ffn_normed.
    ggml_set_output(input->view_src ? input->view_src : input);
    ggml_set_output(ids->view_src ? ids->view_src : ids);
    sites_.push_back(ImatrixSite{name, input, ids, n_in, per_slot});
}

void ImatrixCollector::drain() {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<float>   host_in;
    std::vector<int32_t> host_ids;
    size_t drained = 0;
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
        drained++;
    }
    // A chunk is one graph pass that actually contributed rows -- NOT one
    // prompt, since a long prompt is prefilled in several passes. The .dat's
    // last_chunk field is provenance, not a divisor, so this is honest as long
    // as it is written down. Counted here because the first control run
    // collected all 129 entries and then refused to write: nothing ever called
    // end_chunk(), so the empty-matrix guard saw 0 chunks and did its job.
    if (drained) chunks_++;
    sites_.clear();
}


}  // namespace ds4
